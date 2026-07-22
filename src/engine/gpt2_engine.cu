#include "gpt2_engine.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

namespace vanguard {

void LayoutGpt2Params(Gpt2Params& params, float* base) {
    float* cursor = base;
    const float* const end = base + Gpt2Engine::kExpectedParamCount;

    auto take = [&](size_t count) -> float* {
        if (cursor + count > end) {
            throw std::runtime_error(
                "LayoutGpt2Params: pointer assignment overran buffer");
        }
        float* p = cursor;
        cursor += count;
        return p;
    };

    params.wte = take(static_cast<size_t>(kGpt2VocabSize) * kGpt2Embed);
    params.wpe = take(static_cast<size_t>(kGpt2CtxLen) * kGpt2Embed);

    for (int i = 0; i < kGpt2NumLayers; ++i) {
        Gpt2LayerWeights& layer = params.h[i];

        layer.ln_1_w = take(kGpt2Embed);
        layer.ln_1_b = take(kGpt2Embed);

        layer.attn_c_attn_w =
            take(static_cast<size_t>(kGpt2Embed) * kGpt2QkvOut);
        layer.attn_c_attn_b = take(kGpt2QkvOut);

        layer.attn_c_proj_w =
            take(static_cast<size_t>(kGpt2Embed) * kGpt2Embed);
        layer.attn_c_proj_b = take(kGpt2Embed);

        layer.ln_2_w = take(kGpt2Embed);
        layer.ln_2_b = take(kGpt2Embed);

        layer.mlp_c_fc_w =
            take(static_cast<size_t>(kGpt2Embed) * kGpt2MlpHidden);
        layer.mlp_c_fc_b = take(kGpt2MlpHidden);

        layer.mlp_c_proj_w =
            take(static_cast<size_t>(kGpt2MlpHidden) * kGpt2Embed);
        layer.mlp_c_proj_b = take(kGpt2Embed);
    }

    params.ln_f_w = take(kGpt2Embed);
    params.ln_f_b = take(kGpt2Embed);

    if (cursor != end) {
        throw std::runtime_error(
            "LayoutGpt2Params: pointer assignment did not consume exactly "
            "kExpectedParamCount floats -- layout is out of sync with the "
            "export script");
    }
}

namespace {

// hidden_state[i] = wte[token_id, i] + wpe[position, i]
// One thread per embedding dimension (kGpt2Embed = 768 total threads).
__global__ void encoder_forward(float* hidden_state, int token_id,
                                 int position, const float* wte,
                                 const float* wpe, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C) return;

    const float* wte_row = wte + static_cast<long long>(token_id) * C;
    const float* wpe_row = wpe + static_cast<long long>(position) * C;

    hidden_state[i] = wte_row[i] + wpe_row[i];
}

} // namespace

Gpt2Engine::Gpt2Engine(const std::string& path) {
    mapFile(path);
    LayoutGpt2Params(host_params_, host_data_);
    uploadToDevice();
}

Gpt2Engine::~Gpt2Engine() {
    if (host_data_ != nullptr && host_data_ != MAP_FAILED) {
        munmap(host_data_, file_size_);
    }
    if (fd_ >= 0) {
        close(fd_);
    }
    if (device_weights_ != nullptr) {
        cudaFree(device_weights_);
    }
    if (hidden_state_ != nullptr) {
        cudaFree(hidden_state_);
    }
}

void Gpt2Engine::mapFile(const std::string& path) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error("Gpt2Engine: failed to open '" + path +
                                  "': " + std::strerror(errno));
    }

    struct stat st {};
    if (fstat(fd_, &st) != 0) {
        close(fd_);
        fd_ = -1;
        throw std::runtime_error("Gpt2Engine: fstat failed on '" + path +
                                  "': " + std::strerror(errno));
    }
    file_size_ = static_cast<size_t>(st.st_size);

    const size_t expected_bytes = kExpectedParamCount * sizeof(float);
    if (file_size_ != expected_bytes) {
        close(fd_);
        fd_ = -1;
        throw std::runtime_error(
            "Gpt2Engine: size mismatch for '" + path + "' -- expected " +
            std::to_string(expected_bytes) + " bytes (" +
            std::to_string(kExpectedParamCount) +
            " float32 params, GPT-2 124M exactly) but file is " +
            std::to_string(file_size_) +
            " bytes. Wrong file, or export script version drifted from "
            "this engine's layout?");
    }

    void* mapped = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped == MAP_FAILED) {
        close(fd_);
        fd_ = -1;
        throw std::runtime_error(std::string("Gpt2Engine: mmap failed: ") +
                                  std::strerror(errno));
    }

    // Weights are read sequentially per forward pass; hint the kernel to
    // prefetch rather than page fault one 4KB block at a time.
    madvise(mapped, file_size_, MADV_WILLNEED);

    host_data_ = static_cast<float*>(mapped);
}

void Gpt2Engine::uploadToDevice() {
    const size_t weights_bytes = kExpectedParamCount * sizeof(float);

    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&device_weights_),
                          weights_bytes));
    cudaCheck(cudaMemcpy(device_weights_, host_data_, weights_bytes,
                          cudaMemcpyHostToDevice));

    // Same offset walk as the host mapping, rooted at the device base
    // pointer -- these addresses are only ever formed here, never
    // dereferenced on the host side.
    LayoutGpt2Params(device_params_, device_weights_);

    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&hidden_state_),
                          kHiddenStateCount * sizeof(float)));
}

void Gpt2Engine::forward_step(int token_id, int position) {
    if (token_id < 0 || token_id >= kGpt2VocabSize) {
        throw std::runtime_error(
            "Gpt2Engine::forward_step: token_id " + std::to_string(token_id) +
            " out of range [0, " + std::to_string(kGpt2VocabSize) + ")");
    }
    if (position < 0 || position >= kGpt2CtxLen) {
        throw std::runtime_error(
            "Gpt2Engine::forward_step: position " + std::to_string(position) +
            " out of range [0, " + std::to_string(kGpt2CtxLen) + ")");
    }

    const int block_size = 256;
    const int grid_size = (kGpt2Embed + block_size - 1) / block_size;

    encoder_forward<<<grid_size, block_size>>>(
        hidden_state_, token_id, position, device_params_.wte,
        device_params_.wpe, kGpt2Embed);

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
}

} // namespace vanguard
