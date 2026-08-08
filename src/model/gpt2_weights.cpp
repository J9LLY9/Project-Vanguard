#include "gpt2_weights.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <stdexcept>

namespace vanguard {

void LayoutGpt2Params(Gpt2Params& params, float* base) {
    float* cursor = base;
    const float* const end = base + kGpt2ExpectedParamCount;

    auto take = [&](size_t count) -> float* {
        if (cursor + count > end) {
            throw std::runtime_error("LayoutGpt2Params: pointer assignment overran buffer");
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
        layer.attn_c_attn_w = take(static_cast<size_t>(kGpt2Embed) * kGpt2QkvOut);
        layer.attn_c_attn_b = take(kGpt2QkvOut);
        layer.attn_c_proj_w = take(static_cast<size_t>(kGpt2Embed) * kGpt2Embed);
        layer.attn_c_proj_b = take(kGpt2Embed);

        layer.ln_2_w = take(kGpt2Embed);
        layer.ln_2_b = take(kGpt2Embed);
        layer.mlp_c_fc_w = take(static_cast<size_t>(kGpt2Embed) * kGpt2MlpHidden);
        layer.mlp_c_fc_b = take(kGpt2MlpHidden);
        layer.mlp_c_proj_w = take(static_cast<size_t>(kGpt2MlpHidden) * kGpt2Embed);
        layer.mlp_c_proj_b = take(kGpt2Embed);
    }

    params.ln_f_w = take(kGpt2Embed);
    params.ln_f_b = take(kGpt2Embed);

    if (cursor != end) {
        throw std::runtime_error("LayoutGpt2Params: parameters count mismatch");
    }
}

Gpt2WeightsLoader::Gpt2WeightsLoader(const std::string& filepath) {
    fd_ = ::open(filepath.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error("Gpt2WeightsLoader failed to open '" + filepath +
                                 "': " + std::strerror(errno));
    }

    struct stat sb{};
    if (::fstat(fd_, &sb) != 0) {
        ::close(fd_);
        throw std::runtime_error("Gpt2WeightsLoader failed to stat '" + filepath + "'");
    }

    file_size_ = static_cast<size_t>(sb.st_size);
    const size_t expected_bytes = kGpt2ExpectedParamCount * sizeof(float);
    if (file_size_ != expected_bytes) {
        ::close(fd_);
        throw std::runtime_error("Gpt2WeightsLoader file size mismatch: expected " +
                                 std::to_string(expected_bytes) + " bytes, got " +
                                 std::to_string(file_size_));
    }

    void* addr = ::mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (addr == MAP_FAILED) {
        ::close(fd_);
        throw std::runtime_error("Gpt2WeightsLoader mmap failed for '" + filepath + "'");
    }

    host_data_ = static_cast<float*>(addr);
    LayoutGpt2Params(params_, host_data_);
}

Gpt2WeightsLoader::~Gpt2WeightsLoader() {
    if (host_data_ != nullptr && host_data_ != MAP_FAILED) {
        ::munmap(host_data_, file_size_);
    }
    if (fd_ >= 0) {
        ::close(fd_);
    }
}

} // namespace vanguard
