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

// output[j] = dot(input, weight[:, j]) + bias[j]
// weight is [in_features, out_features] (Conv1D export layout, not
// transposed), so column j is the strided slice weight[i * out_features + j]
// for i in [0, in_features). One thread per output element; threads in a
// warp share consecutive j, so the load of weight[i * out_features + j] is
// coalesced on every step of the loop.
__global__ void linear_forward(const float* input, const float* weight,
                                const float* bias, float* output,
                                int in_features, int out_features) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= out_features) return;

    float acc = 0.0f;
    for (int i = 0; i < in_features; ++i) {
        acc += input[i] * weight[i * out_features + j];
    }
    output[j] = acc + bias[j];
}

// Splits the flat qkv_output (Q|K|V concatenated, each kGpt2Embed wide)
// into three separate per-head buffers. HF's Conv1D produces each of the
// Q/K/V sections already head-major -- i.e. the first head_dim values of a
// section are head 0's, the next head_dim are head 1's, etc. -- so
// splitting into heads only requires locating the right section, not
// reordering within it.
//
// Each output buffer holds one token's [Head, Sequence_Position,
// Head_Dimension] slice; since a single forward_step covers exactly one
// sequence position, that layout collapses to [Head, Head_Dimension]
// here (head * head_dim + hd), which is where the KV-cache's per-position
// stride would be added once this engine tracks more than one token.
// One thread per (head, head_dim) pair.
__global__ void permute_qkv(const float* qkv, float* q_buf, float* k_buf,
                             float* v_buf, int num_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int per_section = num_heads * head_dim;
    if (idx >= per_section) return;

    int head = idx / head_dim;
    int hd = idx % head_dim;
    int flat = head * head_dim + hd; // == idx, spelled out for clarity

    q_buf[idx] = qkv[0 * per_section + flat];
    k_buf[idx] = qkv[1 * per_section + flat];
    v_buf[idx] = qkv[2 * per_section + flat];
}

// Score(head, key_pos) = (Q_head . K_head) / sqrt(head_dim). attention_matrix
// is laid out [Head, max_seq_len] so the same buffer can be reused, one row
// per head, as later forward_step calls fill in more of the sequence.
//
// q_buf_/k_buf_ hold only the current token (no KV-cache yet -- see
// permute_qkv's comment), so key_pos == query_pos is the only column with a
// real key to score against this step:
//   - key_pos > query_pos is a genuine causal violation (a token attending
//     to the future) and is masked to -1e9, per GPT-2's causal mask.
//   - key_pos < query_pos would be a legitimate past token, but this engine
//     doesn't persist past K vectors yet, so there's nothing to score
//     against; it's masked to -1e9 too as a placeholder until a KV-cache
//     lands, not because attending to the past is actually disallowed.
//
// One thread per (head, key_pos) pair, covering the full max_seq_len width
// of the row.
__global__ void attention_scores_forward(const float* q, const float* k,
                                          float* attention_matrix,
                                          int num_heads, int head_dim,
                                          int query_pos, int max_seq_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * max_seq_len;
    if (idx >= total) return;

    int head = idx / max_seq_len;
    int key_pos = idx % max_seq_len;

    if (key_pos > query_pos) {
        attention_matrix[idx] = -1e9f; // future: real causal mask
        return;
    }
    if (key_pos < query_pos) {
        attention_matrix[idx] = -1e9f; // past: no KV-cache yet, TODO
        return;
    }

    const float* q_head = q + head * head_dim;
    const float* k_head = k + head * head_dim;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        dot += q_head[d] * k_head[d];
    }
    attention_matrix[idx] = dot * (1.0f / sqrtf(64.0f));
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
    if (qkv_output_ != nullptr) {
        cudaFree(qkv_output_);
    }
    if (q_buf_ != nullptr) {
        cudaFree(q_buf_);
    }
    if (k_buf_ != nullptr) {
        cudaFree(k_buf_);
    }
    if (v_buf_ != nullptr) {
        cudaFree(v_buf_);
    }
    if (attention_matrix_ != nullptr) {
        cudaFree(attention_matrix_);
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

    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&qkv_output_),
                          kQkvOutputCount * sizeof(float)));

    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&q_buf_),
                          kHeadBufCount * sizeof(float)));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&k_buf_),
                          kHeadBufCount * sizeof(float)));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&v_buf_),
                          kHeadBufCount * sizeof(float)));

    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&attention_matrix_),
                          kAttnMatrixCount * sizeof(float)));
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

    // TODO: ln_1 (LayerNorm) isn't implemented yet -- this runs the c_attn
    // projection directly on the raw embedding sum instead of the
    // normalized hidden state. Swap `hidden_state_` for the ln_1 output
    // once that kernel exists.
    const int qkv_block_size = 256;
    const int qkv_grid_size =
        (kGpt2QkvOut + qkv_block_size - 1) / qkv_block_size;

    linear_forward<<<qkv_grid_size, qkv_block_size>>>(
        hidden_state_, device_params_.h[0].attn_c_attn_w,
        device_params_.h[0].attn_c_attn_b, qkv_output_, kGpt2Embed,
        kGpt2QkvOut);

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    const int permute_block_size = 256;
    const int permute_grid_size =
        (kGpt2Embed + permute_block_size - 1) / permute_block_size;

    permute_qkv<<<permute_grid_size, permute_block_size>>>(
        qkv_output_, q_buf_, k_buf_, v_buf_, kGpt2NumHeads, kGpt2HeadDim);

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    const int attn_block_size = 256;
    const int attn_total = kGpt2NumHeads * kGpt2CtxLen;
    const int attn_grid_size = (attn_total + attn_block_size - 1) / attn_block_size;

    attention_scores_forward<<<attn_grid_size, attn_block_size>>>(
        q_buf_, k_buf_, attention_matrix_, kGpt2NumHeads, kGpt2HeadDim,
        position, kGpt2CtxLen);

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
}

} // namespace vanguard
