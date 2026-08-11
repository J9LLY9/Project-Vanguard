#include "gpt2_engine.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace vanguard {

namespace {

// Threshold for forward_step()'s layer 0/1 safety check: real GPT-2
// activations after a residual add stay in the single-to-low-double
// digits, so anything past this is a strong signal of a numerical
// stability regression (e.g. a missing LayerNorm), not normal variance.
constexpr float kSafetyCheckThreshold = 1e4f;

// Copies hidden_state_ back to the host and prints its max |value| --
// called only for layer 0 and layer 1 so it doesn't add host<->device
// round trips to every layer of every forward_step().
void PrintSafetyCheck(const float* hidden_state, int layer, int embed_dim) {
    std::vector<float> h(embed_dim);
    cudaCheck(cudaMemcpy(h.data(), hidden_state, embed_dim * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float max_abs = 0.0f;
    for (float v : h) {
        max_abs = std::max(max_abs, std::fabs(v));
    }

    std::printf("[safety check] layer %d: hidden_state max|value| = %.6f\n",
                layer, max_abs);
    if (max_abs > kSafetyCheckThreshold) {
        std::printf(
            "  WARNING: numerical instability at layer %d -- max|value| "
            "%.3e exceeds safety threshold %.3e\n",
            layer, max_abs, kSafetyCheckThreshold);
    }
}

} // namespace

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

// Turns each head's row of raw scores into a probability distribution, in
// place: prob(key_pos) = exp(score(key_pos) - row_max) / sum(exp(... )).
// Subtracting the row max before exponentiating (the "safe softmax" trick)
// keeps exp() away from overflow regardless of how large the raw scores
// get -- without it, a score like 82 would exponentiate to inf and poison
// the whole row.
//
// Causal mask: key_pos > query_pos is the future and must never receive
// attention weight, so the reduction only ever runs over [0, query_pos]
// (valid_len = query_pos + 1) -- future columns are zeroed directly rather
// than relying on -1e9 underflowing to ~0. That -1e9 fill still covers
// key_pos < query_pos (no KV-cache yet, see attention_scores_forward), so
// it also softmaxes away to ~0 within the valid range, leaving all weight
// on key_pos == query_pos until a cache lands.
//
// One block per head; threads cooperate via shared-memory tree reductions
// for the row max and the row sum.
__global__ void attention_softmax_forward(float* attention_matrix,
                                           int max_seq_len, int query_pos) {
    extern __shared__ float sdata[];
    int head = blockIdx.x;
    int tid = threadIdx.x;
    float* row = attention_matrix + static_cast<size_t>(head) * max_seq_len;
    int valid_len = query_pos + 1;

    float thread_max = -INFINITY;
    for (int i = tid; i < valid_len; i += blockDim.x) {
        thread_max = fmaxf(thread_max, row[i]);
    }
    sdata[tid] = thread_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    float thread_sum = 0.0f;
    for (int i = tid; i < valid_len; i += blockDim.x) {
        float e = expf(row[i] - row_max);
        row[i] = e; // stash the numerator; normalize once row_sum is known
        thread_sum += e;
    }
    sdata[tid] = thread_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();

    for (int i = tid; i < max_seq_len; i += blockDim.x) {
        row[i] = (i < valid_len) ? (row[i] / row_sum) : 0.0f;
    }
}

// head_output(head, hd) = sum over key_pos of prob(head, key_pos) * V(key_pos,
// head, hd). v_buf_ only ever holds the current token's Value (no KV-cache
// yet, same limitation as attention_scores_forward/permute_qkv), and
// attention_softmax_forward leaves all probability mass on key_pos ==
// query_pos in that same no-cache regime, so the sum reduces exactly to a
// single term: probs[query_pos] * v_buf. This will need to become a real
// loop over a V-cache once one exists.
//
// One thread per (head, head_dim) pair -- num_heads * head_dim ==
// kGpt2Embed == 768 threads total.
__global__ void attention_value_forward(const float* attention_matrix,
                                         const float* v_buf, float* head_output,
                                         int num_heads, int head_dim,
                                         int max_seq_len, int query_pos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * head_dim;
    if (idx >= total) return;

    int head = idx / head_dim;
    const float* probs = attention_matrix + static_cast<size_t>(head) * max_seq_len;

    head_output[idx] = probs[query_pos] * v_buf[idx];
}

// out[i] = a[i] + b[i]. `out` is allowed to alias `a` or `b` for in-place
// accumulation into the residual stream -- each thread only ever touches
// index i, so aliasing causes no read/write hazard between threads.
// One thread per element.
__global__ void residual_add(const float* a, const float* b, float* out,
                              int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] + b[i];
}

// GPT-2's LayerNorm: normalize x to zero mean/unit variance over its C
// features, then rescale/shift by the learned weight/bias.
// y[i] = (x[i] - mean) / sqrt(var + eps) * weight[i] + bias[i]
// Single block; threads cooperate via shared-memory tree reductions for
// the mean and the variance, the same pattern as attention_softmax_forward's
// row reductions.
__global__ void layernorm_forward(const float* x, const float* weight,
                                   const float* bias, float* out, int C) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    constexpr float eps = 1e-5f;

    float thread_sum = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        thread_sum += x[i];
    }
    sdata[tid] = thread_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    float mean = sdata[0] / C;
    __syncthreads();

    float thread_sqsum = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        float d = x[i] - mean;
        thread_sqsum += d * d;
    }
    sdata[tid] = thread_sqsum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    float var = sdata[0] / C;
    __syncthreads();

    float inv_std = rsqrtf(var + eps);
    for (int i = tid; i < C; i += blockDim.x) {
        out[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

// GPT-2's tanh-approximation GELU, applied in place:
// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// One thread per element.
__global__ void gelu_forward(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float v = x[i];
    float inner = 0.7978845608f * (v + 0.044715f * v * v * v);
    x[i] = 0.5f * v * (1.0f + tanhf(inner));
}

// logits[v] = dot(hidden_state, wte_row_v). See the call site's comment for
// why this can't just be linear_forward with different dimensions: wte is
// row-major [vocab_size, embed_dim] (an nn.Embedding table, same layout
// encoder_forward indexes into), the transpose of linear_forward's
// [in_features, out_features] Conv1D convention, and lm_head has no bias.
// One thread per vocab entry.
__global__ void classifier_head_forward(const float* hidden_state,
                                         const float* wte, float* logits,
                                         int embed_dim, int vocab_size) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= vocab_size) return;

    const float* row = wte + static_cast<size_t>(v) * embed_dim;
    float acc = 0.0f;
    for (int i = 0; i < embed_dim; ++i) {
        acc += hidden_state[i] * row[i];
    }
    logits[v] = acc;
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
    if (head_output_ != nullptr) {
        cudaFree(head_output_);
    }
    if (ln1_buf_ != nullptr) {
        cudaFree(ln1_buf_);
    }
    if (attn_proj_buf_ != nullptr) {
        cudaFree(attn_proj_buf_);
    }
    if (ln2_buf_ != nullptr) {
        cudaFree(ln2_buf_);
    }
    if (mlp_hidden_buf_ != nullptr) {
        cudaFree(mlp_hidden_buf_);
    }
    if (mlp_proj_buf_ != nullptr) {
        cudaFree(mlp_proj_buf_);
    }
    if (logits_ != nullptr) {
        cudaFree(logits_);
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

    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&head_output_),
                          kHeadOutputCount * sizeof(float)));

    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&ln1_buf_),
                          kLn1Count * sizeof(float)));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&attn_proj_buf_),
                          kAttnProjCount * sizeof(float)));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&ln2_buf_),
                          kLn2Count * sizeof(float)));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&mlp_hidden_buf_),
                          kMlpHiddenCount * sizeof(float)));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&mlp_proj_buf_),
                          kMlpProjCount * sizeof(float)));

    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&logits_),
                          kLogitsCount * sizeof(float)));
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

  for (int layer = 0; layer < kGpt2NumLayers; ++layer) {
    const Gpt2LayerWeights& w = device_params_.h[layer];

    // ln_1: hidden_state_ (the residual stream, un-normalized) -> ln1_buf_.
    // Without this, c_attn ran on a value that keeps growing every layer
    // (nothing ever renormalizes the residual stream itself), compounding
    // into a numerical blow-up by the later layers.
    const int ln1_block_size = 256;
    const size_t ln1_shmem = ln1_block_size * sizeof(float);

    layernorm_forward<<<1, ln1_block_size, ln1_shmem>>>(
        hidden_state_, w.ln_1_w, w.ln_1_b, ln1_buf_, kGpt2Embed);

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    const int qkv_block_size = 256;
    const int qkv_grid_size =
        (kGpt2QkvOut + qkv_block_size - 1) / qkv_block_size;

    linear_forward<<<qkv_grid_size, qkv_block_size>>>(
        ln1_buf_, w.attn_c_attn_w, w.attn_c_attn_b, qkv_output_, kGpt2Embed,
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

    // One block per head; block size must be a power of two for the
    // shared-memory tree reductions inside the kernel.
    const int softmax_block_size = 256;
    const size_t softmax_shmem = softmax_block_size * sizeof(float);

    attention_softmax_forward<<<kGpt2NumHeads, softmax_block_size,
                                 softmax_shmem>>>(attention_matrix_,
                                                   kGpt2CtxLen, position);

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    const int value_block_size = 256;
    const int value_grid_size =
        (kGpt2Embed + value_block_size - 1) / value_block_size;

    attention_value_forward<<<value_grid_size, value_block_size>>>(
        attention_matrix_, v_buf_, head_output_, kGpt2NumHeads, kGpt2HeadDim,
        kGpt2CtxLen, position);

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    // attn.c_proj: head_output (768) -> attn_proj_buf_ (768).
    const int attn_proj_block_size = 256;
    const int attn_proj_grid_size =
        (kGpt2Embed + attn_proj_block_size - 1) / attn_proj_block_size;

    linear_forward<<<attn_proj_grid_size, attn_proj_block_size>>>(
        head_output_, w.attn_c_proj_w, w.attn_c_proj_b, attn_proj_buf_,
        kGpt2Embed, kGpt2Embed);

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    // Residual add: ln_1 only ever wrote to ln1_buf_, so hidden_state_
    // still holds this layer's un-normalized input (the residual stream)
    // at this point -- this folds attn.c_proj's output back into it in
    // place, per GPT-2's pre-norm architecture (LayerNorm output feeds the
    // sublayer, but the residual add uses the pre-LayerNorm value).
    const int res1_block_size = 256;
    const int res1_grid_size =
        (kGpt2Embed + res1_block_size - 1) / res1_block_size;

    residual_add<<<res1_grid_size, res1_block_size>>>(
        hidden_state_, attn_proj_buf_, hidden_state_, kGpt2Embed);

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    // ln_2: hidden_state_ (the post-attention residual) -> ln2_buf_.
    const int ln2_block_size = 256;
    const size_t ln2_shmem = ln2_block_size * sizeof(float);

    layernorm_forward<<<1, ln2_block_size, ln2_shmem>>>(
        hidden_state_, w.ln_2_w, w.ln_2_b, ln2_buf_, kGpt2Embed);

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    // MLP up-proj: ln2_buf_ (768) -> mlp_hidden_buf_ (3072).
    const int fc_block_size = 256;
    const int fc_grid_size = (kGpt2MlpHidden + fc_block_size - 1) / fc_block_size;

    linear_forward<<<fc_grid_size, fc_block_size>>>(
        ln2_buf_, w.mlp_c_fc_w, w.mlp_c_fc_b, mlp_hidden_buf_, kGpt2Embed,
        kGpt2MlpHidden);

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    // GELU, in place on mlp_hidden_buf_.
    const int gelu_block_size = 256;
    const int gelu_grid_size =
        (kGpt2MlpHidden + gelu_block_size - 1) / gelu_block_size;

    gelu_forward<<<gelu_grid_size, gelu_block_size>>>(mlp_hidden_buf_,
                                                       kGpt2MlpHidden);

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    // MLP down-proj: mlp_hidden_buf_ (3072) -> mlp_proj_buf_ (768).
    const int mlp_proj_block_size = 256;
    const int mlp_proj_grid_size =
        (kGpt2Embed + mlp_proj_block_size - 1) / mlp_proj_block_size;

    linear_forward<<<mlp_proj_grid_size, mlp_proj_block_size>>>(
        mlp_hidden_buf_, w.mlp_c_proj_w, w.mlp_c_proj_b, mlp_proj_buf_,
        kGpt2MlpHidden, kGpt2Embed);

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    // Final residual add: fold the MLP's output back into hidden_state_,
    // which still holds the post-attention residual -- this leaves
    // hidden_state_ holding this layer's complete block output, which the
    // next iteration's c_attn projection reads as its input.
    const int res2_block_size = 256;
    const int res2_grid_size =
        (kGpt2Embed + res2_block_size - 1) / res2_block_size;

    residual_add<<<res2_grid_size, res2_block_size>>>(
        hidden_state_, mlp_proj_buf_, hidden_state_, kGpt2Embed);

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    // Safety check: layers 0 and 1 are where a numerical-stability
    // regression (e.g. a missing/broken LayerNorm) first becomes visible,
    // well before it compounds into something like a 10^10 blow-up by the
    // last layer.
    if (layer == 0 || layer == 1) {
        PrintSafetyCheck(hidden_state_, layer, kGpt2Embed);
    }
  } // for layer

    // Final LayerNorm (ln_f), in place -- nothing downstream needs the
    // pre-ln_f residual, unlike ln_2 which had to preserve it for the MLP's
    // residual add.
    const int lnf_block_size = 256;
    const size_t lnf_shmem = lnf_block_size * sizeof(float);

    layernorm_forward<<<1, lnf_block_size, lnf_shmem>>>(
        hidden_state_, device_params_.ln_f_w, device_params_.ln_f_b,
        hidden_state_, kGpt2Embed);

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    // Classifier head: logits[v] = dot(hidden_state, wte[v, :]). wte is
    // weight-tied to the classifier, but unlike c_attn/c_fc/c_proj (Conv1D
    // layout, [in_features, out_features]) it's a plain nn.Embedding table
    // stored [vocab_size, embed_dim] row-major -- the same per-row layout
    // encoder_forward already reads from. That's the transpose of what
    // linear_forward's indexing assumes, and GPT-2's lm_head has no bias,
    // so this can't reuse linear_forward unmodified; classifier_head_forward
    // below is structurally the same one-thread-per-output dot product, just
    // wired to wte's actual layout and without a bias term.
    const int logits_block_size = 256;
    const int logits_grid_size =
        (kGpt2VocabSize + logits_block_size - 1) / logits_block_size;

    classifier_head_forward<<<logits_grid_size, logits_block_size>>>(
        hidden_state_, device_params_.wte, logits_, kGpt2Embed,
        kGpt2VocabSize);

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
}

} // namespace vanguard
