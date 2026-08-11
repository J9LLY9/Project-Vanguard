#include "cuda_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace vanguard {

namespace {

constexpr float kSafetyCheckThreshold = 1e4f;

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

__global__ void encoder_forward(float* hidden_state, int token_id, int position,
                                const float* wte, const float* wpe,
                                int embed_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < embed_dim) {
        size_t token_offset = static_cast<size_t>(token_id) * embed_dim + i;
        size_t pos_offset = static_cast<size_t>(position) * embed_dim + i;
        hidden_state[i] = wte[token_offset] + wpe[pos_offset];
    }
}

__global__ void layernorm_forward(const float* x, const float* weight,
                                  const float* bias, float* out,
                                  int embed_dim) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    float sum = 0.0f;
    for (int i = tid; i < embed_dim; i += blockDim.x) {
        sum += x[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / static_cast<float>(embed_dim);
    __syncthreads();

    float var_sum = 0.0f;
    for (int i = tid; i < embed_dim; i += blockDim.x) {
        float diff = x[i] - mean;
        var_sum += diff * diff;
    }
    sdata[tid] = var_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float var = sdata[0] / static_cast<float>(embed_dim);
    float inv_std = rsqrtf(var + 1e-5f);

    for (int i = tid; i < embed_dim; i += blockDim.x) {
        float norm = (x[i] - mean) * inv_std;
        out[i] = norm * weight[i] + bias[i];
    }
}

__global__ void linear_forward(const float* x, const float* weight,
                               const float* bias, float* out, int in_dim,
                               int out_dim) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < out_dim) {
        float sum = bias ? bias[col] : 0.0f;
        for (int i = 0; i < in_dim; ++i) {
            sum += x[i] * weight[static_cast<size_t>(i) * out_dim + col];
        }
        out[col] = sum;
    }
}

__global__ void permute_qkv(const float* qkv_output, float* q_buf, float* k_buf,
                            float* v_buf, int num_heads, int head_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int embed_dim = num_heads * head_dim;
    if (i < embed_dim) {
        q_buf[i] = qkv_output[i];
        k_buf[i] = qkv_output[embed_dim + i];
        v_buf[i] = qkv_output[2 * embed_dim + i];
    }
}

// Stores current token's projected K and V into the persistent KV cache
__global__ void update_kv_cache_kernel(float* key_cache, float* value_cache,
                                       const float* k_buf, const float* v_buf,
                                       int layer, int position, int embed_dim, int ctx_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < embed_dim) {
        size_t offset = (static_cast<size_t>(layer) * ctx_len + position) * embed_dim + i;
        key_cache[offset] = k_buf[i];
        value_cache[offset] = v_buf[i];
    }
}

// Computes attention scores between current query Q and stored keys in key_cache for 0..position
__global__ void kv_attention_scores_forward(const float* q_buf, const float* key_cache,
                                             float* attention_matrix,
                                             int layer, int num_heads, int head_dim,
                                             int position, int ctx_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_heads * ctx_len;

    if (idx < total_elements) {
        int head = idx / ctx_len;
        int key_pos = idx % ctx_len;

        if (key_pos > position) {
            attention_matrix[idx] = -1e9f;
        } else {
            const float* q_head = q_buf + head * head_dim;
            size_t past_k_offset = (static_cast<size_t>(layer) * ctx_len + key_pos) * (num_heads * head_dim) + head * head_dim;
            const float* k_head_past = key_cache + past_k_offset;

            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += q_head[d] * k_head_past[d];
            }
            attention_matrix[idx] = dot / sqrtf(static_cast<float>(head_dim));
        }
    }
}

__global__ void attention_softmax_forward(float* attention_matrix,
                                           int max_seq_len, int position) {
    extern __shared__ float sdata[];
    int head = blockIdx.x;
    int tid = threadIdx.x;

    float* row = attention_matrix + head * max_seq_len;

    float max_val = -1e9f;
    for (int i = tid; i <= position; i += blockDim.x) {
        if (row[i] > max_val) max_val = row[i];
    }
    sdata[tid] = max_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }
    float block_max = sdata[0];
    __syncthreads();

    float exp_sum = 0.0f;
    for (int i = tid; i <= position; i += blockDim.x) {
        float val = expf(row[i] - block_max);
        row[i] = val;
        exp_sum += val;
    }
    for (int i = position + 1 + tid; i < max_seq_len; i += blockDim.x) {
        row[i] = 0.0f;
    }
    sdata[tid] = exp_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float total_sum = sdata[0];
    float inv_sum = 1.0f / total_sum;

    for (int i = tid; i <= position; i += blockDim.x) {
        row[i] *= inv_sum;
    }
}

// Computes weighted sum of values over stored value_cache for 0..position
__global__ void kv_attention_value_forward(const float* attention_matrix,
                                           const float* value_cache, float* head_output,
                                           int layer, int num_heads, int head_dim,
                                           int ctx_len, int position) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int embed_dim = num_heads * head_dim;

    if (i < embed_dim) {
        int head = i / head_dim;
        int dim_idx = i % head_dim;

        const float* attn_row = attention_matrix + head * ctx_len;
        
        float acc = 0.0f;
        for (int t = 0; t <= position; ++t) {
            size_t past_v_offset = (static_cast<size_t>(layer) * ctx_len + t) * embed_dim + head * head_dim + dim_idx;
            float alpha = attn_row[t];
            acc += alpha * value_cache[past_v_offset];
        }
        head_output[i] = acc;
    }
}

__global__ void residual_add(const float* a, const float* b, float* out, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = a[i] + b[i];
    }
}

__global__ void gelu_forward(float* x, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float val = x[i];
        float cube = val * val * val;
        float inner = 0.7978845608028654f * (val + 0.044715f * cube);
        x[i] = 0.5f * val * (1.0f + tanhf(inner));
    }
}

__global__ void classifier_head_forward(const float* hidden_state,
                                         const float* wte, float* logits,
                                         int embed_dim, int vocab_size) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < vocab_size) {
        const float* wte_row = wte + static_cast<size_t>(v) * embed_dim;
        float dot = 0.0f;
        for (int i = 0; i < embed_dim; ++i) {
            dot += hidden_state[i] * wte_row[i];
        }
        logits[v] = dot;
    }
}

} // namespace

CudaEngine::CudaEngine(const std::string& model_path)
    : loader_(model_path), host_params_(loader_.params()) {

    kv_cache_mgr_ = std::make_unique<KVCacheManager>();

    const size_t weights_bytes = kGpt2ExpectedParamCount * sizeof(float);

    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&device_weights_), weights_bytes));
    cudaCheck(cudaMemcpy(device_weights_, loader_.data(), weights_bytes, cudaMemcpyHostToDevice));

    LayoutGpt2Params(device_params_, device_weights_);

    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&hidden_state_), kGpt2Embed * sizeof(float)));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&qkv_output_), kGpt2QkvOut * sizeof(float)));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&q_buf_), kGpt2Embed * sizeof(float)));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&k_buf_), kGpt2Embed * sizeof(float)));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&v_buf_), kGpt2Embed * sizeof(float)));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&attention_matrix_), static_cast<size_t>(kGpt2NumHeads) * kGpt2CtxLen * sizeof(float)));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&head_output_), kGpt2Embed * sizeof(float)));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&ln1_buf_), kGpt2Embed * sizeof(float)));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&attn_proj_buf_), kGpt2Embed * sizeof(float)));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&ln2_buf_), kGpt2Embed * sizeof(float)));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&mlp_hidden_buf_), kGpt2MlpHidden * sizeof(float)));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&mlp_proj_buf_), kGpt2Embed * sizeof(float)));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&logits_), kGpt2VocabSize * sizeof(float)));
}

CudaEngine::~CudaEngine() {
    if (device_weights_) cudaFree(device_weights_);
    if (hidden_state_) cudaFree(hidden_state_);
    if (qkv_output_) cudaFree(qkv_output_);
    if (q_buf_) cudaFree(q_buf_);
    if (k_buf_) cudaFree(k_buf_);
    if (v_buf_) cudaFree(v_buf_);
    if (attention_matrix_) cudaFree(attention_matrix_);
    if (head_output_) cudaFree(head_output_);
    if (ln1_buf_) cudaFree(ln1_buf_);
    if (attn_proj_buf_) cudaFree(attn_proj_buf_);
    if (ln2_buf_) cudaFree(ln2_buf_);
    if (mlp_hidden_buf_) cudaFree(mlp_hidden_buf_);
    if (mlp_proj_buf_) cudaFree(mlp_proj_buf_);
    if (logits_) cudaFree(logits_);
}

void CudaEngine::resetCache() {
    if (kv_cache_mgr_) {
        kv_cache_mgr_->reset();
    }
}

const float* CudaEngine::prefill(const std::vector<int>& prompt_tokens) {
    resetCache();
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        forward_step(prompt_tokens[i], static_cast<int>(i));
    }
    return logits_;
}

void CudaEngine::forward_step(int token_id, int position) {
    if (token_id < 0 || token_id >= kGpt2VocabSize) {
        throw std::runtime_error("CudaEngine::forward_step: token_id out of range");
    }
    if (position < 0 || position >= kGpt2CtxLen) {
        throw std::runtime_error("CudaEngine::forward_step: position out of range");
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

        const int ln1_block_size = 256;
        const size_t ln1_shmem = ln1_block_size * sizeof(float);

        layernorm_forward<<<1, ln1_block_size, ln1_shmem>>>(
            hidden_state_, w.ln_1_w, w.ln_1_b, ln1_buf_, kGpt2Embed);

        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());

        const int qkv_block_size = 256;
        const int qkv_grid_size = (kGpt2QkvOut + qkv_block_size - 1) / qkv_block_size;

        linear_forward<<<qkv_grid_size, qkv_block_size>>>(
            ln1_buf_, w.attn_c_attn_w, w.attn_c_attn_b, qkv_output_, kGpt2Embed, kGpt2QkvOut);

        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());

        const int permute_block_size = 256;
        const int permute_grid_size = (kGpt2Embed + permute_block_size - 1) / permute_block_size;

        permute_qkv<<<permute_grid_size, permute_block_size>>>(
            qkv_output_, q_buf_, k_buf_, v_buf_, kGpt2NumHeads, kGpt2HeadDim);

        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());

        // --- Update Persistent KV Cache for current layer and position ---
        const int kv_block_size = 256;
        const int kv_grid_size = (kGpt2Embed + kv_block_size - 1) / kv_block_size;

        update_kv_cache_kernel<<<kv_grid_size, kv_block_size>>>(
            kv_cache_mgr_->key_cache(), kv_cache_mgr_->value_cache(),
            k_buf_, v_buf_, layer, position, kGpt2Embed, kGpt2CtxLen);

        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());

        // --- Compute Attention Scores against KV Cache (0..position) ---
        const int attn_block_size = 256;
        const int attn_total = kGpt2NumHeads * kGpt2CtxLen;
        const int attn_grid_size = (attn_total + attn_block_size - 1) / attn_block_size;

        kv_attention_scores_forward<<<attn_grid_size, attn_block_size>>>(
            q_buf_, kv_cache_mgr_->key_cache(), attention_matrix_,
            layer, kGpt2NumHeads, kGpt2HeadDim, position, kGpt2CtxLen);

        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());

        // --- Softmax Over Attention Scores ---
        const int softmax_block_size = 256;
        const size_t softmax_shmem = softmax_block_size * sizeof(float);

        attention_softmax_forward<<<kGpt2NumHeads, softmax_block_size, softmax_shmem>>>(
            attention_matrix_, kGpt2CtxLen, position);

        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());

        // --- Weighted Sum Over Stored Values in KV Cache ---
        const int value_block_size = 256;
        const int value_grid_size = (kGpt2Embed + value_block_size - 1) / value_block_size;

        kv_attention_value_forward<<<value_grid_size, value_block_size>>>(
            attention_matrix_, kv_cache_mgr_->value_cache(), head_output_,
            layer, kGpt2NumHeads, kGpt2HeadDim, kGpt2CtxLen, position);

        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());

        const int attn_proj_block_size = 256;
        const int attn_proj_grid_size = (kGpt2Embed + attn_proj_block_size - 1) / attn_proj_block_size;

        linear_forward<<<attn_proj_grid_size, attn_proj_block_size>>>(
            head_output_, w.attn_c_proj_w, w.attn_c_proj_b, attn_proj_buf_, kGpt2Embed, kGpt2Embed);

        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());

        const int res1_block_size = 256;
        const int res1_grid_size = (kGpt2Embed + res1_block_size - 1) / res1_block_size;

        residual_add<<<res1_grid_size, res1_block_size>>>(
            hidden_state_, attn_proj_buf_, hidden_state_, kGpt2Embed);

        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());

        const int ln2_block_size = 256;
        const size_t ln2_shmem = ln2_block_size * sizeof(float);

        layernorm_forward<<<1, ln2_block_size, ln2_shmem>>>(
            hidden_state_, w.ln_2_w, w.ln_2_b, ln2_buf_, kGpt2Embed);

        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());

        const int fc_block_size = 256;
        const int fc_grid_size = (kGpt2MlpHidden + fc_block_size - 1) / fc_block_size;

        linear_forward<<<fc_grid_size, fc_block_size>>>(
            ln2_buf_, w.mlp_c_fc_w, w.mlp_c_fc_b, mlp_hidden_buf_, kGpt2Embed, kGpt2MlpHidden);

        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());

        const int gelu_block_size = 256;
        const int gelu_grid_size = (kGpt2MlpHidden + gelu_block_size - 1) / gelu_block_size;

        gelu_forward<<<gelu_grid_size, gelu_block_size>>>(mlp_hidden_buf_, kGpt2MlpHidden);

        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());

        const int mlp_proj_block_size = 256;
        const int mlp_proj_grid_size = (kGpt2Embed + mlp_proj_block_size - 1) / mlp_proj_block_size;

        linear_forward<<<mlp_proj_grid_size, mlp_proj_block_size>>>(
            mlp_hidden_buf_, w.mlp_c_proj_w, w.mlp_c_proj_b, mlp_proj_buf_, kGpt2MlpHidden, kGpt2Embed);

        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());

        const int res2_block_size = 256;
        const int res2_grid_size = (kGpt2Embed + res2_block_size - 1) / res2_block_size;

        residual_add<<<res2_grid_size, res2_block_size>>>(
            hidden_state_, mlp_proj_buf_, hidden_state_, kGpt2Embed);

        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());

        if (layer == 0 || layer == 1) {
            PrintSafetyCheck(hidden_state_, layer, kGpt2Embed);
        }
    }

    const int lnf_block_size = 256;
    const size_t lnf_shmem = lnf_block_size * sizeof(float);

    layernorm_forward<<<1, lnf_block_size, lnf_shmem>>>(
        hidden_state_, device_params_.ln_f_w, device_params_.ln_f_b, hidden_state_, kGpt2Embed);

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    const int logits_block_size = 256;
    const int logits_grid_size = (kGpt2VocabSize + logits_block_size - 1) / logits_block_size;

    classifier_head_forward<<<logits_grid_size, logits_block_size>>>(
        hidden_state_, device_params_.wte, logits_, kGpt2Embed, kGpt2VocabSize);

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
}

} // namespace vanguard
