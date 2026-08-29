#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "../../core/types.hpp"
#include "../../core/model/gpt2_weights.hpp"
#include "kv_cache.hpp"

namespace vanguard {

inline void CudaCheckImpl(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[CUDA ERROR] ") + file + ":" +
                                  std::to_string(line) + " -- " +
                                  cudaGetErrorString(err));
    }
}

#define cudaCheck(err) ::vanguard::CudaCheckImpl((err), __FILE__, __LINE__)

class CudaEngine {
public:
    explicit CudaEngine(const std::string& model_path);
    ~CudaEngine();

    CudaEngine(const CudaEngine&) = delete;
    CudaEngine& operator=(const CudaEngine&) = delete;

    const Gpt2Params& deviceParams() const { return device_params_; }
    const Gpt2Params& hostParams() const { return host_params_; }

    float* hiddenState() const { return hidden_state_; }
    float* logits() const { return logits_; }

    // Resets stored KV Cache for a new sequence
    void resetCache();

    // Runs forward step for a single token at position `position`.
    // Uses KV-cache to attend to past context tokens.
    void forward_step(int token_id, int position);

    // Prefill phase: processes prompt tokens in sequence and populates KV cache.
    // Returns logits pointer for the last prompt token.
    const float* prefill(const std::vector<int>& prompt_tokens);

private:
    Gpt2WeightsLoader loader_;
    const Gpt2Params& host_params_;
    std::unique_ptr<KVCacheManager> kv_cache_mgr_;

    // Device pointers
    float* device_weights_ = nullptr;
    float* hidden_state_ = nullptr;
    float* qkv_output_ = nullptr;
    float* q_buf_ = nullptr;
    float* k_buf_ = nullptr;
    float* v_buf_ = nullptr;
    float* attention_matrix_ = nullptr;
    float* head_output_ = nullptr;
    float* attn_proj_buf_ = nullptr;
    float* ln2_buf_ = nullptr;
    float* mlp_hidden_buf_ = nullptr;
    float* ln1_buf_ = nullptr;
    float* mlp_proj_buf_ = nullptr;
    float* logits_ = nullptr;
    Gpt2Params device_params_{};
    cudaStream_t stream_ = nullptr;
};

} // namespace vanguard
