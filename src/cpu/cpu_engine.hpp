#pragma once

#include <string>
#include <vector>
#include "../core/types.hpp"
#include "../model/gpt2_weights.hpp"

namespace vanguard {

class CpuEngine {
public:
    explicit CpuEngine(const std::string& model_path);
    ~CpuEngine() = default;

    CpuEngine(const CpuEngine&) = delete;
    CpuEngine& operator=(const CpuEngine&) = delete;

    // Resets KV-cache for a new generation sequence
    void resetCache();

    // Runs a single forward step for token_id at sequence position `pos`.
    // Returns a pointer to the computed logits [kGpt2VocabSize].
    const float* forwardStep(int token_id, int pos);

    // Getters for inspecting activation states
    const float* hiddenState() const { return hidden_state_.data(); }
    const float* logits() const { return logits_.data(); }

private:
    Gpt2WeightsLoader loader_;
    const Gpt2Params& params_;

    // Activation Scratchpad Buffers
    std::vector<float> hidden_state_;  // [768]
    std::vector<float> ln1_buf_;       // [768]
    std::vector<float> qkv_buf_;       // [2304]
    std::vector<float> q_buf_;         // [768] (12 heads x 64 head_dim)
    std::vector<float> attn_out_buf_;  // [768]
    std::vector<float> attn_proj_buf_; // [768]
    std::vector<float> ln2_buf_;       // [768]
    std::vector<float> mlp_hidden_buf_;// [3072]
    std::vector<float> mlp_proj_buf_;  // [768]
    std::vector<float> logits_;        // [50257]

    // Persistent KV Cache for CPU:
    // Shape: [12 layers, 1024 pos, 768 dim]
    std::vector<float> key_cache_;   // [12 * 1024 * 768]
    std::vector<float> value_cache_; // [12 * 1024 * 768]
};

} // namespace vanguard
