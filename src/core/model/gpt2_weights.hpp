#pragma once

#include <cstddef>
#include <string>
#include "../types.hpp"

namespace vanguard {

struct Gpt2LayerWeights {
    float* ln_1_w;        // [768]
    float* ln_1_b;        // [768]
    float* attn_c_attn_w; // [768, 2304]
    float* attn_c_attn_b; // [2304]
    float* attn_c_proj_w; // [768, 768]
    float* attn_c_proj_b; // [768]
    float* ln_2_w;        // [768]
    float* ln_2_b;        // [768]
    float* mlp_c_fc_w;    // [768, 3072]
    float* mlp_c_fc_b;    // [3072]
    float* mlp_c_proj_w;  // [3072, 768]
    float* mlp_c_proj_b;  // [768]
};

struct Gpt2Params {
    float* wte; // [50257, 768] token embeddings (tied to lm_head)
    float* wpe; // [1024, 768]  position embeddings
    Gpt2LayerWeights h[kGpt2NumLayers];
    float* ln_f_w; // [768]
    float* ln_f_b; // [768]
};

// Maps a contiguous float array of kGpt2ExpectedParamCount floats to Gpt2Params pointers.
void LayoutGpt2Params(Gpt2Params& params, float* base);

// Loads mmap'd GPT-2 weight binary file from host path.
class Gpt2WeightsLoader {
public:
    explicit Gpt2WeightsLoader(const std::string& filepath);
    ~Gpt2WeightsLoader();

    Gpt2WeightsLoader(const Gpt2WeightsLoader&) = delete;
    Gpt2WeightsLoader& operator=(const Gpt2WeightsLoader&) = delete;

    float* data() const { return host_data_; }
    size_t size_bytes() const { return file_size_; }
    const Gpt2Params& params() const { return params_; }

private:
    float* host_data_ = nullptr;
    size_t file_size_ = 0;
    int fd_ = -1;
    Gpt2Params params_{};
};

} // namespace vanguard
