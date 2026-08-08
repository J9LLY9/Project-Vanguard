#include "cpu_engine.hpp"
#include "cpu_kernels.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace vanguard {

CpuEngine::CpuEngine(const std::string& model_path)
    : loader_(model_path), params_(loader_.params()) {
    hidden_state_.resize(kGpt2Embed);
    ln1_buf_.resize(kGpt2Embed);
    qkv_buf_.resize(kGpt2QkvOut);
    q_buf_.resize(kGpt2Embed);
    attn_out_buf_.resize(kGpt2Embed);
    attn_proj_buf_.resize(kGpt2Embed);
    ln2_buf_.resize(kGpt2Embed);
    mlp_hidden_buf_.resize(kGpt2MlpHidden);
    mlp_proj_buf_.resize(kGpt2Embed);
    logits_.resize(kGpt2VocabSize);

    key_cache_.resize(static_cast<size_t>(kGpt2NumLayers) * kGpt2CtxLen * kGpt2Embed, 0.0f);
    value_cache_.resize(static_cast<size_t>(kGpt2NumLayers) * kGpt2CtxLen * kGpt2Embed, 0.0f);
}

void CpuEngine::resetCache() {
    std::fill(key_cache_.begin(), key_cache_.end(), 0.0f);
    std::fill(value_cache_.begin(), value_cache_.end(), 0.0f);
}

const float* CpuEngine::forwardStep(int token_id, int pos) {
    if (token_id < 0 || token_id >= kGpt2VocabSize) {
        throw std::runtime_error("CpuEngine: token_id out of bounds: " + std::to_string(token_id));
    }
    if (pos < 0 || pos >= kGpt2CtxLen) {
        throw std::runtime_error("CpuEngine: pos out of bounds: " + std::to_string(pos));
    }

    // 1. Initial Embedding Lookup (wte + wpe)
    cpu::EmbeddingLookup(hidden_state_.data(), params_.wte, params_.wpe, token_id, pos, kGpt2Embed);

    const float scale = 1.0f / std::sqrt(static_cast<float>(kGpt2HeadDim));

    // 2. Transformer Blocks Loop
    for (int l = 0; l < kGpt2NumLayers; ++l) {
        const Gpt2LayerWeights& layer = params_.h[l];

        // --- LayerNorm 1 ---
        cpu::LayerNorm(ln1_buf_.data(), hidden_state_.data(), layer.ln_1_w, layer.ln_1_b, kGpt2Embed);

        // --- QKV Projection ---
        cpu::MatMul(qkv_buf_.data(), ln1_buf_.data(), layer.attn_c_attn_w, layer.attn_c_attn_b, kGpt2Embed, kGpt2QkvOut);

        // Store K and V into KV Cache for current layer `l` and position `pos`
        // qkv_buf layout: Q (0..767), K (768..1535), V (1536..2303)
        const float* k_current = qkv_buf_.data() + kGpt2Embed;
        const float* v_current = qkv_buf_.data() + 2 * kGpt2Embed;

        size_t cache_layer_offset = (static_cast<size_t>(l) * kGpt2CtxLen + pos) * kGpt2Embed;
        std::memcpy(key_cache_.data() + cache_layer_offset, k_current, kGpt2Embed * sizeof(float));
        std::memcpy(value_cache_.data() + cache_layer_offset, v_current, kGpt2Embed * sizeof(float));

        // --- Multi-Head Self-Attention ---
        const float* q_current = qkv_buf_.data();

        for (int h = 0; h < kGpt2NumHeads; ++h) {
            const float* q_head = q_current + h * kGpt2HeadDim;

            // Attention scores vector for positions 0..pos
            std::vector<float> scores(pos + 1, 0.0f);
            for (int t = 0; t <= pos; ++t) {
                size_t past_k_offset = (static_cast<size_t>(l) * kGpt2CtxLen + t) * kGpt2Embed + h * kGpt2HeadDim;
                const float* k_head_past = key_cache_.data() + past_k_offset;

                float dot = 0.0f;
                for (int d = 0; d < kGpt2HeadDim; ++d) {
                    dot += q_head[d] * k_head_past[d];
                }
                scores[t] = dot * scale;
            }

            // Softmax over valid past positions
            cpu::Softmax(scores.data(), pos + 1);

            // Weighted sum over V
            float* head_out = attn_out_buf_.data() + h * kGpt2HeadDim;
            std::fill_n(head_out, kGpt2HeadDim, 0.0f);

            for (int t = 0; t <= pos; ++t) {
                size_t past_v_offset = (static_cast<size_t>(l) * kGpt2CtxLen + t) * kGpt2Embed + h * kGpt2HeadDim;
                const float* v_head_past = value_cache_.data() + past_v_offset;
                float alpha = scores[t];

                for (int d = 0; d < kGpt2HeadDim; ++d) {
                    head_out[d] += alpha * v_head_past[d];
                }
            }
        }

        // --- Attention Output Projection ---
        cpu::MatMul(attn_proj_buf_.data(), attn_out_buf_.data(), layer.attn_c_proj_w, layer.attn_c_proj_b, kGpt2Embed, kGpt2Embed);

        // --- Residual Add 1 ---
        cpu::ResidualAdd(hidden_state_.data(), hidden_state_.data(), attn_proj_buf_.data(), kGpt2Embed);

        // --- LayerNorm 2 ---
        cpu::LayerNorm(ln2_buf_.data(), hidden_state_.data(), layer.ln_2_w, layer.ln_2_b, kGpt2Embed);

        // --- MLP Up-Projection ---
        cpu::MatMul(mlp_hidden_buf_.data(), ln2_buf_.data(), layer.mlp_c_fc_w, layer.mlp_c_fc_b, kGpt2Embed, kGpt2MlpHidden);

        // --- GELU ---
        cpu::GELUInPlace(mlp_hidden_buf_.data(), kGpt2MlpHidden);

        // --- MLP Down-Projection ---
        cpu::MatMul(mlp_proj_buf_.data(), mlp_hidden_buf_.data(), layer.mlp_c_proj_w, layer.mlp_c_proj_b, kGpt2MlpHidden, kGpt2Embed);

        // --- Residual Add 2 ---
        cpu::ResidualAdd(hidden_state_.data(), hidden_state_.data(), mlp_proj_buf_.data(), kGpt2Embed);
    }

    // 3. Final LayerNorm (ln_f)
    cpu::LayerNorm(hidden_state_.data(), hidden_state_.data(), params_.ln_f_w, params_.ln_f_b, kGpt2Embed);

    // 4. Classifier Head (LM Head weight tied to wte: [50257, 768])
    cpu::ClassifierHead(logits_.data(), hidden_state_.data(), params_.wte, kGpt2Embed, kGpt2VocabSize);

    return logits_.data();
}

} // namespace vanguard
