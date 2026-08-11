#include "cpu_kernels.hpp"

#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace vanguard {
namespace cpu {

void EmbeddingLookup(float* out, const float* wte, const float* wpe, int token_id, int pos, int embed_dim) {
    const float* token_emb = wte + static_cast<size_t>(token_id) * embed_dim;
    const float* pos_emb   = wpe + static_cast<size_t>(pos) * embed_dim;

    for (int i = 0; i < embed_dim; ++i) {
        out[i] = token_emb[i] + pos_emb[i];
    }
}

void LayerNorm(float* out, const float* in, const float* w, const float* b, int size, float eps) {
    float mean = 0.0f;
    for (int i = 0; i < size; ++i) {
        mean += in[i];
    }
    mean /= static_cast<float>(size);

    float var = 0.0f;
    for (int i = 0; i < size; ++i) {
        float diff = in[i] - mean;
        var += diff * diff;
    }
    var /= static_cast<float>(size);

    float inv_std = 1.0f / std::sqrt(var + eps);

    for (int i = 0; i < size; ++i) {
        float norm = (in[i] - mean) * inv_std;
        out[i] = norm * (w ? w[i] : 1.0f) + (b ? b[i] : 0.0f);
    }
}

void MatMul(float* out, const float* in, const float* weight, const float* bias, int in_dim, int out_dim) {
    for (int j = 0; j < out_dim; ++j) {
        float sum = bias ? bias[j] : 0.0f;
        for (int i = 0; i < in_dim; ++i) {
            sum += in[i] * weight[static_cast<size_t>(i) * out_dim + j];
        }
        out[j] = sum;
    }
}

void ClassifierHead(float* logits, const float* hidden_state, const float* wte, int embed_dim, int vocab_size) {
    for (int v = 0; v < vocab_size; ++v) {
        const float* wte_row = wte + static_cast<size_t>(v) * embed_dim;
        float dot = 0.0f;
        for (int i = 0; i < embed_dim; ++i) {
            dot += hidden_state[i] * wte_row[i];
        }
        logits[v] = dot;
    }
}

void GELU(float* out, const float* in, int size) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f; // sqrt(2/pi)
    constexpr float kCoeff       = 0.044715f;

    for (int i = 0; i < size; ++i) {
        float x = in[i];
        float cube = x * x * x;
        float inner = kSqrt2OverPi * (x + kCoeff * cube);
        out[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
}

void GELUInPlace(float* data, int size) {
    GELU(data, data, size);
}

void Softmax(float* data, int size) {
    if (size <= 0) return;

    float max_val = data[0];
    for (int i = 1; i < size; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        data[i] = std::exp(data[i] - max_val);
        sum += data[i];
    }

    float inv_sum = 1.0f / sum;
    for (int i = 0; i < size; ++i) {
        data[i] *= inv_sum;
    }
}

void ResidualAdd(float* out, const float* in1, const float* in2, int size) {
    for (int i = 0; i < size; ++i) {
        out[i] = in1[i] + in2[i];
    }
}

} // namespace cpu
} // namespace vanguard
