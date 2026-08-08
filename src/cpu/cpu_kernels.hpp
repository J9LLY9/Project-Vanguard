#pragma once

#include <cstddef>
#include <vector>
#include "../core/types.hpp"

namespace vanguard {
namespace cpu {

// Embedding Lookup: out[i] = wte[token_id, i] + wpe[pos, i]
void EmbeddingLookup(float* out, const float* wte, const float* wpe, int token_id, int pos, int embed_dim);

// Layer Normalization: y = (x - mean) / sqrt(var + eps) * w + b
void LayerNorm(float* out, const float* in, const float* w, const float* b, int size, float eps = 1e-5f);

// Matrix Multiplication with Bias for Conv1D layout weights: y = x @ W + b
// x: [1, in_dim], W: [in_dim, out_dim] (row-major), b: [out_dim], y: [1, out_dim]
void MatMul(float* out, const float* in, const float* weight, const float* bias, int in_dim, int out_dim);

// Classifier Head Dot Product: logits[v] = dot(hidden_state, wte[v, :])
// hidden_state: [embed_dim], wte: [vocab_size, embed_dim], logits: [vocab_size]
void ClassifierHead(float* logits, const float* hidden_state, const float* wte, int embed_dim, int vocab_size);

// Elementwise GELU activation (GPT-2 approximation formulation)
void GELU(float* out, const float* in, int size);

// Elementwise GELU in-place
void GELUInPlace(float* data, int size);

// Numerically stable Softmax: x_i = exp(x_i - max) / sum(exp(x_j - max))
void Softmax(float* data, int size);

// Residual Addition: out[i] = in1[i] + in2[i]
void ResidualAdd(float* out, const float* in1, const float* in2, int size);

} // namespace cpu
} // namespace vanguard
