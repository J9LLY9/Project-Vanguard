#pragma once

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace vanguard {

// ---------------------------------------------------------------------
// GPT-2 (124M) Architecture Constants
// ---------------------------------------------------------------------
constexpr int kGpt2VocabSize = 50257;
constexpr int kGpt2CtxLen    = 1024;
constexpr int kGpt2Embed     = 768;
constexpr int kGpt2NumLayers = 12;
constexpr int kGpt2QkvOut    = 3 * kGpt2Embed; // 2304, combined q/k/v projection
constexpr int kGpt2MlpHidden = 4 * kGpt2Embed; // 3072
constexpr int kGpt2NumHeads  = 12;
constexpr int kGpt2HeadDim   = kGpt2Embed / kGpt2NumHeads; // 64

// Total Float32 parameter count for GPT-2 124M
constexpr size_t kGpt2ExpectedParamCount = 124'439'808;

// Precision enum for future optimization phases
enum class Precision {
    FP32,
    FP16,
    BF16,
    INT8
};

// Error handling helper
inline void Check(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error("[Vanguard Error] " + message);
    }
}

} // namespace vanguard
