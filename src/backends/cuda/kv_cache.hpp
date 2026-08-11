#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include "../../core/types.hpp"

namespace vanguard {

// ---------------------------------------------------------------------
// KVCacheManager: Manages persistent Key and Value VRAM buffers for
// autoregressive Transformer inference.
//
// Shape: [kGpt2NumLayers, kGpt2NumHeads, kGpt2CtxLen, kGpt2HeadDim]
// Total size per cache: 12 * 12 * 1024 * 64 * sizeof(float) = 37.7 MB
// ---------------------------------------------------------------------
class KVCacheManager {
public:
    KVCacheManager();
    ~KVCacheManager();

    KVCacheManager(const KVCacheManager&) = delete;
    KVCacheManager& operator=(const KVCacheManager&) = delete;

    // Resets cache memory (clears stored keys/values)
    void reset();

    float* key_cache() const { return key_cache_; }
    float* value_cache() const { return value_cache_; }

    size_t size_bytes() const { return total_elements_ * sizeof(float); }

private:
    float* key_cache_ = nullptr;
    float* value_cache_ = nullptr;
    size_t total_elements_ = 0;
};

} // namespace vanguard
