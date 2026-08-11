#include "kv_cache.hpp"
#include "cuda_engine.hpp"

namespace vanguard {

KVCacheManager::KVCacheManager() {
    total_elements_ = static_cast<size_t>(kGpt2NumLayers) * kGpt2CtxLen * kGpt2Embed;

    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&key_cache_), total_elements_ * sizeof(float)));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&value_cache_), total_elements_ * sizeof(float)));

    reset();
}

KVCacheManager::~KVCacheManager() {
    if (key_cache_) cudaFree(key_cache_);
    if (value_cache_) cudaFree(value_cache_);
}

void KVCacheManager::reset() {
    cudaCheck(cudaMemset(key_cache_, 0, total_elements_ * sizeof(float)));
    cudaCheck(cudaMemset(value_cache_, 0, total_elements_ * sizeof(float)));
}

} // namespace vanguard
