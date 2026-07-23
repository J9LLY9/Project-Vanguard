#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace vanguard {

// ---------------------------------------------------------------------
// CUDA error checking -- wrap every CUDA runtime call in cudaCheck(...).
// Throws (rather than exit()s) so mmap/cudaFree cleanup in destructors
// still runs during unwinding.
// ---------------------------------------------------------------------

inline void CudaCheckImpl(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[CUDA ERROR] ") + file + ":" +
                                  std::to_string(line) + " -- " +
                                  cudaGetErrorString(err));
    }
}

} // namespace vanguard

#define cudaCheck(err) ::vanguard::CudaCheckImpl((err), __FILE__, __LINE__)

namespace vanguard {

// ---------------------------------------------------------------------
// GPT-2 (124M) architecture constants
// ---------------------------------------------------------------------
constexpr int kGpt2VocabSize = 50257;
constexpr int kGpt2CtxLen    = 1024;
constexpr int kGpt2Embed     = 768;
constexpr int kGpt2NumLayers = 12;
constexpr int kGpt2QkvOut    = 3 * kGpt2Embed; // 2304, combined q/k/v projection
constexpr int kGpt2MlpHidden = 4 * kGpt2Embed; // 3072
constexpr int kGpt2NumHeads  = 12;
constexpr int kGpt2HeadDim   = kGpt2Embed / kGpt2NumHeads; // 64

// Weights for a single transformer block. HuggingFace's GPT2 uses Conv1D
// (not nn.Linear) for c_attn / c_proj / c_fc, which stores its weight as
// [in_features, out_features] -- the transpose of a typical Linear layer.
// These pointers preserve that layout; matmuls should be y = x @ W + b.
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
    float* wte; // [50257, 768] token embeddings (weight-tied to lm_head)
    float* wpe; // [1024, 768]  position embeddings
    Gpt2LayerWeights h[kGpt2NumLayers];
    float* ln_f_w; // [768]
    float* ln_f_b; // [768]
};

// Carves `base` (Gpt2Engine::kExpectedParamCount contiguous floats --
// host or device) into per-tensor pointers via pointer arithmetic, in
// the exact order export_gpt2_weights.py writes tensors: wte, wpe, then
// per-layer (ln_1, attn.c_attn, attn.c_proj, ln_2, mlp.c_fc, mlp.c_proj)
// x 12, ln_f. `base` need not be dereferenceable on the calling side
// (e.g. a CUDA device pointer computed on the host) -- this only forms
// addresses. Shared by the host mmap layout and the device layout so the
// two can never drift apart. Throws std::runtime_error if the walk
// doesn't consume exactly kExpectedParamCount floats.
void LayoutGpt2Params(Gpt2Params& params, float* base);

// ---------------------------------------------------------------------
// Gpt2Engine: mmaps the weight file, uploads it to VRAM, and runs
// forward-pass kernels against the GPU copy.
// ---------------------------------------------------------------------
class Gpt2Engine {
public:
    // Total float32 parameters expected in the file (wte, wpe, 12
    // blocks, ln_f) -- matches GPT-2 124M exactly.
    static constexpr size_t kExpectedParamCount = 124'439'808;

    // forward_step() writes one token's embedding: kGpt2Embed floats.
    static constexpr size_t kHiddenStateCount = static_cast<size_t>(kGpt2Embed);

    // forward_step() also writes the QKV projection: kGpt2QkvOut floats.
    static constexpr size_t kQkvOutputCount = static_cast<size_t>(kGpt2QkvOut);

    // forward_step() also splits qkv_output into q_buf/k_buf/v_buf, each
    // holding this one token's [Head, Head_Dimension] slice: kGpt2Embed
    // floats (kGpt2NumHeads * kGpt2HeadDim).
    static constexpr size_t kHeadBufCount = static_cast<size_t>(kGpt2Embed);

    // forward_step() also fills one column of attention_matrix, which is
    // laid out [Head, max_seq_len] (kGpt2CtxLen doubling as max_seq_len)
    // so the buffer only needs allocating once, up front, and later steps
    // can fill in more of each head's row without reallocating.
    static constexpr size_t kAttnMatrixCount =
        static_cast<size_t>(kGpt2NumHeads) * kGpt2CtxLen;

    // Mmaps `path`, lays out host pointers, then allocates VRAM and
    // uploads the entire weight blob in one cudaMemcpy. Throws
    // std::runtime_error on any open/mmap/size/CUDA failure.
    explicit Gpt2Engine(const std::string& path);
    ~Gpt2Engine();

    Gpt2Engine(const Gpt2Engine&) = delete;
    Gpt2Engine& operator=(const Gpt2Engine&) = delete;

    // Device pointers -- only valid passed to CUDA APIs / dereferenced
    // from device code, never dereferenced on the host.
    const Gpt2Params& deviceParams() const { return device_params_; }

    // Host pointers -- into the mmap'd file, read-only.
    const Gpt2Params& hostParams() const { return host_params_; }

    // Device pointer, kHiddenStateCount floats: the output of the most
    // recent forward_step().
    float* hiddenState() const { return hidden_state_; }

    // Device pointer, kQkvOutputCount floats: the c_attn (QKV) projection
    // of the most recent forward_step().
    float* qkvOutput() const { return qkv_output_; }

    // Device pointers, kHeadBufCount floats each: this token's Q/K/V split
    // into [Head, Head_Dimension] layout (head-major, 64 floats per head).
    float* qBuf() const { return q_buf_; }
    float* kBuf() const { return k_buf_; }
    float* vBuf() const { return v_buf_; }

    // Device pointer, kAttnMatrixCount floats: [Head, max_seq_len] causally
    // masked attention scores. Only column `position` of each head's row is
    // valid after the most recent forward_step(); every other column reads
    // -1e9 (see attention_scores_forward's comment for why).
    float* attentionMatrix() const { return attention_matrix_; }

    // Embedding lookup for a single (token_id, position) pair:
    // hidden_state[i] = wte[token_id, i] + wpe[position, i]. Then projects
    // that hidden state through layer 0's c_attn weights into qkvOutput(),
    // splits the result into qBuf()/kBuf()/vBuf(), and computes this
    // token's causally-masked attention scores into attentionMatrix().
    // Throws if token_id/position are out of range, or if a kernel
    // launch/execution fails.
    void forward_step(int token_id, int position);

private:
    void mapFile(const std::string& path);
    void uploadToDevice();

    // Host mmap state
    float* host_data_ = nullptr; // base of the mmap'd region
    size_t file_size_ = 0;       // bytes
    int fd_ = -1;
    Gpt2Params host_params_{};

    // Device state
    float* device_weights_ = nullptr; // VRAM, kExpectedParamCount floats
    float* hidden_state_ = nullptr;   // VRAM, kHiddenStateCount floats
    float* qkv_output_ = nullptr;     // VRAM, kQkvOutputCount floats
    float* q_buf_ = nullptr;          // VRAM, kHeadBufCount floats
    float* k_buf_ = nullptr;          // VRAM, kHeadBufCount floats
    float* v_buf_ = nullptr;          // VRAM, kHeadBufCount floats
    float* attention_matrix_ = nullptr; // VRAM, kAttnMatrixCount floats
    Gpt2Params device_params_{};
};

} // namespace vanguard
