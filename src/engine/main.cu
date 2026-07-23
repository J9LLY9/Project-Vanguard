#include <cstdio>
#include <exception>

#include <cuda_runtime.h>

#include "gpt2_engine.hpp"

// Entry point: mmap + upload the weights, then run a single-token test
// inference through the embedding layer and print the result.
int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <gpt2_124M.bin>\n", argv[0]);
        return 1;
    }

    try {
        vanguard::Gpt2Engine engine(argv[1]);
        std::printf("Engine ready: weights mmap'd and uploaded to VRAM.\n");

        // Test inference: GPT-2's <|endoftext|> token id, position 0.
        const int token_id = 50256;
        const int position = 0;
        engine.forward_step(token_id, position);

        float h_out[10] = {};
        cudaCheck(cudaMemcpy(h_out, engine.hiddenState(), sizeof(h_out),
                              cudaMemcpyDeviceToHost));

        std::printf("hidden_state[0..9] for token %d @ position %d:\n",
                    token_id, position);
        for (int i = 0; i < 10; ++i) {
            std::printf("  [%d] = %.6f\n", i, h_out[i]);
        }

        float qkv_out[10] = {};
        cudaCheck(cudaMemcpy(qkv_out, engine.qkvOutput(), sizeof(qkv_out),
                              cudaMemcpyDeviceToHost));

        std::printf("qkv_output[0..9] for token %d @ position %d:\n",
                    token_id, position);
        for (int i = 0; i < 10; ++i) {
            std::printf("  [%d] = %.6f\n", i, qkv_out[i]);
        }

        float q_head0[5] = {};
        cudaCheck(cudaMemcpy(q_head0, engine.qBuf(), sizeof(q_head0),
                              cudaMemcpyDeviceToHost));

        float q_head1[5] = {};
        cudaCheck(cudaMemcpy(q_head1, engine.qBuf() + 64, sizeof(q_head1),
                              cudaMemcpyDeviceToHost));

        std::printf("q_buf head 0 [0..4]:\n");
        for (int i = 0; i < 5; ++i) {
            std::printf("  [%d] = %.6f\n", i, q_head0[i]);
        }

        std::printf("q_buf head 1 [64..68]:\n");
        for (int i = 0; i < 5; ++i) {
            std::printf("  [%d] = %.6f\n", 64 + i, q_head1[i]);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    return 0;
}
