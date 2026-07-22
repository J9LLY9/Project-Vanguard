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
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    return 0;
}
