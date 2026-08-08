#include <cmath>
#include <cstdio>
#include <exception>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#include "../src/cuda/cuda_engine.hpp"
#include "../src/cpu/cpu_engine.hpp"
#include "../src/tokenizer/gpt2_tokenizer.hpp"

int main(int argc, char** argv) {
    std::string model_path = "gpt2_124M.bin";
    std::string vocab_path = "vocab.json";

    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) vocab_path = argv[2];

    std::printf("====================================================\n");
    std::printf(" Running Vanguard CUDA Engine Correctness Test      \n");
    std::printf("====================================================\n");

    try {
        vanguard::CudaEngine cuda_engine(model_path);
        vanguard::CpuEngine cpu_engine(model_path);
        vanguard::Gpt2Tokenizer tokenizer(vocab_path);

        std::string prompt = "The square root of 4";
        std::vector<int> tokens = tokenizer.encode(prompt);

        std::printf("Prompt: '%s'\n", prompt.c_str());
        std::printf("Token IDs:");
        for (int t : tokens) std::printf(" %d", t);
        std::printf("\n");

        // Run CPU Engine
        cpu_engine.resetCache();
        const float* cpu_logits = nullptr;
        for (size_t i = 0; i < tokens.size(); ++i) {
            cpu_logits = cpu_engine.forwardStep(tokens[i], static_cast<int>(i));
        }

        // Run CUDA Engine (Prefill phase with persistent KV-Cache)
        cuda_engine.prefill(tokens);

        std::vector<float> cuda_logits(vanguard::kGpt2VocabSize);
        cudaCheck(cudaMemcpy(cuda_logits.data(), cuda_engine.logits(),
                              cuda_logits.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // Find argmax prediction for CUDA Engine
        int cuda_best_id = 0;
        float cuda_best_score = cuda_logits[0];
        for (int i = 1; i < vanguard::kGpt2VocabSize; ++i) {
            if (cuda_logits[i] > cuda_best_score) {
                cuda_best_score = cuda_logits[i];
                cuda_best_id = i;
            }
        }

        std::string pred_str = tokenizer.decode(cuda_best_id);
        std::printf("CUDA Engine Prediction: Token ID %d ('%s'), Score: %.6f\n",
                    cuda_best_id, pred_str.c_str(), cuda_best_score);

        constexpr int kExpectedId = 13;
        constexpr float kExpectedScore = -83.264160f;
        constexpr float kTolerance = 1e-3f;

        if (cuda_best_id != kExpectedId) {
            std::printf("[FAIL] CUDA Engine predicted ID %d, expected ID %d ('.')\n", cuda_best_id, kExpectedId);
            return 1;
        }

        float score_diff = std::fabs(cuda_best_score - kExpectedScore);
        if (score_diff > kTolerance) {
            std::printf("[FAIL] CUDA score %.6f drifted from PyTorch expected %.6f (diff: %.6f)\n",
                        cuda_best_score, kExpectedScore, score_diff);
            return 1;
        }

        // Compare CUDA vs CPU logits across all 50,257 vocabulary entries
        float max_diff = 0.0f;
        for (int i = 0; i < vanguard::kGpt2VocabSize; ++i) {
            float diff = std::fabs(cuda_logits[i] - cpu_logits[i]);
            if (diff > max_diff) {
                max_diff = diff;
            }
        }

        std::printf("Max logit difference between CUDA and CPU engines: %.6f\n", max_diff);

        if (max_diff > kTolerance) {
            std::printf("[FAIL] CUDA and CPU engines drifted by max difference %.6f > tolerance %.6f\n",
                        max_diff, kTolerance);
            return 1;
        }

        std::printf("[SUCCESS] CUDA Engine with KV-Cache matches CPU Reference and PyTorch golden baseline 100%%!\n");
        return 0;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "CUDA Correctness Test Failed with exception: %s\n", e.what());
        return 1;
    }
}
