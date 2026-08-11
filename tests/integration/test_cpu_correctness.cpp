#include <cmath>
#include <cstdio>
#include <exception>
#include <iostream>
#include <vector>
#include "../../src/backends/cpu/cpu_engine.hpp"
#include "../../src/core/model/gpt2_tokenizer.hpp"

int main(int argc, char** argv) {
    std::string model_path = "gpt2_124M.bin";
    std::string vocab_path = "vocab.json";

    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) vocab_path = argv[2];

    std::printf("====================================================\n");
    std::printf(" Running Vanguard CPU Engine Correctness Test       \n");
    std::printf("====================================================\n");

    try {
        vanguard::CpuEngine engine(model_path);
        vanguard::Gpt2Tokenizer tokenizer(vocab_path);

        std::string prompt = "The square root of 4";
        std::vector<int> tokens = tokenizer.encode(prompt);

        std::printf("Prompt: '%s'\n", prompt.c_str());
        std::printf("Token IDs:");
        for (int t : tokens) std::printf(" %d", t);
        std::printf("\n");

        engine.resetCache();
        const float* logits = nullptr;

        for (size_t i = 0; i < tokens.size(); ++i) {
            logits = engine.forwardStep(tokens[i], static_cast<int>(i));
        }

        // Find argmax prediction
        int best_id = 0;
        float best_score = logits[0];
        for (int i = 1; i < vanguard::kGpt2VocabSize; ++i) {
            if (logits[i] > best_score) {
                best_score = logits[i];
                best_id = i;
            }
        }

        std::string pred_str = tokenizer.decode(best_id);
        std::printf("CPU Engine Prediction: Token ID %d ('%s'), Score: %.6f\n",
                    best_id, pred_str.c_str(), best_score);

        // Expected PyTorch output for "The square root of 4" is Token ID 13 ('.') with logit ~ -83.264160
        constexpr int kExpectedId = 13;
        constexpr float kExpectedScore = -83.264160f;
        constexpr float kTolerance = 1e-3f;

        if (best_id != kExpectedId) {
            std::printf("[FAIL] CPU Engine predicted ID %d, expected ID %d ('.')\n", best_id, kExpectedId);
            return 1;
        }

        float score_diff = std::fabs(best_score - kExpectedScore);
        if (score_diff > kTolerance) {
            std::printf("[FAIL] CPU score %.6f drifted from PyTorch expected %.6f (diff: %.6f)\n",
                        best_score, kExpectedScore, score_diff);
            return 1;
        }

        std::printf("[SUCCESS] CPU Engine prediction matches PyTorch golden baseline (ID %d '%.6f', diff: %.6f <= %.3f).\n",
                    best_id, best_score, score_diff, kTolerance);
        return 0;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "CPU Correctness Test Failed with exception: %s\n", e.what());
        return 1;
    }
}
