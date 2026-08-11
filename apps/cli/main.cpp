#include <cstdio>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../../src/backends/cpu/cpu_engine.hpp"
#include "../../src/core/model/gpt2_tokenizer.hpp"

#ifdef VANGUARD_WITH_CUDA
#include <cuda_runtime.h>
#include "../../src/backends/cuda/cuda_engine.hpp"
#endif

int main(int argc, char** argv) {
    std::string model_path = "gpt2_124M.bin";
    std::string vocab_path = "vocab.json";
    std::string prompt = "The square root of 4";

    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) vocab_path = argv[2];
    if (argc >= 4) prompt = argv[3];

    std::printf("====================================================\n");
    std::printf(" Vanguard CLI Inference Engine                     \n");
    std::printf("====================================================\n");
    std::printf(" Model:  %s\n", model_path.c_str());
    std::printf(" Vocab:  %s\n", vocab_path.c_str());
    std::printf(" Prompt: '%s'\n", prompt.c_str());

    try {
        vanguard::Gpt2Tokenizer tokenizer(vocab_path);
        std::vector<int> tokens = tokenizer.encode(prompt);

        std::printf(" Tokens:");
        for (int t : tokens) std::printf(" %d", t);
        std::printf("\n");

#ifdef VANGUARD_WITH_CUDA
        std::printf(" Target: CUDA Backend\n");
        vanguard::CudaEngine engine(model_path);
        engine.prefill(tokens);

        std::vector<float> logits(vanguard::kGpt2VocabSize);
        cudaCheck(cudaMemcpy(logits.data(), engine.logits(),
                              logits.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
#else
        std::printf(" Target: CPU Backend\n");
        vanguard::CpuEngine engine(model_path);
        engine.resetCache();
        const float* logits = nullptr;
        for (size_t i = 0; i < tokens.size(); ++i) {
            logits = engine.forwardStep(tokens[i], static_cast<int>(i));
        }
#endif

        int best_id = 0;
        float best_score = logits[0];
        for (int i = 1; i < vanguard::kGpt2VocabSize; ++i) {
            if (logits[i] > best_score) {
                best_score = logits[i];
                best_id = i;
            }
        }

        std::string pred = tokenizer.decode(best_id);
        std::printf(" Prediction: Token ID %d ('%s'), Score: %.6f\n",
                    best_id, pred.c_str(), best_score);
        std::printf("====================================================\n");
        return 0;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "[Vanguard CLI Error] %s\n", e.what());
        return 1;
    }
}
