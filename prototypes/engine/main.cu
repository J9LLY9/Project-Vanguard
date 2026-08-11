#include <cstdio>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "gpt2_engine.hpp"
#include "gpt2_tokenizer.hpp"

namespace {

// Copies logits() back to the host and returns the argmax (index, score) --
// the greedy next-token prediction. logits() isn't a probability
// distribution (no softmax applied), but the argmax is identical either way.
struct Prediction {
    int token_id;
    float score;
};

Prediction ArgmaxLogits(const vanguard::Gpt2Engine& engine) {
    std::vector<float> logits(vanguard::kGpt2VocabSize);
    cudaCheck(cudaMemcpy(logits.data(), engine.logits(),
                          logits.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    Prediction best{0, logits[0]};
    for (int i = 1; i < vanguard::kGpt2VocabSize; ++i) {
        if (logits[i] > best.score) {
            best = {i, logits[i]};
        }
    }
    return best;
}

// Runs `token_ids` through the engine one position at a time (position ==
// index in the sequence) and reports the argmax prediction from the last
// position's logits.
//
// IMPORTANT: this engine has no KV-cache yet (see attention_scores_forward's
// comment in gpt2_engine.cu), so each forward_step() only ever attends to
// its own position -- key_pos < query_pos is masked out right alongside the
// real future-masking, because there's no persisted K/V history to attend
// to. Concretely: this means predicting what follows "The square root of 4"
// currently only ever sees token " 4" in isolation (plus its position
// index), not the four words before it. The loop below is what the task
// asked for -- feed a sequence through the engine one token at a time -- but
// it is not yet the same thing as full-sentence-context prediction; that
// needs a real KV-cache.
void RunSequence(vanguard::Gpt2Engine& engine,
                  const vanguard::Gpt2Tokenizer* tokenizer,
                  const std::vector<int>& token_ids, const std::string& label) {
    std::printf("\n=== %s (%zu token%s) ===\n", label.c_str(),
                token_ids.size(), token_ids.size() == 1 ? "" : "s");
    if (token_ids.size() > 1) {
        std::printf(
            "(no KV-cache yet: each step below only attends to itself, not "
            "the ones before it -- see the comment in RunSequence)\n");
    }

    for (size_t i = 0; i < token_ids.size(); ++i) {
        int token_id = token_ids[i];
        engine.forward_step(token_id, static_cast<int>(i));
        if (tokenizer != nullptr) {
            std::printf("  step %zu: token id %d (\"%s\")\n", i, token_id,
                        tokenizer->decode(token_id).c_str());
        } else {
            std::printf("  step %zu: token id %d\n", i, token_id);
        }
    }

    Prediction pred = ArgmaxLogits(engine);
    if (tokenizer != nullptr) {
        std::printf("-> predicted next token: id %d (\"%s\"), logit %.6f\n",
                    pred.token_id, tokenizer->decode(pred.token_id).c_str(),
                    pred.score);
    } else {
        std::printf("-> predicted next token: id %d, logit %.6f\n",
                    pred.token_id, pred.score);
    }
}

} // namespace

// Entry point: mmap + upload the weights, then run one or more token
// sequences through the full 12-layer forward pass.
//
// usage: gpt2_engine <gpt2_124M.bin> [<vocab.json> [<text>]]
//   - With a vocab.json, <text> (default "The square root of 4") is
//     tokenized via Gpt2Tokenizer and run as a sequence.
//   - Without one, a hardcoded id sequence for "The square root of 4" is
//     used instead (verified once against HuggingFace's GPT2Tokenizer), so
//     the sequence loop still works before a vocab.json is available.
int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                      "usage: %s <gpt2_124M.bin> [<vocab.json> [<text>]]\n",
                      argv[0]);
        return 1;
    }

    try {
        vanguard::Gpt2Engine engine(argv[1]);
        std::printf("Engine ready: weights mmap'd and uploaded to VRAM.\n");

        std::unique_ptr<vanguard::Gpt2Tokenizer> tokenizer;
        std::vector<int> sequence_ids;
        std::string sequence_label;

        if (argc >= 3) {
            tokenizer = std::make_unique<vanguard::Gpt2Tokenizer>(argv[2]);
            sequence_label = (argc >= 4) ? argv[3] : "The square root of 4";
            sequence_ids = tokenizer->encode(sequence_label);
        } else {
            std::printf(
                "(no vocab.json given -- using hardcoded token ids for "
                "\"The square root of 4\"; pass a vocab.json path to "
                "tokenize arbitrary text instead)\n");
            sequence_label = "The square root of 4";
            sequence_ids = {464, 6616, 6808, 286, 604};
        }

        RunSequence(engine, tokenizer.get(), sequence_ids, sequence_label);

        // Single-token sanity check from the previous verification pass,
        // still useful as a quick smoke test independent of the sequence
        // above.
        RunSequence(engine, tokenizer.get(), {15496}, "Hello");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    return 0;
}
