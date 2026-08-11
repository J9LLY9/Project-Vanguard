#pragma once

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

namespace vanguard {

// Loads GPT-2's vocab.json (the "encoder.json" from OpenAI's public
// release -- a flat {token_string: id} JSON object) and turns plain text
// into a sequence of token ids.
//
// This is NOT the full GPT-2 tokenizer. Real GPT-2 encoding is byte-level
// BPE: text is first mapped through the byte<->unicode table below (same
// as here), then merged greedily according to merge-rank rules from a
// second file, merges.txt/vocab.bpe, which this class doesn't load. Instead,
// this does a simpler greedy longest-match directly against the vocab
// table. For whole words that are already single vocab entries -- true for
// most common English words, digits, and punctuation -- the two approaches
// agree exactly (verified against HuggingFace's GPT2Tokenizer for plain
// sentences). They can disagree on rarer words that would need multiple.
// BPE merge steps to reach a token that isn't itself a whole vocab entry.
class Gpt2Tokenizer {
public:
    // Throws std::runtime_error if the file can't be opened or doesn't
    // parse as a flat {string: int} JSON object.
    explicit Gpt2Tokenizer(const std::string& vocab_json_path);

    // Encodes `text` into GPT-2 token ids. Throws std::runtime_error if a
    // byte-level unit somehow isn't in the vocab (shouldn't happen: the
    // vocab's base 256 single-byte entries make every byte representable).
    std::vector<int> encode(const std::string& text) const;

    // Decodes a single token id back to its raw text, e.g. for printing a
    // predicted next token as a word instead of a bare id. Returns
    // "<unknown>" for an id outside the loaded vocab's range.
    std::string decode(int id) const;

private:
    std::unordered_map<std::string, int> vocab_; // byte-level string -> id
    std::vector<std::string> id_to_token_;         // id -> byte-level string
    std::array<std::string, 256> byte_to_unicode_; // UTF-8 encoding of GPT-2's byte-level codepoint for each raw byte
    std::unordered_map<std::string, unsigned char> unicode_to_byte_; // inverse of byte_to_unicode_
};

} // namespace vanguard
