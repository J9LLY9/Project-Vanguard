#include "gpt2_tokenizer.hpp"

#include <cctype>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace vanguard {

namespace {

// Appends codepoint `cp` to `out`, UTF-8 encoded. GPT-2's byte-level
// codepoints never exceed ~380 decimal, but this handles the full BMP
// range so it also works for any literal unicode text already present in
// vocab.json's token strings.
void AppendUtf8(std::string& out, uint32_t cp) {
    if (cp <= 0x7F) {
        out += static_cast<char>(cp);
    } else if (cp <= 0x7FF) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp <= 0xFFFF) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

// GPT-2's byte<->unicode mapping (bytes_to_unicode() in OpenAI's original
// gpt-2 repo): printable ASCII/Latin-1 bytes map to themselves; every other
// byte (control chars, space, DEL, etc.) maps to a codepoint starting at
// 256, in ascending byte order. This exists so byte-level BPE can treat
// arbitrary bytes (including whitespace and control bytes) as regular
// "printable" symbols a text-based vocab file can store as JSON keys.
std::array<std::string, 256> BuildByteToUnicode() {
    std::array<bool, 256> is_base{};
    for (int b = 33; b <= 126; ++b) is_base[b] = true;   // '!'..'~'
    for (int b = 161; b <= 172; ++b) is_base[b] = true;  // '¡'..'¬'
    for (int b = 174; b <= 255; ++b) is_base[b] = true;  // '®'..'ÿ'

    std::array<std::string, 256> table;
    int next_extra = 0;
    for (int b = 0; b < 256; ++b) {
        uint32_t cp = is_base[b] ? static_cast<uint32_t>(b)
                                  : static_cast<uint32_t>(256 + next_extra++);
        AppendUtf8(table[b], cp);
    }
    return table;
}

// Minimal parser for vocab.json's shape: a single flat JSON object mapping
// string keys to non-negative integer ids, e.g. {"!": 0, "\"": 1, ...}.
// Handles standard JSON string escapes plus \uXXXX (no surrogate pairs --
// GPT-2's byte-level vocab entries are always built from codepoints under
// 0x800, so none are needed).
class JsonVocabParser {
public:
    explicit JsonVocabParser(const std::string& data) : data_(data) {}

    void Parse(std::unordered_map<std::string, int>& out) {
        SkipWhitespace();
        Expect('{');
        SkipWhitespace();
        if (Peek() == '}') {
            ++pos_;
            return;
        }
        while (true) {
            SkipWhitespace();
            Expect('"');
            std::string key = ParseStringBody();
            SkipWhitespace();
            Expect(':');
            SkipWhitespace();
            int value = ParseInt();
            out.emplace(std::move(key), value);
            SkipWhitespace();
            char c = Peek();
            if (c == ',') {
                ++pos_;
                continue;
            }
            if (c == '}') {
                ++pos_;
                break;
            }
            throw std::runtime_error(
                "Gpt2Tokenizer: malformed vocab.json (expected ',' or '}' "
                "at byte " + std::to_string(pos_) + ")");
        }
    }

private:
    char Peek() const {
        if (pos_ >= data_.size()) {
            throw std::runtime_error(
                "Gpt2Tokenizer: unexpected end of vocab.json");
        }
        return data_[pos_];
    }

    void Expect(char c) {
        if (Peek() != c) {
            throw std::runtime_error(
                "Gpt2Tokenizer: malformed vocab.json (expected '" +
                std::string(1, c) + "' at byte " + std::to_string(pos_) + ")");
        }
        ++pos_;
    }

    void SkipWhitespace() {
        while (pos_ < data_.size() &&
               std::isspace(static_cast<unsigned char>(data_[pos_]))) {
            ++pos_;
        }
    }

    // Consumes the closing quote too; the opening quote must already have
    // been consumed by the caller.
    std::string ParseStringBody() {
        std::string out;
        while (true) {
            char c = Peek();
            ++pos_;
            if (c == '"') break;
            if (c != '\\') {
                out += c;
                continue;
            }
            char esc = Peek();
            ++pos_;
            switch (esc) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp = ParseHex4();
                    AppendUtf8(out, cp);
                    break;
                }
                default:
                    throw std::runtime_error(
                        "Gpt2Tokenizer: unknown escape '\\" +
                        std::string(1, esc) + "' in vocab.json");
            }
        }
        return out;
    }

    uint32_t ParseHex4() {
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            char c = Peek();
            ++pos_;
            v <<= 4;
            if (c >= '0' && c <= '9') v |= static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') v |= static_cast<uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= static_cast<uint32_t>(c - 'A' + 10);
            else throw std::runtime_error(
                "Gpt2Tokenizer: bad \\u escape in vocab.json");
        }
        return v;
    }

    int ParseInt() {
        size_t start = pos_;
        if (Peek() == '-') ++pos_;
        while (pos_ < data_.size() &&
               std::isdigit(static_cast<unsigned char>(data_[pos_]))) {
            ++pos_;
        }
        if (pos_ == start) {
            throw std::runtime_error(
                "Gpt2Tokenizer: expected integer in vocab.json at byte " +
                std::to_string(start));
        }
        return std::stoi(data_.substr(start, pos_ - start));
    }

    const std::string& data_;
    size_t pos_ = 0;
};

// Approximates GPT-2's regex pre-tokenizer for plain ASCII text: splits
// into chunks of "optional leading space + run of letters", "... + run of
// digits", "... + one other symbol", or a trailing whitespace run. Real
// GPT-2 uses a unicode-aware regex (`\p{L}+`, `\p{N}+`, etc.); this covers
// the common case (English words, numbers, punctuation) but won't handle
// multi-byte unicode letters/digits as a single run.
std::vector<std::string> PretokenizeWords(const std::string& text) {
    std::vector<std::string> chunks;
    size_t i = 0;
    const size_t n = text.size();
    while (i < n) {
        size_t start = i;
        if (text[i] == ' ' && i + 1 < n) {
            ++i;
        }
        auto is_alpha = [](char c) {
            return std::isalpha(static_cast<unsigned char>(c)) != 0;
        };
        auto is_digit = [](char c) {
            return std::isdigit(static_cast<unsigned char>(c)) != 0;
        };
        if (i < n && is_alpha(text[i])) {
            while (i < n && is_alpha(text[i])) ++i;
        } else if (i < n && is_digit(text[i])) {
            while (i < n && is_digit(text[i])) ++i;
        } else if (i < n && text[i] == ' ') {
            while (i < n && text[i] == ' ') ++i;
        } else if (i < n) {
            ++i; // single punctuation/symbol byte
        }
        chunks.push_back(text.substr(start, i - start));
    }
    return chunks;
}

} // namespace

Gpt2Tokenizer::Gpt2Tokenizer(const std::string& vocab_json_path)
    : byte_to_unicode_(BuildByteToUnicode()) {
    std::ifstream file(vocab_json_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Gpt2Tokenizer: failed to open '" +
                                  vocab_json_path + "'");
    }
    std::ostringstream buf;
    buf << file.rdbuf();
    std::string data = buf.str();

    JsonVocabParser(data).Parse(vocab_);
    if (vocab_.empty()) {
        throw std::runtime_error("Gpt2Tokenizer: '" + vocab_json_path +
                                  "' parsed to an empty vocabulary");
    }

    int max_id = -1;
    for (const auto& [token, id] : vocab_) {
        if (id > max_id) max_id = id;
    }
    id_to_token_.assign(static_cast<size_t>(max_id) + 1, std::string());
    for (const auto& [token, id] : vocab_) {
        id_to_token_[static_cast<size_t>(id)] = token;
    }

    for (int b = 0; b < 256; ++b) {
        unicode_to_byte_.emplace(byte_to_unicode_[b],
                                  static_cast<unsigned char>(b));
    }
}

std::vector<int> Gpt2Tokenizer::encode(const std::string& text) const {
    std::vector<int> ids;

    for (const std::string& chunk : PretokenizeWords(text)) {
        std::vector<std::string> units;
        units.reserve(chunk.size());
        for (unsigned char b : chunk) {
            units.push_back(byte_to_unicode_[b]);
        }

        size_t start = 0;
        while (start < units.size()) {
            bool matched = false;
            for (size_t end = units.size(); end > start; --end) {
                std::string candidate;
                for (size_t k = start; k < end; ++k) candidate += units[k];

                auto it = vocab_.find(candidate);
                if (it != vocab_.end()) {
                    ids.push_back(it->second);
                    start = end;
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                throw std::runtime_error(
                    "Gpt2Tokenizer::encode: no vocab entry covers byte " +
                    std::to_string(static_cast<unsigned char>(chunk[start])) +
                    " in chunk \"" + chunk + "\"");
            }
        }
    }

    return ids;
}

std::string Gpt2Tokenizer::decode(int id) const {
    if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) {
        return "<unknown>";
    }
    const std::string& encoded = id_to_token_[static_cast<size_t>(id)];

    std::string out;
    size_t i = 0;
    while (i < encoded.size()) {
        bool matched = false;
        for (size_t len = 3; len >= 1; --len) {
            if (i + len > encoded.size()) continue;
            auto it = unicode_to_byte_.find(encoded.substr(i, len));
            if (it != unicode_to_byte_.end()) {
                out += static_cast<char>(it->second);
                i += len;
                matched = true;
                break;
            }
        }
        if (!matched) {
            ++i; // shouldn't happen: every byte-level unit is in the map
        }
    }
    return out;
}

} // namespace vanguard
