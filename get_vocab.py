#!/usr/bin/env python3
import json
from transformers import GPT2Tokenizer

print("Downloading/loading GPT2Tokenizer vocab...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
vocab = tokenizer.get_vocab()

with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

print(f"SUCCESS: Saved vocab.json with {len(vocab):,} entries.")
