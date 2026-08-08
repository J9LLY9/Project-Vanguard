#!/usr/bin/env python3
"""
Generates golden reference tensors using PyTorch HuggingFace GPT-2 124M
for numerical cross-verification against CPU and CUDA Vanguard engines.

Usage:
    python3 scripts/verify_correctness.py [--prompt "The square root of 4"] [--output golden_tensors.json]
"""

import argparse
import json
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def main():
    parser = argparse.ArgumentParser(description="Export golden intermediate tensors from HuggingFace GPT-2")
    parser.add_argument("--prompt", type=str, default="The square root of 4", help="Input prompt string")
    parser.add_argument("--output", type=str, default="golden_tensors.json", help="Output JSON golden file")
    args = parser.parse_args()

    print(f"Loading HuggingFace GPT-2 ('gpt2')...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"][0].tolist()
    print(f"Prompt: '{args.prompt}'")
    print(f"Token IDs: {input_ids}")

    # Extract golden activations for each token position
    # PyTorch GPT-2 forward step
    with torch.no_grad():
        wte = model.transformer.wte(torch.tensor([input_ids[0]]))
        wpe = model.transformer.wpe(torch.tensor([0]))
        emb0 = (wte + wpe).squeeze(0).numpy()

        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits[0, -1].numpy()  # Last position logits
        pred_token_id = int(np.argmax(logits))
        pred_token_str = tokenizer.decode([pred_token_id])

    print(f"Top Logit Token: ID {pred_token_id} ('{pred_token_str}'), Score: {logits[pred_token_id]:.6f}")

    golden_data = {
        "prompt": args.prompt,
        "input_ids": input_ids,
        "first_token_emb": emb0.tolist()[:16],  # First 16 values for fast check
        "logits_top5_indices": np.argsort(logits)[-5:][::-1].tolist(),
        "logits_top5_values": [float(v) for v in np.sort(logits)[-5:][::-1]],
        "logits_sample_first10": [float(v) for v in logits[:10]],
    }

    with open(args.output, "w") as f:
        json.dump(golden_data, f, indent=2)

    print(f"Successfully saved golden tensors to '{args.output}'.")

if __name__ == "__main__":
    main()
