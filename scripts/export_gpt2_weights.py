#!/usr/bin/env python3
"""
Export HuggingFace GPT-2 (124M / "gpt2") weights to a single flat
Float32 binary file for consumption by a C++ inference engine.

Usage:
    pip install torch transformers numpy
    python3 export_gpt2_weights.py

Output:
    gpt2_124M.bin         - raw float32 weights, concatenated in the
                             order printed below (no header, no padding)
    gpt2_124M_shapes.txt  - "name: [dims...]" for every tensor, in the
                             same order they were written to the .bin

IMPORTANT for the C++ side:
    HuggingFace's GPT2 implementation uses `Conv1D` (not `nn.Linear`)
    for c_attn / c_proj / c_fc / mlp.c_proj. Conv1D stores its weight
    as [in_features, out_features] -- the TRANSPOSE of what nn.Linear
    (and most C++ matmul code written by hand) expects. If your C++
    engine does `y = x @ W + b` with W as [in, out], you're fine using
    these shapes as-is. If you wrote your kernels assuming a
    [out, in] Linear-style layout, transpose those four weights
    on load (or transpose in this script before writing).

    lm_head.weight is tied to transformer.wte.weight (same tensor).
    This script skips writing it a second time -- reuse
    transformer.wte.weight in your C++ engine for the output
    projection.
"""

import numpy as np
import torch
from transformers import GPT2LMHeadModel

MODEL_NAME = "gpt2"  # the 124M variant
OUTPUT_BIN = "gpt2_124M.bin"
OUTPUT_SHAPES = "gpt2_124M_shapes.txt"


def main():
    print(f"Loading '{MODEL_NAME}' from HuggingFace...")
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.eval()

    state_dict = model.state_dict()

    # lm_head.weight is weight-tied to transformer.wte.weight (identical
    # tensor) -- skip it so we don't write the 50257x768 embedding twice.
    keys = [k for k in state_dict.keys() if k != "lm_head.weight"]

    shapes = []
    total_params = 0

    with open(OUTPUT_BIN, "wb") as f:
        for key in keys:
            tensor = state_dict[key].detach().to(torch.float32).contiguous()
            arr = tensor.numpy().astype(np.float32)

            shape = list(arr.shape)
            shapes.append((key, shape))
            print(f"{key}: {shape}")

            f.write(arr.tobytes())
            total_params += arr.size

    with open(OUTPUT_SHAPES, "w") as f:
        for key, shape in shapes:
            f.write(f"{key}: {shape}\n")

    print(f"\nWrote {len(keys)} tensors, {total_params:,} float32 params")
    print(f"  -> {OUTPUT_BIN}  ({total_params * 4:,} bytes)")
    print(f"  -> {OUTPUT_SHAPES}")


if __name__ == "__main__":
    main()
