from transformers import GPT2Tokenizer
import os
try:
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.save_vocabulary(".")
    print("SUCCESS: vocab.json and merges.txt created in " + os.getcwd())
except Exception as e:
    print(f"ERROR: {e}")
