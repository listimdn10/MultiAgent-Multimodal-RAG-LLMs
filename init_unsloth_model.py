# init_unsloth_model.py
from unsloth import FastLanguageModel
import torch

def load_unsloth_model():
    max_seq_length = 1024
    dtype = None
    load_in_4bit = True
    NER_MODEL_ID = "Nhudang/LLama-3B-Solidity"
    print(f"[*] Tải model  bằng Unsloth: {NER_MODEL_ID}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Nhudang/LLama-3B-Solidity",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Loaded Unsloth model on {device}")

    return model, tokenizer, device
