# ================================
# FILE STATUS: FROZEN
# Phase2_representation : bert_tokenizer.py
# Verified on: 2026-01-23 by Dhanshri time: 10.25pm
# Do NOT modify this file
# ================================
"""
Wrapper for HuggingFace BERT tokenizer (bert-base-cased).

Provides:
- bert_tokenize(sentence: str) -> dict

Rules:
- Use BertTokenizer.from_pretrained("bert-base-cased")
- return_tensors="pt", truncation=True, padding=False
- Do NOT perform pooling or modify text
- Return dictionary containing only 'input_ids' and 'attention_mask'
"""
from typing import Dict, Any
from transformers import BertTokenizer


# Load tokenizer once
_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


def bert_tokenize(sentence: str) -> Dict[str, Any]:
    """
    Tokenize a sentence using the BERT WordPiece tokenizer.

    Args:
        sentence: raw text (do not modify before calling)

    Returns:
        dict with keys 'input_ids' and 'attention_mask' (PyTorch tensors)
    """
    if not isinstance(sentence, str):
        raise TypeError("sentence must be a string")

    encoded = _tokenizer(sentence, return_tensors="pt", truncation=True, padding=False)
    # Return only the required fields
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}


# if __name__ == "__main__":
#     examples = [
#         "This is a short sentence.",
#         "The quick brown fox jumps over the lazy dog.",
#     ]

#     for sent in examples:
#         out = bert_tokenize(sent)
#         input_ids = out["input_ids"]
#         attention_mask = out["attention_mask"]
#         print(f"Sentence: {sent!r}")
#         print("input_ids shape:", tuple(input_ids.shape))
#         print("attention_mask shape:", tuple(attention_mask.shape))
#         print("---")

