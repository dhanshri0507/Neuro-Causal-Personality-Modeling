# ================================
# FILE STATUS: FROZEN
# Phase2_representation : bert_encoder.py
# Verified on: 2026-01-23 by Dhanshri time: 11.01pm
# Do NOT modify this file
# ================================
"""
BERT sentence encoder wrapper.

Converts tokenized BERT inputs into a 768-d sentence embedding using mean
pooling over the last_hidden_state.

Functions:
- encode_sentence(tokenized_input) -> torch.Tensor of shape (768,)

Rules enforced:
- Use BertModel.from_pretrained("bert-base-cased")
- Freeze all BERT parameters (requires_grad = False)
- Use last_hidden_state and mean pooling over token dimension
- Do NOT use [CLS] pooling
- No training, no gradients
"""
from typing import Dict, Any
import torch
from transformers import BertModel


# Load model once and freeze parameters
_model = BertModel.from_pretrained("bert-base-cased")
_model.eval()
for p in _model.parameters():
    p.requires_grad = False


def encode_sentence(tokenized_input: Dict[str, Any]) -> torch.Tensor:
    """
    Encode a tokenized sentence into a 768-d embedding.

    Args:
        tokenized_input: dict returned by a BERT tokenizer with PyTorch tensors,
                        expected keys: 'input_ids', 'attention_mask' (both tensors)

    Returns:
        torch.Tensor of shape (768,) (detached, on CPU)
    """
    if (
    not isinstance(tokenized_input, dict)
    or "input_ids" not in tokenized_input
    or "attention_mask" not in tokenized_input
):
        raise TypeError(
        "tokenized_input must be a dict with keys {'input_ids', 'attention_mask'} "
        "returned by bert_tokenize(sentence)"
    )



    # Ensure tensors are on the same device as the model
    device = next(_model.parameters()).device
    inputs = {}
    for k, v in tokenized_input.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
        else:
            inputs[k] = torch.tensor(v).to(device)

    # No gradients / no training
    with torch.no_grad():
        outputs = _model(**inputs)
        # last_hidden_state: (batch_size, seq_len, hidden_size)
        last_hidden = outputs.last_hidden_state

        # Use attention_mask if provided to avoid pooling padded tokens
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            # attention_mask: (batch_size, seq_len) -> expand to (batch_size, seq_len, 1)
            mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
            summed = (last_hidden * mask).sum(dim=1)  # sum over seq_len
            denom = mask.sum(dim=1).clamp(min=1e-9)  # avoid division by zero
            mean_pooled = summed / denom
        else:
            # simple mean over tokens
            mean_pooled = last_hidden.mean(dim=1)

        # mean_pooled shape: (batch_size, hidden_size)
        # We expect single-sentence input (batch_size = 1). Return 1-D tensor.
        emb = mean_pooled[0].detach().cpu()
        return emb

# if __name__ == "__main__":
#     # Sanity test using our own tokenizer wrapper
#     from phase2_representation.semantic_encoder.bert_tokenizer import bert_tokenize

#     sentence = "This is a short test sentence for encoding."
#     tokenized = bert_tokenize(sentence)   # ✅ CORRECT SOURCE
#     emb = encode_sentence(tokenized)

#     print("Embedding shape:", tuple(emb.shape))
