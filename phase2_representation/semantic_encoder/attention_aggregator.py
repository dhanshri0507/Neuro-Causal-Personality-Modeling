# ================================
# FILE STATUS: FROZEN
# Phase2_representation : attention_aggregator.py
# Verified on: 2026-01-23 by Dhanshri time: 11.06pm
# Do NOT modify this file
# ================================
"""
Attention-based sentence embedding aggregator.

Computes document-level semantic vector S_doc from sentence embeddings H
using a single learnable attention vector w:

alpha_i = softmax(w^T tanh(h_i))
  S_doc = sum_i alpha_i * h_i

Rules:
- Single learnable weight vector w
- No multi-head attention
- Deterministic forward pass
"""
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionAggregator(nn.Module):
    """
    Attention aggregator that maps a set of sentence embeddings H (T, D)
    to a single document embedding S_doc (D,).
    """

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size
        # Single learnable weight vector w of shape (hidden_size,)
        self.w = nn.Parameter(torch.randn(hidden_size))

    def forward(self, H: torch.Tensor, return_weights: bool = False):

        """
        Forward pass.

        Args:
            H: Tensor of shape (T, D) where D == hidden_size

        Returns:
            S_doc: Tensor of shape (D,)
        """
        if H is None:
            raise ValueError("H must be a tensor of shape (T, D)")
        if H.dim() != 2:
            raise ValueError("H must be 2-D tensor with shape (T, D)")
        T, D = H.shape
        if D != self.hidden_size:
            raise ValueError(f"Expected hidden_size={self.hidden_size}, got D={D}")

        # Apply tanh nonlinearity
        H_tanh = torch.tanh(H)  # (T, D)
        # Compute scores: (T, D) dot (D,) -> (T,)
        scores = H_tanh.matmul(self.w)  # (T,)
        # Attention weights
        alphas = F.softmax(scores, dim=0)  # (T,)
        # Weighted sum
        S_doc = (alphas.unsqueeze(-1) * H).sum(dim=0)  # (D,)
        if return_weights:
            return S_doc, alphas
        return S_doc



# if __name__ == "__main__":
#     # Sanity test
#     torch.manual_seed(0)
#     T = 5
#     D = 768
#     H = torch.randn(T, D)
#     agg = AttentionAggregator(hidden_size=D)
#     S = agg(H)
#     print("Input shape:", H.shape)
#     print("Output shape:", tuple(S.shape))

