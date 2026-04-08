# ================================
# FILE STATUS: FROZEN
# Phase2_representation : gated_fusion.py
# Verified on: 2026-01-23 by dhanshri time: 11.09pm
# Do NOT modify this file
# ================================
"""
Gated fusion of cognitive and semantic document vectors.

Class: GatedFusion(nn.Module)

Inputs:
- C_doc: Tensor shape (N,) -- cognitive vector
- S_doc: Tensor shape (768,) -- semantic vector

Process:
1. Project C_doc -> C' in R^512
2. Project S_doc -> S' in R^512
3. Concatenate [C'; S'] -> (1024,)
4. Compute gate g = sigmoid(W_g [C'; S'] + b_g)  -> (512,)
5. Fuse: J = g ⊙ C' + (1 - g) ⊙ S'

Output:
- J: Tensor shape (512,)

Rules:
- Gate is element-wise.
- No normalization or residuals.
"""
from typing import Any
import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    def __init__(self, cognitive_dim: int, projection_dim: int = 512, semantic_dim: int = 768):
        """
        Args:
            cognitive_dim: dimensionality of C_doc (N)
            projection_dim: target projection dimension (default 512)
            semantic_dim: dimensionality of S_doc (default 768)
        """
        super().__init__()
        self.cognitive_dim = cognitive_dim
        self.projection_dim = projection_dim
        self.semantic_dim = semantic_dim

        # Projections
        self.proj_c = nn.Linear(cognitive_dim, projection_dim)
        self.proj_s = nn.Linear(semantic_dim, projection_dim)

        # Gate: input is concatenated [C'; S'] of size 2 * projection_dim
        self.gate = nn.Linear(2 * projection_dim, projection_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, C_doc: torch.Tensor, S_doc: torch.Tensor) -> torch.Tensor:
        """
        Forward fusion pass.

        Args:
            C_doc: Tensor (N,)
            S_doc: Tensor (768,)

        Returns:
            J: Tensor (projection_dim,)
        """
        if C_doc is None or S_doc is None:
            raise ValueError("C_doc and S_doc must be provided")
        if C_doc.dim() != 1:
            raise ValueError("C_doc must be a 1-D tensor of shape (N,)")
        if S_doc.dim() != 1:
            raise ValueError("S_doc must be a 1-D tensor of shape (768,)")

        # ensure device alignment
        device = next(self.parameters()).device
        C = C_doc.to(device).float()
        S = S_doc.to(device).float()

        # 1 & 2: projections
        C_prime = self.proj_c(C)  # (projection_dim,)
        S_prime = self.proj_s(S)  # (projection_dim,)

        # 3: concatenate
        concat = torch.cat([C_prime, S_prime], dim=0)  # (2*projection_dim,)

        # 4: gate
        g = self.sigmoid(self.gate(concat))  # (projection_dim,)

        # 5: fused output (element-wise gate)
        J = g * C_prime + (1.0 - g) * S_prime  # (projection_dim,)

        return J


# if __name__ == "__main__":
#     # Sanity test
#     torch.manual_seed(0)
#     cognitive_dim = 10
#     C_doc = torch.randn(cognitive_dim)
#     S_doc = torch.randn(768)

#     fusion = GatedFusion(cognitive_dim=cognitive_dim, projection_dim=512, semantic_dim=768)
#     J = fusion(C_doc, S_doc)

#     print("C' shape:", tuple(fusion.proj_c(C_doc).shape))
#     print("S' shape:", tuple(fusion.proj_s(S_doc).shape))
#     # compute gate for printing
#     concat = torch.cat([fusion.proj_c(C_doc), fusion.proj_s(S_doc)], dim=0)
#     g = fusion.sigmoid(fusion.gate(concat))
#     print("g shape:", tuple(g.shape))
#     print("J shape:", tuple(J.shape))

