"""
Vectorized gated fusion + classifier for end-to-end hybrid MBTI training.

Fusion matches the paper (project C/S, sigmoid gate, element-wise blend). A small
MLP on top of J improves capacity versus four independent linear heads on J alone.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BatchedGatedFusion(nn.Module):
    """C (B, d_c), S (B, d_s) -> J (B, d_p)."""

    def __init__(self, cognitive_dim: int, projection_dim: int = 512, semantic_dim: int = 768):
        super().__init__()
        self.projection_dim = projection_dim
        self.proj_c = nn.Linear(cognitive_dim, projection_dim)
        self.proj_s = nn.Linear(semantic_dim, projection_dim)
        self.gate = nn.Linear(2 * projection_dim, projection_dim)
        self.norm_c = nn.LayerNorm(projection_dim)
        self.norm_s = nn.LayerNorm(projection_dim)

    def forward(self, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        c_p = self.norm_c(self.proj_c(c))
        s_p = self.norm_s(self.proj_s(s))
        g = torch.sigmoid(self.gate(torch.cat([c_p, s_p], dim=-1)))
        return g * c_p + (1.0 - g) * s_p


class HybridMBTIModel(nn.Module):
    """
    Fusion -> optional MLP trunk -> four binary heads.
    Set use_mlp_trunk=False to match the original linear-head-only setup.
    """

    def __init__(
        self,
        cognitive_dim: int = 10,
        projection_dim: int = 512,
        semantic_dim: int = 768,
        use_mlp_trunk: bool = True,
        mlp_hidden: int = 384,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.fusion = BatchedGatedFusion(cognitive_dim, projection_dim, semantic_dim)
        self.use_mlp_trunk = use_mlp_trunk
        if use_mlp_trunk:
            self.trunk = nn.Sequential(
                nn.Linear(projection_dim, mlp_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, projection_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.out_norm = nn.LayerNorm(projection_dim)
        else:
            self.trunk = nn.Identity()
            self.out_norm = nn.LayerNorm(projection_dim)

        self.heads = nn.ModuleList([nn.Linear(projection_dim, 1) for _ in range(4)])

    def forward(self, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        j = self.fusion(c, s)
        h = self.out_norm(self.trunk(j))
        return torch.cat([head(h) for head in self.heads], dim=-1)

    def encode_joint(self, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Representation fed to heads (after fusion + trunk), shape (B, projection_dim)."""
        j = self.fusion(c, s)
        return self.out_norm(self.trunk(j))
