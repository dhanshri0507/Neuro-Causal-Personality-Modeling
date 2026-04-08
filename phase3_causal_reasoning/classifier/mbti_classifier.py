# ================================
# FILE STATUS: FROZEN
# Phase2_representation : mbti_classifier.py
# Verified on: 2026-01-23 by dhanshri time: 11.35pm
# Do NOT modify this file
# ================================
"""
MBTI classifier (multi-head logistic heads).

Predicts four MBTI dimension probabilities from joint vector J ∈ R^512.

Each head is an independent logistic (sigmoid) head:
    p_d = sigmoid(W_d · J + b_d)

Outputs:
- probabilities dict {'IE','NS','TF','JP'}
- predicted MBTI type string (e.g., 'INTJ')
- overall confidence = mean probability
"""
from typing import Dict, Any
import torch
import torch.nn as nn


class MBTIClassifier(nn.Module):
    def __init__(self, input_dim: int = 512):
        """
        Args:
            input_dim: dimensionality of joint vector J (default 512)
        """
        super().__init__()
        self.input_dim = input_dim
        # Four independent logistic heads (each maps input_dim -> 1)
        self.ie_head = nn.Linear(input_dim, 1)
        self.ns_head = nn.Linear(input_dim, 1)
        self.tf_head = nn.Linear(input_dim, 1)
        self.jp_head = nn.Linear(input_dim, 1)

    def forward(self, J: torch.Tensor) -> Dict[str, Any]:
        """
        Forward pass: compute four probabilities, MBTI type, and confidence.

        Args:
            J: torch.Tensor of shape (input_dim,) representing joint vector

        Returns:
            dict with keys:
                'probabilities': {'IE': p_ie, 'NS': p_ns, 'TF': p_tf, 'JP': p_jp}
                'type': 4-letter MBTI string
                'confidence': mean probability (float)
        """
        if not isinstance(J, torch.Tensor):
            raise TypeError("J must be a torch.Tensor")
        if J.dim() != 1 or J.shape[0] != self.input_dim:
            raise ValueError(f"J must be 1-D tensor with shape ({self.input_dim},)")

        device = next(self.parameters()).device
        x = J.to(device).float()

        # Compute logits and probabilities
        with torch.no_grad():
            p_ie = torch.sigmoid(self.ie_head(x)).item()
            p_ns = torch.sigmoid(self.ns_head(x)).item()
            p_tf = torch.sigmoid(self.tf_head(x)).item()
            p_jp = torch.sigmoid(self.jp_head(x)).item()

        probs = {"IE": p_ie, "NS": p_ns, "TF": p_tf, "JP": p_jp}

        # MBTI letter mapping per dimension:
        # If p >= 0.5 -> second letter (E, S, F, P)
        # If p <  0.5 -> first letter  (I, N, T, J)
        letters = []
        letters.append("E" if p_ie >= 0.5 else "I")
        letters.append("S" if p_ns >= 0.5 else "N")
        letters.append("F" if p_tf >= 0.5 else "T")
        letters.append("P" if p_jp >= 0.5 else "J")
        mbti_type = "".join(letters)

        confidence = (p_ie + p_ns + p_tf + p_jp) / 4.0

        return {"probabilities": probs, "type": mbti_type, "confidence": confidence}


# if __name__ == "__main__":
#     # Sanity test with random joint vector
#     torch.manual_seed(0)
#     J = torch.randn(512)
#     model = MBTIClassifier(input_dim=512)
#     out = model(J)
#     print("Probabilities:", out["probabilities"])
#     print("Predicted MBTI type:", out["type"])
#     print("Confidence:", out["confidence"])

