# ================================
# FILE STATUS: FROZEN
# Phase2_representation : do_intervention.py
# Verified on: 2026-01-24 by dhanshri time: 11.54pm
# Do NOT modify this file
# ================================
"""
Feature-level do() intervention (relative).

Computes a controlled counterfactual for a single scalar cognitive feature:

    x_cf = x + lam * (x_ref - x)

Rules:
- lam must be in [0.2, 0.4]
- Deterministic, scalar-only
- No clipping or normalization
"""
from typing import Union


def do_intervention(x: float, x_ref: float, lam: float = 0.3) -> float:
    """
    Apply relative do() intervention to a scalar cognitive feature.

    Args:
        x: original feature value (float)
        x_ref: reference feature value (float)
        lam: intervention strength in [0.2, 0.4] (default 0.3)

    Returns:
        x_cf: counterfactual feature value (float)
    """
    if not isinstance(x, (int, float)) or not isinstance(x_ref, (int, float)):
        raise TypeError("x and x_ref must be numeric scalars")
    if not isinstance(lam, (int, float)):
        raise TypeError("lam must be a numeric scalar")

    lam = float(lam)
    if lam < 0.2 or lam > 0.4:
        raise ValueError("lam (intervention strength) must be in [0.2, 0.4]")

    x_cf = x + lam * (x_ref - x)
    return float(x_cf)


# if __name__ == "__main__":
#     tests = [
#         (0.2, 0.5, 0.2),
#         (0.5, 0.0, 0.3),
#         (0.8, 0.2, 0.4),
#     ]
#     for x, x_ref, lam in tests:
#         cf = do_intervention(x, x_ref, lam)
#         print(f"x={x}, x_ref={x_ref}, lam={lam} -> x_cf={cf}")

