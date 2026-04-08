# ================================
# FILE STATUS: FROZEN
# Phase2_representation : probability_shift.py
# Verified on: 2026-01-24 by dhanshri time: 12.15pm
# Do NOT modify this file
# ================================
"""
Probability shift for a single MBTI dimension.

Δ = p_cf - p_factual

Deterministic, scalar-only.
"""

def probability_shift(p_factual: float, p_cf: float) -> float:
    """
    Compute probability shift for one MBTI dimension.

    Args:
        p_factual: factual probability (scalar)
        p_cf: counterfactual probability (scalar)

    Returns:
        float: p_cf - p_factual (in [-1, +1])
    """
    if not isinstance(p_factual, (int, float)) or not isinstance(p_cf, (int, float)):
        raise TypeError("p_factual and p_cf must be numeric scalars")
    return float(p_cf) - float(p_factual)


# if __name__ == "__main__":
#     tests = [
#         (0.3, 0.6),  # positive shift
#         (0.7, 0.4),  # negative shift
#         (0.5, 0.5),  # zero shift
#     ]
#     for p0, p1 in tests:
#         print(f"p_factual={p0}, p_cf={p1} -> shift={probability_shift(p0, p1)}")

