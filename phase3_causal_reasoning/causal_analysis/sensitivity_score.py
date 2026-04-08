# ================================
# FILE STATUS: FROZEN
# Phase2_representation : sensitivity_score.py
# Verified on: 2026-01-24 by dhanshri time: 12.18pm
# Do NOT modify this file
# ================================
"""
Sensitivity score for a single MBTI trait w.r.t. a single cognitive feature.

sensitivity = |delta_p| / |delta_x|

Rules:
- Deterministic
- If delta_x == 0 -> return 0.0
- No smoothing, no epsilon, no clipping
"""

from typing import Union


def sensitivity_score(delta_p: Union[float, int], delta_x: Union[float, int]) -> float:
    """
    Compute sensitivity = |delta_p| / |delta_x|.

    Args:
        delta_p: change in probability (p_cf - p_factual)
        delta_x: change in feature value (x_cf - x)

    Returns:
        float: sensitivity score, or 0.0 if delta_x == 0
    """
    if not isinstance(delta_p, (int, float)) or not isinstance(delta_x, (int, float)):
        raise TypeError("delta_p and delta_x must be numeric scalars")

    if delta_x == 0:
        return 0.0

    return abs(float(delta_p)) / abs(float(delta_x))


# if __name__ == "__main__":
#     tests = [
#         (0.2, 0.1),   # non-zero delta_x (sensitivity = 2.0)
#         (0.05, 0.0),  # delta_x == 0 -> 0.0
#         (0.1, 1.0),   # small delta_p, large delta_x -> 0.1
#     ]
#     for dp, dx in tests:
#         s = sensitivity_score(dp, dx)
#         print(f"delta_p={dp}, delta_x={dx} -> sensitivity={s}")

