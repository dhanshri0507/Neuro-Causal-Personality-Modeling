# ================================
# FILE STATUS: FROZEN
# Phase2_representation : decision_logic.py
# Verified on: 2026-01-24 by dhanshri time: 12.08pm
# Do NOT modify this file
# ================================
"""
Decision logic for trait flip under counterfactual intervention.

Rule (with buffer zone):
- A trait flip occurs if:
    (p_factual >= 0.6 and p_cf <= 0.4)
    OR
    (p_factual <= 0.4 and p_cf >= 0.6)

Values in (0.4, 0.6) are a buffer zone -> no flip.
"""

def trait_flip(p_factual: float, p_cf: float) -> bool:
    """
    Decide whether a single MBTI trait flips between factual and counterfactual.

    Args:
        p_factual: probability under factual scenario (scalar)
        p_cf:       probability under counterfactual scenario (scalar)

    Returns:
        True if trait flips according to the rule, False otherwise.
    """
    if not isinstance(p_factual, (int, float)) or not isinstance(p_cf, (int, float)):
        raise TypeError("p_factual and p_cf must be numeric scalars")

    # Inclusive bounds per specification
    if (p_factual >= 0.6 and p_cf <= 0.4) or (p_factual <= 0.4 and p_cf >= 0.6):
        return True
    return False


if __name__ == "__main__":
    tests = [
        (0.6, 0.4),   # flip
        (0.59, 0.41), # no flip (inside buffer)
        (0.4, 0.6),   # flip
        (0.5, 0.5),   # no flip
    ]
    for p0, p1 in tests:
        print(f"p_factual={p0}, p_cf={p1} -> flip={trait_flip(p0, p1)}")

