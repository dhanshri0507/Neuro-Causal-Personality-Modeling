# ================================
# FILE STATUS: FROZEN
# Phase2_representation : template_generator.py
# Verified on: 2026-01-24 by dhanshri time: 12.26pm
# Do NOT modify this file
# ================================
"""
Template-based explanation generator.

Converts numeric causal outputs into a simple, deterministic, human-readable
explanation using fixed templates. No learning or LLMs are used.
"""
from typing import Dict, Any, List


DIMENSIONS = ["IE", "NS", "TF", "JP"]


def _mean_confidence(probs: Dict[str, float]) -> float:
    if not probs:
        return 0.0
    vals = [float(v) for v in probs.values()]
    return sum(vals) / len(vals)


def generate_explanation(
    factual_mbti: str,
    factual_probs: Dict[str, float],
    counterfactual_mbti: str,
    counterfactual_probs: Dict[str, float],
    probability_shifts: Dict[str, float],
    sensitivity_scores: Dict[str, float],
    trait_flip_flags: Dict[str, bool],
) -> str:
    """
    Generate a multi-sentence explanation following the strict templates.
    """
    lines: List[str] = []

    # 1) Final factual MBTI prediction with confidence
    confidence = _mean_confidence(factual_probs)
    lines.append(f"Final MBTI type: {factual_mbti} (confidence {confidence:.2f}).")

    # 2) Causal impact summary: only dimensions where |Δ| > 0.1
    impacted = []
    for dim in DIMENSIONS:
        delta = float(probability_shifts.get(dim, 0.0))
        if abs(delta) > 0.1:
            direction = "increased" if delta > 0 else "decreased"
            impacted.append(f"{dim} probability {direction} by {delta:+.2f}.")

    if impacted:
        lines.append("Causal impact summary:")
        for s in impacted:
            lines.append(s)
    else:
        lines.append("Causal impact summary: No dimension had a probability shift greater than 0.10.")

    # 3) Counterfactual outcome: flips or no flip
    any_flip = any(bool(trait_flip_flags.get(dim, False)) for dim in DIMENSIONS)
    if any_flip:
        # State overall MBTI change
        if counterfactual_mbti != factual_mbti:
            lines.append(f"Under the intervention the model would predict {counterfactual_mbti} instead of {factual_mbti}.")
        # Per-dimension flip statements
        for i, dim in enumerate(DIMENSIONS):
            if trait_flip_flags.get(dim, False):
                before = factual_mbti[i] if i < len(factual_mbti) else "?"
                after = counterfactual_mbti[i] if i < len(counterfactual_mbti) else "?"
                lines.append(f"Trait {dim} flips from {before} to {after}.")
    else:
        lines.append("No trait flips occurred under the intervention; the predicted MBTI remains unchanged.")

    return "\n".join(lines)


# if __name__ == "__main__":
#     # Hardcoded dummy inputs for sanity demonstration
#     factual_mbti = "INFJ"
#     factual_probs = {"IE": 0.2, "NS": 0.8, "TF": 0.3, "JP": 0.7}
#     counterfactual_mbti = "INTP"
#     counterfactual_probs = {"IE": 0.8, "NS": 0.8, "TF": 0.2, "JP": 0.6}
#     probability_shifts = {"IE": 0.6, "NS": 0.0, "TF": -0.10, "JP": -0.10}
#     sensitivity_scores = {"IE": 2.0, "NS": 0.0, "TF": 0.5, "JP": 0.3}
#     trait_flip_flags = {"IE": True, "NS": False, "TF": False, "JP": False}

#     explanation = generate_explanation(
#         factual_mbti,
#         factual_probs,
#         counterfactual_mbti,
#         counterfactual_probs,
#         probability_shifts,
#         sensitivity_scores,
#         trait_flip_flags,
#     )
#     print(explanation)

