# ================================
# FILE STATUS: FROZEN
# Phase2_representation : counterfactual_predictor.py
# Verified on: 2026-01-24 by dhanshri time: 12.05pm
# Do NOT modify this file
# ================================
"""
Counterfactual predictor.

Generates counterfactual MBTI predictions by applying do()-style interventions
to selected cognitive features while keeping the semantic vector fixed.
"""
from typing import Dict, Any, List, Tuple
import torch

from .do_intervention import do_intervention

# We mirror the aggregation ordering used in Phase-2 representation:
RATIO_FEATURES = [
    "pronoun_ratio",
    "modality_score",
    "emotion_intensity",
    "lexical_diversity",
    "readability_score",
]

COUNT_FEATURES = [
    "negation_count",
    "sentence_length",
]

BINARY_FEATURES = [
    "reasoning",
    "planning",
    "uncertainty",
]

ALL_FEATURES_ORDER = RATIO_FEATURES + [f + "_var" for f in COUNT_FEATURES] + [f + "_prop" for f in BINARY_FEATURES]


def _build_tensor_from_C_doc(C_doc: Any) -> torch.Tensor:
    """
    Convert C_doc (tensor or dict) into a 1-D torch.Tensor following the
    deterministic ordering used across the pipeline:
    [<ratio>_mean..., <count>_var..., <binary>_prop...]

    If a feature is missing, default to 0.0.
    """
    if isinstance(C_doc, torch.Tensor):
        if C_doc.dim() != 1:
            raise ValueError("C_doc tensor must be 1-D")
        return C_doc.clone().detach().float()

    if isinstance(C_doc, dict):
        vals = []
        # ratio features expect '_mean' suffix in aggregated dicts
        for feat in RATIO_FEATURES:
            v = None
            if f"{feat}_mean" in C_doc:
                v = C_doc[f"{feat}_mean"]
            elif feat in C_doc:
                v = C_doc[feat]
            vals.append(float(v) if v is not None else 0.0)
        # count features expect '_var'
        for feat in COUNT_FEATURES:
            v = None
            if f"{feat}_var" in C_doc:
                v = C_doc[f"{feat}_var"]
            elif feat in C_doc:
                v = C_doc[feat]
            vals.append(float(v) if v is not None else 0.0)
        # binary features expect '_prop'
        for feat in BINARY_FEATURES:
            v = None
            if f"{feat}_prop" in C_doc:
                v = C_doc[f"{feat}_prop"]
            elif feat in C_doc:
                v = C_doc[feat]
            vals.append(float(v) if v is not None else 0.0)

        return torch.tensor(vals, dtype=torch.float32)

    raise TypeError("C_doc must be a 1-D torch.Tensor or a dict of feature values")


def counterfactual_predictor(
    C_doc: Any,
    S_doc: torch.Tensor,
    classifier: Any,
    fusion_model: Any,
    interventions: Dict[str, Tuple[float, float]],
) -> Dict[str, Any]:
    """
    Apply feature-wise do-interventions and produce factual and counterfactual MBTI predictions.

    Args:
        C_doc: original cognitive vector (tensor or dict)
        S_doc: original semantic tensor (768,)
        classifier: pretrained MBTIClassifier instance
        fusion_model: pretrained GatedFusion instance
        interventions: mapping feature_name -> (x_ref, lam)
            where feature_name is one of the base feature names (e.g. 'pronoun_ratio')
            and lam in [0.2, 0.4]

    Returns:
        dict with keys:
            'factual_probabilities', 'counterfactual_probabilities',
            'factual_type', 'counterfactual_type'
    """
    # Build C_doc tensor (do not modify original)
    C_tensor = _build_tensor_from_C_doc(C_doc)

    # Ensure S_doc is tensor and shape matches
    if not isinstance(S_doc, torch.Tensor):
        raise TypeError("S_doc must be a torch.Tensor")
    if S_doc.dim() != 1:
        raise ValueError("S_doc must be 1-D tensor")

    # 2) Counterfactual: apply interventions feature-wise
    C_cf = C_tensor.clone().detach()

    # Build mapping from base feature name to index in tensor
    index_map: Dict[str, int] = {}
    idx = 0
    for feat in RATIO_FEATURES:
        index_map[feat] = idx
        idx += 1
    for feat in COUNT_FEATURES:
        index_map[feat] = idx
        idx += 1
    for feat in BINARY_FEATURES:
        index_map[feat] = idx
        idx += 1

    # Apply interventions
    for feat_name, params in interventions.items():
        if feat_name not in index_map:
            # skip unknown features silently per deterministic skip rule
            continue
        x_ref, lam = params
        i = index_map[feat_name]
        x = float(C_cf[i].item())
        x_cf = do_intervention(x, float(x_ref), float(lam))
        C_cf[i] = torch.tensor(x_cf, dtype=C_cf.dtype)

    # 3) Fuse C_cf with same S_doc and classify
    J_cf = fusion_model(C_cf, S_doc)
    counterfactual_out = classifier(J_cf)

    if not counterfactual_out or "probabilities" not in counterfactual_out:
        return None

    counterfactual_probs = counterfactual_out["probabilities"]
    counterfactual_type = counterfactual_out["type"]


    return {
        "counterfactual_probabilities": counterfactual_probs,
        "counterfactual_type": counterfactual_type,
    }
    