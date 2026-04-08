# ================================
# FILE STATUS: FROZEN
# Phase2_representation : rule_based_aggregation.py
# Verified on: 2026-01-23 by Dhanshri time: 10.15pm
# Do NOT modify this file
# ================================
"""
Rule-based aggregation of sentence-level cognitive features into a document-level vector.

Aggregates sentence-level features according to the Phase-2 specification:

1. Ratio-based features (MEAN):
- pronoun_ratio
- modality_score
- emotion_intensity
- lexical_diversity
- readability_score

2. Count-based features (VARIANCE):
- negation_count
- sentence_length

3. Binary cognitive markers (PROPORTION):
- reasoning
- planning
- uncertainty

Aggregation formulas:
 - Mean: μ = (1 / T) * sum f_j
 - Variance: σ² = (1 / T) * sum (f_j - μ)²
 - Proportion: p = (1 / T) * sum 1[f_j == 1]

Rules:
- If sentence_features list is empty, return {}
- If a feature is missing for a sentence, skip that sentence for that feature
- Deterministic only
"""
from typing import List, Dict, Any


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


def aggregate_cognitive_features(sentence_features: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregate a list of sentence-level feature dictionaries into document-level features.

    Args:
        sentence_features: list of dicts, one per sentence. Each dict maps feature
                        names to numeric values (float or int). Features may be
                        missing in some sentence dicts; those sentences are skipped
                        for that particular feature.

    Returns:
        dict: aggregated features with explicit suffixes:
            - <feature>_mean for ratio features
            - <feature>_var for count features
            - <feature>_prop for binary markers
            If a feature has no contributing sentences (no values), it is omitted.
    """
    if not sentence_features:
        return {}

    aggregated: Dict[str, float] = {}

    # Helper to collect values for a given feature name
    def _collect(feature_name: str):
        vals = []
        for s in sentence_features:
            if feature_name in s and s[feature_name] is not None:
                try:
                    # ensure numeric-like
                    v = s[feature_name]
                    vals.append(v)
                except Exception:
                    continue
        return vals

    # Ratio-based: compute means
    for feat in RATIO_FEATURES:
        vals = _collect(feat)
        if vals:
            mean = sum(vals) / len(vals)
            aggregated[f"{feat}_mean"] = mean

    # Count-based: compute variances (population variance with denominator T)
    for feat in COUNT_FEATURES:
        vals = _collect(feat)
        if vals:
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            aggregated[f"{feat}_var"] = var

    # Binary markers: compute proportions (1 if value == 1)
    for feat in BINARY_FEATURES:
        vals = _collect(feat)
        if vals:
            count_ones = sum(1 for v in vals if v == 1)
            prop = count_ones / len(vals)
            aggregated[f"{feat}_prop"] = prop

    return aggregated


if __name__ == "__main__":
    # Small sanity test with mock sentence feature dictionaries
    sentence_features = [
        {
            "pronoun_ratio": 0.20,
            "modality_score": 0.10,
            "emotion_intensity": 0.0,
            "lexical_diversity": 0.50,
            "readability_score": 60.0,
            "negation_count": 0,
            "sentence_length": 10,
            "reasoning": 0,
            "planning": 0,
            "uncertainty": 0,
        },
        {
            "pronoun_ratio": 0.15,
            "modality_score": 0.00,
            "emotion_intensity": 0.10,
            "lexical_diversity": 0.60,
            "readability_score": 55.0,
            "negation_count": 1,
            "sentence_length": 12,
            "reasoning": 1,
            "planning": 0,
            "uncertainty": 1,
        },
        # sentence with some missing features
        {
            "pronoun_ratio": 0.22,
            "sentence_length": 8,
            "reasoning": 0,
        },
    ]

    print("Input sentence features:")
    for s in sentence_features:
        print(s)
    print("\nAggregated document features:")
    agg = aggregate_cognitive_features(sentence_features)
    for k, v in agg.items():
        print(f"{k}: {v}")

