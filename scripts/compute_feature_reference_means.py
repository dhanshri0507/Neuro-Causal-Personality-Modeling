#!/usr/bin/env python3
# FILE STATUS: FROZEN
# Do not modify unless Phase-2 logic changes

"""
Compute dataset-level reference means for cognitive features (C).

Loads data/processed/phase2_features.npz, computes numpy.nanmean per cognitive
feature (column-wise), and saves the results to data/stats/feature_reference_means.json.
"""
import os
import sys
import json
from typing import Dict, Any

import numpy as np


# Feature ordering must match Phase-2 aggregation ordering used elsewhere.
FEATURE_NAMES = [
    "pronoun_ratio",
    "modality_score",
    "emotion_intensity",
    "lexical_diversity",
    "readability_score",
    "negation_count",
    "sentence_length",
    "reasoning",
    "planning",
    "uncertainty",
]


def compute_reference_means(input_npz: str) -> Dict[str, float]:
    if not os.path.exists(input_npz):
        raise FileNotFoundError(f"Input file not found: {input_npz}")

    data = np.load(input_npz, allow_pickle=True)
    if "C" not in data:
        raise KeyError("Input NPZ does not contain 'C' array")

    C = data["C"]  # shape (N, num_features)
    if C.ndim != 2:
        raise ValueError("C array must be 2-dimensional (N, num_features)")

    num_features = C.shape[1]
    means = np.nanmean(C, axis=0)

    # Map means to feature names when possible, otherwise use index keys
    out: Dict[str, float] = {}
    for i in range(num_features):
        name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feature_{i}"
        out[name] = float(means[i])

    return out


if __name__ == "__main__":
    input_npz = os.path.join("data", "processed", "phase2_features.npz")
    output_json = os.path.join("data", "stats", "feature_reference_means.json")

    if not os.path.exists(input_npz):
        print(f"Input NPZ not found: {input_npz}. Skipping computation.")
        sys.exit(0)

    refs = compute_reference_means(input_npz)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as fh:
        json.dump(refs, fh, ensure_ascii=False, indent=2)

    print(f"Computed {len(refs)} feature reference means and saved to {output_json}")
    for k, v in refs.items():
        print(f"{k}: {v:.6f}")

