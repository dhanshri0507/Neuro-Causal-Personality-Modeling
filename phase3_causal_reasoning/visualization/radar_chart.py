# ================================
# FILE STATUS: FROZEN
# Phase2_representation : radar_chart.py
# Verified on: 2026-01-24 by dhanshri time: 05.45pm
# Do NOT modify this file
# ================================
"""
Radar chart for normalized cognitive features.

Input:
- features_dict: mapping feature_name -> normalized_value (assumed normalized)

This module only visualizes the provided values; it does not compute or
normalize features.
"""
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt
import numpy as np


def plot_cognitive_radar(features_dict: Dict[str, float]) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a radar chart of cognitive features.

    Args:
        features_dict: dict mapping feature_name -> numeric value (normalized)

    Returns:
        (fig, ax): matplotlib Figure and Axes
    """
    if not isinstance(features_dict, dict) or not features_dict:
        raise ValueError("features_dict must be a non-empty dict")

    labels = list(features_dict.keys())
    values = [float(features_dict[k]) for k in labels]
    N = len(labels)

    # Compute angles for radar chart
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    # close the loop
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color="#1f77b4", linewidth=2)
    ax.fill(angles, values, color="#1f77b4", alpha=0.25)

    # Set the feature labels at the correct angles
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # Set radial limits assuming normalized inputs in [0,1]
    ax.set_ylim(0, 1)
    ax.set_rlabel_position(30)
    ax.grid(True)
    ax.set_title("Cognitive Trait Radar (normalized)", y=1.08)

    plt.tight_layout()
    plt.show()
    return fig, ax


# if __name__ == "__main__":
#     # Sanity test with mock normalized features
#     mock = {
#         "pronoun_ratio": 0.25,
#         "modality_score": 0.10,
#         "negation_count": 0.05,
#         "emotion_intensity": 0.12,
#         "reasoning_prop": 0.4,
#         "planning_prop": 0.2,
#         "uncertainty_prop": 0.15,
#         "lexical_diversity": 0.6,
#     }
#     plot_cognitive_radar(mock)

