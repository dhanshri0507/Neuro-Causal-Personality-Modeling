# ================================
# FILE STATUS: FROZEN
# Phase2_representation : probability_bars.py
# Verified on: 2026-01-24 by dhanshri time: 05.30pm
# Do NOT modify this file
# ================================
"""
Probability bars visualization for MBTI dimensions.

Provides a simple bar chart for the four MBTI dimension probabilities:
IE, NS, TF, JP.
"""
from typing import Dict, Iterable, Tuple, Union
import matplotlib.pyplot as plt


DIM_ORDER = ["IE", "NS", "TF", "JP"]


def _parse_probabilities(probabilities: Union[Dict[str, float], Iterable[float]]) -> Tuple[list, list]:
    """
    Accept either a dict mapping dimension->probability or an iterable/list
    of 4 probabilities in DIM_ORDER. Returns (labels, values).
    """
    if isinstance(probabilities, dict):
        vals = []
        labels = DIM_ORDER
        for d in DIM_ORDER:
            if d not in probabilities:
                raise ValueError(f"Missing probability for dimension '{d}' in dict input")
            v = float(probabilities[d])
            vals.append(v)
        return labels, vals
    else:
        # iterable/list expected of length 4
        probs_list = list(probabilities)
        if len(probs_list) != 4:
            raise ValueError("Iterable input must have exactly 4 probability values in order [IE, NS, TF, JP]")
        labels = DIM_ORDER
        vals = [float(x) for x in probs_list]
        return labels, vals


def plot_probability_bars(probabilities: Union[Dict[str, float], Iterable[float]]):
    """
    Plot a bar chart of MBTI dimension probabilities.

    Args:
        probabilities: dict {dim: prob} with dims 'IE','NS','TF','JP' OR
                    iterable/list of 4 probabilities in order [IE,NS,TF,JP].

    Returns:
        matplotlib Figure and Axes objects (fig, ax)
    """
    labels, vals = _parse_probabilities(probabilities)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(labels))
    bars = ax.bar(x, vals, color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"])

    # Formatting
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Probability")
    ax.set_title("MBTI Dimension Probabilities")

    # Annotate bars with numeric values
    for bar, v in zip(bars, vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f"{v:.2f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.show()
    return fig, ax


# if __name__ == "__main__":
#     # Sanity test with dummy probabilities
#     dummy = {"IE": 0.23, "NS": 0.78, "TF": 0.41, "JP": 0.64}
#     plot_probability_bars(dummy)

