# ================================
# FILE STATUS: FROZEN
# Phase2_representation : sentence_attribution.py
# Verified on: 2026-01-24 by dhanshri time: 05.57pm
# Do NOT modify this file
# ================================
"""
Sentence attribution via attention weights.

Provides a deterministic utility to rank sentences by their attention weight.
"""
from typing import Iterable, List, Tuple, Any


def get_sentence_attribution(attn_weights: Iterable[float], sentences: Iterable[str]) -> List[Tuple[str, float]]:
    """
    Pair sentences with attention weights and return them sorted by descending weight.

    Args:
        attn_weights: iterable of numeric attention weights (one per sentence)
        sentences: iterable of sentence strings (same length as attn_weights)

    Returns:
        List of (sentence, weight) tuples sorted by weight descending.
    """
    weights = list(attn_weights)
    sents = list(sentences)
    if len(weights) != len(sents):
        raise ValueError("attn_weights and sentences must have the same length")

    paired = list(zip(sents, [float(w) for w in weights]))
    # Sort by weight descending (deterministic)
    paired.sort(key=lambda x: x[1], reverse=True)
    return paired


# if __name__ == "__main__":
#     # Sanity test
#     sentences = [
#         "I prefer working alone.",
#         "I enjoy collaborating with others.",
#         "I plan my tasks carefully.",
#     ]
#     attn = [0.15, 0.6, 0.25]
#     ranked = get_sentence_attribution(attn, sentences)
#     print("Ranked sentences by attention weight:")
#     for i, (sent, w) in enumerate(ranked, 1):
#         print(f"{i}. ({w:.2f}) {sent}")

