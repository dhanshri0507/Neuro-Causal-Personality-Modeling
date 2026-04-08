# ================================
# FILE STATUS: FROZEN
# Phase2_representation : sentiment_vader.py
# Verified on: 2026-01-23 by Dhanshri time: 06.36pm
# Do NOT modify this file
# ================================
"""
Sentence-level sentiment polarity using VADER.

sentiment_polarity(s_j) ∈ [-1, +1] taken from the VADER compound score.

This module exposes a single function `sentiment_score(text: str) -> float`
which returns the VADER compound score for the provided sentence. The
implementation follows the requirements strictly:
- uses vaderSentiment.vaderSentiment.SentimentIntensityAnalyzer
- returns only the 'compound' score
- performs no custom normalization, clamping, tokenization, or extra rules
"""
from typing import Any
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Instantiate analyzer once.
_analyzer = SentimentIntensityAnalyzer()


def sentiment_score(text: str) -> float:
    """
    Return the VADER compound sentiment score for `text`.

    Args:
        text: a sentence or short text string

    Returns:
        float: the 'compound' score from VADER (range [-1.0, 1.0])
    """
    if not isinstance(text, str):
        # Keep deterministic behavior; raise to indicate incorrect usage.
        raise TypeError("text must be a string")

    scores: Any = _analyzer.polarity_scores(text)
    # Use only the compound score per specification.
    return scores["compound"]


# if __name__ == "__main__":
#     examples = [
#         ("I love this, it's absolutely wonderful!", "positive"),
#         ("This is an average day.", "neutral"),
#         ("I hate this. It was the worst.", "negative"),
#     ]

#     for sent, label in examples:
#         score = sentiment_score(sent)
#         print(f"Sentence ({label}): {sent!r} -> compound: {score}")

