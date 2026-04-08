# ================================
# FILE STATUS: FROZEN
# Phase2_representation : readability_metrics.py
# Verified on: 2026-01-23 by Dhanshri time: 09.54pm
# Do NOT modify this file
# ================================
"""
Sentence-level readability using Flesch Reading Ease (FRE).

FRE(s_j) = 206.835
        - 1.015 × (words(s_j) / 1)
        - 84.6 × (syllables(s_j) / words(s_j))

This module exposes `readability_score(sentence_text: str) -> float` which
returns the sentence-level Flesch Reading Ease score computed by textstat.

Rules:
- Sentence-level only (one sentence in, one float out)
- Deterministic
- Do NOT normalize or clamp output
- Return 0.0 if input is empty or contains no alphabetic characters

Typical Range: 30 - 80 (higher = simpler)
"""
from typing import Any
from textstat import textstat


def readability_score(sentence_text: str) -> float:
    """
    Compute sentence-level Flesch Reading Ease using textstat.

    Args:
        sentence_text: single sentence as a string

    Returns:
        float: FRE score for the sentence, or 0.0 if sentence is empty or has no alphabetic tokens
    """
    if not isinstance(sentence_text, str):
        raise TypeError("sentence_text must be a string")

    if not sentence_text or not any(ch.isalpha() for ch in sentence_text):
        return 0.0

    # textstat.flesch_reading_ease expects a text string; for a single sentence
    # it returns the FRE score directly.
    return textstat.flesch_reading_ease(sentence_text)


# if __name__ == "__main__":
#     examples = [
#         ("The cat sat on the mat.", "very simple sentence"),
#         (
#             "Despite the ubiquity of computational methods, the theoretical underpinnings "
#             "of syntactic parsing remain a complex and nuanced field, requiring careful "
#             "consideration of linguistic variation and algorithmic trade-offs.",
#             "long complex sentence",
#         ),
#     ]

#     for sent, desc in examples:
#         score = readability_score(sent)
#         print(f"{desc}: {sent!r}\n  FRE score: {score}\n")

