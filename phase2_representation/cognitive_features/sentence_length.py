# ================================
# FILE STATUS: FROZEN
# Phase2_representation : sentence_length.py
# Verified on: 2026-01-23 by Dhanshri time: 09.41pm
# Do NOT modify this file
# ================================
"""
Sentence length feature.

sentence_length(s_j) = |s_j|

This module provides a single function `sentence_length(doc) -> int` which
returns the number of spaCy tokens in the provided sentence (Doc or Span).
Rules:
- Count ALL tokens (including punctuation)
- Do not filter or remove stopwords
- Deterministic
"""
from typing import Iterable, Any


def sentence_length(doc: Iterable[Any]) -> int:
    """
    Return the number of tokens in `doc`.

    Args:
        doc: a spaCy Doc or Span (or any iterable of token-like objects)

    Returns:
        int: total number of tokens (0 if doc is None or empty)
    """
    if doc is None:
        return 0

    # For spaCy Doc/Span, len(doc) gives token count; for general iterables,
    # materialize into a list and count.
    try:
        return len(doc)
    except Exception:
        # Fallback: iterate and count
        count = 0
        for _ in doc:
            count += 1
        return count


# if __name__ == "__main__":
#     # Sanity checks using spaCy
#     import spacy

#     nlp = spacy.load("en_core_web_sm")
#     examples = [
#         "This is a short sentence.",
#         "Wait... what?!",
#         "",
#     ]

#     for sent in examples:
#         doc = nlp(sent)
#         length = sentence_length(doc)
#         print(f"Sentence: {sent!r} -> token count: {length}")

