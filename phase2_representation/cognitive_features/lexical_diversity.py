# ================================
# FILE STATUS: FROZEN
# Phase2_representation : lexical_diversity.py
# Verified on: 2026-01-23 by Dhanshri time: 09.4pm
# Do NOT modify this file
# ================================
"""
Lexical diversity feature.

Definition:
    lexical_diversity(s_j) = (# unique alphabetic tokens) / (# alphabetic tokens)

Rules:
- Use spaCy tokens
- Consider ONLY alphabetic tokens (token.is_alpha == True)
- Lowercase token.text before counting uniqueness
- Do NOT include punctuation, numbers, or symbols
- Return 0.0 if there are no alphabetic tokens
"""
from typing import Iterable, Any


def lexical_diversity(doc: Iterable[Any]) -> float:
    """
    Compute lexical diversity for a spaCy Doc/Span (sentence-level).

    Args:
        doc: iterable of spaCy Token-like objects

    Returns:
        float: unique_alpha_count / total_alpha_count, or 0.0 if no alpha tokens
    """
    if doc is None:
        return 0.0

    total_alpha = 0
    uniques = set()

    for token in doc:
        # Prefer spaCy token attribute .is_alpha; fall back to text.isalpha()
        is_alpha = getattr(token, "is_alpha", None)
        text = getattr(token, "text", None)
        if is_alpha is None:
            # tolerant fallback: require a string and .isalpha()
            if not isinstance(text, str) or not text.isalpha():
                continue
            alpha = True
        else:
            if not is_alpha:
                continue
            alpha = True

        # At this point token is alphabetic; count it
        total_alpha += 1
        # Lowercase token text for uniqueness per spec
        if isinstance(text, str):
            uniques.add(text.lower())
        else:
            uniques.add(str(text).lower())

    if total_alpha == 0:
        return 0.0
    return len(uniques) / total_alpha


# if __name__ == "__main__":
#     # Sanity checks with spaCy
#     import spacy

#     nlp = spacy.load("en_core_web_sm")
#     examples = [
#         ("dog dog dog dog", "repeated words (low diversity)"),
#         ("The quick brown fox jumps over the lazy dog", "varied vocabulary (high diversity)"),
#         ("123 456 !!!", "no alphabetic tokens"),
#     ]

#     for sent, desc in examples:
#         doc = nlp(sent)
#         score = lexical_diversity(doc)
#         print(f"{desc}: {sent!r} -> lexical_diversity = {score:.3f}")

