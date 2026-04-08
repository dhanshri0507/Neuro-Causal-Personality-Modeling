# ================================
# FILE STATUS: FROZEN
# Phase2_representation : negation_count.py
# Verified on: 2026-01-23 by Dhanshri time: 06.16pm
# Do NOT modify this file
# ================================
"""
Sentence-level negation count feature.

negation_count(s_j) = Σ 1[token.dep_ == "neg"]

This module counts tokens whose dependency label equals the literal string
"neg". It uses dependency labels only (no text matching) and supports common
token-like shapes (spaCy Token, dicts, tuples/lists).
"""
from typing import Iterable, Any, Tuple


def _extract_dep(token: Any) -> Tuple[Any, Any]:
    """
    Safely extract (text, dep) from a token-like object without modifying it.

    Supported shapes:
    - spaCy Token (.text, .dep_)
    - dict with keys 'text' and 'dep_' or 'dep'
    - tuple/list where dep is commonly at index 2: (text, pos, dep)
    Returns (text, dep) where either may be None.
    """
    text = None
    dep = None

    if hasattr(token, "text"):
        try:
            text = token.text
        except Exception:
            text = None
    if hasattr(token, "dep_"):
        try:
            dep = token.dep_
        except Exception:
            dep = None

    if isinstance(token, dict):
        if text is None:
            text = token.get("text") or token.get("token")
        if dep is None:
            dep = token.get("dep_") or token.get("dep")

    if isinstance(token, (list, tuple)):
        if text is None and len(token) >= 1:
            text = token[0]
        # common tuple shape: (text, pos, dep)
        if dep is None and len(token) >= 3:
            dep = token[2]

    return text, dep


def negation_count(tokens: Iterable[Any]) -> int:
    """
    Count negation dependency labels in the provided token iterable.

    Args:
        tokens: iterable of token-like objects

    Returns:
        int: number of tokens with dependency label exactly equal to 'neg'
    """
    if tokens is None:
        return 0

    count = 0
    for tok in tokens:
        _, dep = _extract_dep(tok)
        if dep == "neg":
            count += 1

    return count


# if __name__ == "__main__":
#     # Sanity checks comparing negated vs affirmative sentences.
#     examples = [
#         # tuple form: (text, pos, dep)
#         ([( "I", "PRON", "nsubj"), ("do", "AUX", "aux"), ("not", "PART", "neg"), ("like", "VERB", "ROOT")], "I do not like"),
#         ([( "She", "PRON", "nsubj"), ("likes", "VERB", "ROOT"), ("it", "PRON", "dobj")], "She likes it"),
#         # dict-like tokens
#         ([{"text": "I", "dep_": "nsubj"}, {"text": "never", "dep_": "neg"}, {"text": "forgot", "dep_": "ROOT"}], "I never forgot"),
#     ]

#     for toks, text in examples:
#         cnt = negation_count(toks)
#         print(f"Sentence: {text!r}  ->  Negation count: {cnt}")