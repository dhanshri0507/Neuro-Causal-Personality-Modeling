# ================================
# FILE STATUS: FROZEN
# Phase2_representation : modality_score.py
# Verified on: 2026-01-23 by Dhanshri time: 02.30pm
# Do NOT modify this file
# ================================
"""
Sentence-level modality score.

Computes modality_score(s_j) = (# tokens whose lemma ∈ M) / (total valid tokens)
where M = {must, should, might, could, would, may}.

The function is lemma-based, case-insensitive, and excludes empty/whitespace
tokens. It is deterministic and does not modify tokens or text.
"""
from typing import Iterable, Any, Tuple


MODAL_LEMMAS = {"must", "should", "might", "could", "would", "may"}


def _extract_text_lemma(token: Any) -> Tuple[Any, Any]:
    """
    Attempt to extract (text, lemma) from a token-like object without
    modifying it. Returns (text, lemma) where either may be None.
    """
    text = None
    lemma = None

    # spaCy Token-like: .text and .lemma_
    if hasattr(token, "text"):
        try:
            text = token.text
        except Exception:
            text = None
    if hasattr(token, "lemma_"):
        try:
            lemma = token.lemma_
        except Exception:
            lemma = None

    # dict-like token
    if text is None and isinstance(token, dict):
        text = token.get("text") or token.get("token")
    if lemma is None and isinstance(token, dict):
        lemma = token.get("lemma_") or token.get("lemma")

    # tuple/list-like token: try common placements
    if text is None and isinstance(token, (list, tuple)) and len(token) >= 1:
        text = token[0]
    if lemma is None and isinstance(token, (list, tuple)):
        # heuristics: prefer element that looks like a lemma (str)
        for candidate in token[1:4]:
            if isinstance(candidate, str):
                lemma = candidate
                break

    # Fallback: if lemma missing but text itself matches a modal lemma,
    # use text as lemma (case-insensitive)
    if lemma is None and isinstance(text, str) and text.strip().lower() in MODAL_LEMMAS:
        lemma = text

    return text, lemma


def modality_score(tokens: Iterable[Any]) -> float:
    """
    Compute the modality score for a sentence represented as an iterable of tokens.

    Args:
        tokens: iterable of token-like objects (spaCy Token, tuple, dict, etc.)

    Returns:
        float in [0.0, 1.0] representing proportion of tokens whose lemma is in M.
        Returns 0.0 if input is empty or there are no valid linguistic tokens.
    """
    if tokens is None:
        return 0.0

    token_list = list(tokens)
    if not token_list:
        return 0.0

    valid_count = 0
    modal_count = 0

    for tok in token_list:
        text, lemma = _extract_text_lemma(tok)
        # Exclude non-linguistic tokens: require non-empty, non-whitespace text
        if not isinstance(text, str) or not text.strip():
            continue
        valid_count += 1
        if isinstance(lemma, str) and lemma.strip().lower() in MODAL_LEMMAS:
            modal_count += 1

    if valid_count == 0:
        return 0.0

    return modal_count / valid_count


# if __name__ == "__main__":
#     # Sanity checks with token-like inputs.
#     examples = [
#         # spaCy-like dict tokens
#         [
#             {"text": "I", "lemma_": "I"},
#             {"text": "must", "lemma_": "must"},
#             {"text": "go", "lemma_": "go"},
#         ],
#         # tuple tokens where lemma is at index 1
#         [("He", "he"), ("could", "could"), ("try", "try")],
#         # no modals
#         [("Read", "read"), ("the", "the"), ("book", "book")],
#         # whitespace tokens are excluded
#         [("may", "may"), (" ", ""), ("be", "be")],
#     ]

#     for toks in examples:
#         # Build representative sentence text for printing
#         words = []
#         for t in toks:
#             if isinstance(t, dict):
#                 words.append(t.get("text", ""))
#             elif isinstance(t, (list, tuple)) and len(t) >= 1:
#                 words.append(t[0])
#             else:
#                 words.append(str(t))
#         sentence_text = " ".join([w for w in words if w and w.strip()])
#         score = modality_score(toks)
#         print(f"Sentence: {sentence_text!r}  ->  Modality score: {score:.3f}")

