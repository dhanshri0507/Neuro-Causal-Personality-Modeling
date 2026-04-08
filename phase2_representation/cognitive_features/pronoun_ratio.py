# ================================
# FILE STATUS: FROZEN
# Phase2_representation : pronoun_ratio.py
# Verified on: 2026-01-23 by Dhanshri time: 02.16pm
# Do NOT modify this file
# ================================
"""
Sentence-level pronoun ratio feature.

Computes pronoun_ratio(s_j) = (# tokens with POS == "PRON") / (total valid tokens)

The function accepts an iterable of token representations. Each token must
contain at least POS information; common accepted shapes are:
- spaCy Token objects (have .text and .pos_)
- tuples/lists like (text, pos, dep)
- dicts with keys "text" and "pos_"

Rules enforced:
- Count only linguistic tokens (exclude empty/whitespace tokens)
- Deterministic computation
- Do NOT modify tokens or text
- Return 0.0 for empty input or if no valid tokens
"""
from typing import Iterable, Tuple, Any


def _extract_text_pos(token: Any) -> Tuple[str, Any]:
    """
    Extract (text, pos) from a token-like object. If text cannot be
    determined, returns (None, None). Does not modify the token.
    """
    text = None
    pos = None

    # spaCy Token-like
    if hasattr(token, "text"):
        try:
            text = token.text
        except Exception:
            text = None
    # dict-like token
    if text is None and isinstance(token, dict):
        text = token.get("text") or token.get("token")
    # tuple/list-like token: (text, pos, ...)
    if text is None and isinstance(token, (list, tuple)) and len(token) >= 1:
        text = token[0]

    # POS extraction
    if hasattr(token, "pos_"):
        try:
            pos = token.pos_
        except Exception:
            pos = None
    if pos is None and isinstance(token, dict):
        pos = token.get("pos_") or token.get("pos")
    if pos is None and isinstance(token, (list, tuple)) and len(token) >= 2:
        pos = token[1]

    return text, pos


def pronoun_ratio(tokens: Iterable[Any]) -> float:
    """
    Compute pronoun ratio for a sentence represented as an iterable of tokens.

    Args:
        tokens: iterable of token-like objects

    Returns:
        float between 0.0 and 1.0
    """
    # Materialize iterable in order to avoid multiple traversals.
    token_list = list(tokens) if tokens is not None else []
    if not token_list:
        return 0.0

    valid_count = 0
    pron_count = 0

    for tok in token_list:
        text, pos = _extract_text_pos(tok)
        # Exclude non-linguistic tokens: require non-empty, non-whitespace text
        if not isinstance(text, str) or not text.strip():
            continue
        valid_count += 1
        if pos == "PRON":
            pron_count += 1

    if valid_count == 0:
        return 0.0

    return pron_count / valid_count


# if __name__ == "__main__":
#     # Sanity checks with example sentences represented as token tuples.
#     examples = [
#         # "I like it" => 2 pronouns out of 3 tokens -> 0.666...
#         [("I", "PRON", "nsubj"), ("like", "VERB", "ROOT"), ("it", "PRON", "dobj")],
#         # "He is happy" => 1 pronoun out of 3 -> 0.333...
#         [("He", "PRON", "nsubj"), ("is", "AUX", "cop"), ("happy", "ADJ", "acomp")],
#         # "Read the book" => 0 pronouns out of 3 -> 0.0
#         [("Read", "VERB", "ROOT"), ("the", "DET", "det"), ("book", "NOUN", "dobj")],
#         # include a whitespace token and empty token to ensure exclusion
#         [("I", "PRON", "nsubj"), (" ", "SPACE", ""), ("", "", "")],
#     ]

#     for toks in examples:
#         text = " ".join([t[0] for t in toks if isinstance(t, (list, tuple)) and t[0]])
#         ratio = pronoun_ratio(toks)
#         print(f"Sentence: {text!r}  →  Pronoun ratio: {ratio:.3f}")

