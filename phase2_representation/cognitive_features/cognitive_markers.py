# ================================
# FILE STATUS: FROZEN
# Phase2_representation : cognitive_markers.py
# Verified on: 2026-01-23 by Dhanshri time: 06.53pm
# Do NOT modify this file
# ================================
"""
Cognitive markers (binary indicators) computed per sentence.

Markers:
- reasoning: 1 if any token lemma in {"because", "therefore", "if"}
- planning: 1 if any token lemma in {"plan", "decide", "goal"}
- uncertainty: 1 if any token lemma in {"unsure", "might", "maybe"}

Rules:
- Use token.lemma_.lower() for matching
- Existential checks only (presence -> 1, absence -> 0)
- Do not count occurrences or use POS/dep
"""
from typing import Iterable, Any, Dict


REASONING_SET = {"because", "therefore", "if"}
PLANNING_SET = {"plan", "decide", "goal"}
UNCERTAINTY_SET = {"unsure", "might", "maybe"}


def cognitive_markers(doc: Iterable[Any]) -> Dict[str, int]:
    """
    Compute cognitive marker presence for a sentence represented as an iterable
    of spaCy Token-like objects.

    Returns a dict with keys 'reasoning', 'planning', 'uncertainty' mapped to
    0 or 1.
    """
    # Default values
    result = {"reasoning": 0, "planning": 0, "uncertainty": 0}
    if doc is None:
        return result

    # Track presence with booleans; stop early when all found
    found_reasoning = False
    found_planning = False
    found_uncertainty = False

    for token in doc:
        try:
            lemma = getattr(token, "lemma_", None)
        except Exception:
            lemma = None
        if not isinstance(lemma, str):
            continue
        l = lemma.lower()
        if not found_reasoning and l in REASONING_SET:
            found_reasoning = True
            result["reasoning"] = 1
        if not found_planning and l in PLANNING_SET:
            found_planning = True
            result["planning"] = 1
        if not found_uncertainty and l in UNCERTAINTY_SET:
            found_uncertainty = True
            result["uncertainty"] = 1
        if found_reasoning and found_planning and found_uncertainty:
            break

    return result


# if __name__ == "__main__":
#     # Sanity checks using spaCy
#     import spacy

#     nlp = spacy.load("en_core_web_sm")
#     examples = [
#         ("This happened because I tried", "reasoning"),
#         ("I plan to finish this project", "planning"),
#         ("I might go later", "uncertainty"),
#         ("A plain neutral sentence", "none"),
#     ]

#     for sent, label in examples:
#         doc = nlp(sent)
#         markers = cognitive_markers(doc)
#         print(f"Sentence: {sent!r} -> {markers}  (expected: {label})")

