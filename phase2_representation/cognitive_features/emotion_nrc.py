# ================================
# FILE STATUS: FROZEN
# Phase2_representation : emotion_nrc.py
# Verified on: 2026-01-23 by Dhanshri time: 06.41pm
# Do NOT modify this file
# ================================
"""
NRC-based emotion intensity feature.

This module provides utilities to load an NRC-style lexicon (as a set of
lemmas) and compute sentence-level emotion intensity using spaCy tokens.

Functions:
- load_nrc_lexicon(path) -> Set[str]
- emotion_intensity(doc, nrc_set) -> float

Rules followed:
- use spaCy tokens and token.lemma_.lower() for matching
- treat NRC as binary indicator (present / absent)
- deterministic and simple; returns 0.0 for empty sentences
"""
from typing import Set, Iterable, Any


def load_nrc_lexicon(path: str) -> Set[str]:
    """
    Load an NRC lexicon file into a set of lemma strings (lowercased).

    The loader is permissive: it reads the file line-by-line, ignores empty
    lines and lines starting with '#', and takes the first token on each line
    as the lemma. This keeps the loader compatible with common simple
    lexicon formats.
    """
    lemmas: Set[str] = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if not parts:
                continue
            lemma = parts[0].lower()
            lemmas.add(lemma)
    return lemmas


def emotion_intensity(doc: Iterable[Any], nrc_set: Set[str]) -> float:
    """
    Compute emotion intensity for a spaCy Doc or iterable of tokens.

    Args:
        doc: spaCy Doc or iterable of spaCy Token-like objects
        nrc_set: set of emotion lemmas (lowercased)

    Returns:
        float: (# tokens whose lemma is in nrc_set) / (total valid tokens)
            returns 0.0 if no valid tokens
    """
    if doc is None:
        return 0.0

    total = 0
    hits = 0

    for token in doc:
        # Expect spaCy tokens; be tolerant of other shapes but require a .lemma_
        lemma = None
        try:
            lemma = getattr(token, "lemma_", None)
        except Exception:
            lemma = None

        # Exclude empty / whitespace tokens
        text = None
        try:
            text = getattr(token, "text", None)
        except Exception:
            text = None
        if not isinstance(text, str) or not text.strip():
            continue

        total += 1
        if isinstance(lemma, str) and lemma.lower() in nrc_set:
            hits += 1

    if total == 0:
        return 0.0
    return hits / total


# if __name__ == "__main__":
#     # Small sanity check using a mock NRC set
#     import spacy

#     nrc_mock = {"happy", "sad", "angry"}
#     nlp = spacy.load("en_core_web_sm")

#     examples = [
#         "I am happy today",
#         "I feel a bit sad and lonely",
#         "This is a neutral sentence with no emotion words",
#         "Angry people shout sometimes",
#     ]

#     for sent in examples:
#         doc = nlp(sent)
#         score = emotion_intensity(doc, nrc_mock)
#         print(f"Sentence: {sent!r}  ->  Emotion intensity: {score:.3f}")

