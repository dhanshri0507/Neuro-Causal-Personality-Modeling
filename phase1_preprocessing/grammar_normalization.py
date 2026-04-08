# ================================
# FILE STATUS: FROZEN (EXPERIMENTAL / ABLATION-CONTROLLED)
# Phase1_preprocessing : grammar_normalization.py
# Verified on: 2026-01-23 by Dhanshri time: 00.35am
# Do NOT modify this file
# ================================
"""
Rule-based grammar normalization pipeline.

This module implements `grammar_normalize(text: str) -> str` following the
strict, ordered pipeline defined in the Phase 1 documentation. The function
is deterministic and rule-based; it performs only the transformations listed
in the requirements and avoids lowercasing the entire sentence or any ML.
"""
import re
from typing import Dict


# --- Rule dictionaries (must be present) ---
CONTRACTIONS: Dict[str, str] = {
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "didn't": "did not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "wouldn't": "would not",
    "i'm": "i am",
    "i've": "i have",
    "it's": "it is",
    "ain't": "is not",
}

SLANG: Dict[str, str] = {
    "idk": "I do not know",
    "imo": "in my opinion",
    "btw": "by the way",
    "lol": "",
    "u": "you",
}

WEAK_NEGATION: Dict[str, str] = {
    "rarely": "do not often",
    "hardly": "do not usually",
    "seldom": "do not frequently",
    "unsure": "not sure",
    "uncertain": "not certain",
}

MODALITY: Dict[str, str] = {
    "maybe": "might",
    "probably": "might",
    "i guess": "I think it might",
    "have to": "must",
    "need to": "must",
}

# Phrases that imply an omitted first-person subject; mapping value is
# the verb phrase to use after inserting "I ".
IMPLICIT_PHRASES: Dict[str, str] = {
    "prefer": "prefer",
    "want to": "want to",
    "intend to": "intend to",
    "intend": "intend",
    "need to": "need to",
    "like": "like",
    "i guess": "I think it might",
}


def _preserve_case_replace(match, replacement):
    text = match.group(0)

    # Safety: empty replacement
    if not replacement:
        return replacement

    # All caps
    if text.isupper():
        return replacement.upper()

    # Capitalized (only first letter)
    if text[0].isupper():
        if len(replacement) == 1:
            return replacement.upper()
        return replacement[0].upper() + replacement[1:]

    return replacement



def _replace_from_dict(text: str, mapping: Dict[str, str]) -> str:
    """Replace whole-word keys from mapping in text using case-preserving replace."""
    for key, val in mapping.items():
        # Word-boundary pattern (works with multi-word keys too)
        pattern = re.compile(r"\b" + re.escape(key) + r"\b", flags=re.IGNORECASE)
        text = pattern.sub(lambda m: _preserve_case_replace(m, val), text)
    return text


def grammar_normalize(text: str) -> str:
    """
    Apply rule-based grammar normalization in the required order.

    Pipeline order (exact):
    1. Expand contractions
    2. Slang expansion
    3. Implicit phrasing normalization (add "I " when sentence starts with verb)
    4. Weak negation normalization
    5. Explicit negation normalization (ensure explicit "not"/"do not"/"cannot")
    6. Modality normalization
    7. Case normalization for first-person singular (i -> I)
    8. Sentence completeness enforcement (prepend subject+verb if <4 tokens)
    9. Reasoning explicitness (insert "because" when causal phrasing exists)
    10. Punctuation enforcement (ensure ending punctuation)
    """
    # Work on a single sentence string (caller is responsible for segmentation).
    original = text

    # 1) Expand contractions (do not lowercase entire text)
    text = _replace_from_dict(text, CONTRACTIONS)

    # 2) Slang expansion
    text = _replace_from_dict(text, SLANG)

    # 3) Implicit phrasing normalization
    # If sentence begins with an implicit verb phrase (no explicit subject),
    # insert "I " before it. We check for common pronouns to avoid false positives.
    pronoun_pattern = re.compile(r"^\s*(?:I|You|He|She|They|We|It)\b", flags=re.IGNORECASE)
    if not pronoun_pattern.search(text):
        for key, val in IMPLICIT_PHRASES.items():
            # match phrase at sentence start
            start_pattern = re.compile(r"^\s*" + re.escape(key) + r"\b", flags=re.IGNORECASE)
            if start_pattern.search(text):
                # Insert "I " before the matched phrase, preserve capitalization of the phrase
                text = start_pattern.sub(lambda m: "I " + _preserve_case_replace(m, val), text, count=1)
                break

    # 4) Weak negation normalization
    text = _replace_from_dict(text, WEAK_NEGATION)

    # 5) Explicit negation normalization
    # Ensure contracted negations have been expanded; if any residual "n't" tokens
    # exist (rare), convert them to explicit " not".
    text = re.sub(r"n['’]t\b", " not", text)

    # 6) Modality normalization
    text = _replace_from_dict(text, MODALITY)

    # 7) Case normalization ONLY for first-person singular token 'i' -> 'I'
    # Use word boundaries to avoid changing other words.
    text = re.sub(r"\bi\b", "I", text)

    # 8) Sentence completeness enforcement (if too short, prepend minimal subject+verb)
    token_count = len(text.split())
    if token_count < 4:
        text = "I am " + text

    # 9) Reasoning explicitness (simple rule)
    # If there is a sentence-like delimiter and the second part is sufficiently
    # long and "because" is not already present, join with " because ".
    if "." in text and "because" not in text:
        parts = text.split(".")
        if len(parts) == 2 and len(parts[1].strip()) > 3:
            text = parts[0].strip() + " because " + parts[1].strip()

    # 10) Punctuation enforcement: ensure sentence ends with ., !, or ?
    if not text.endswith((".", "!", "?")):
        text = text + "."

    return text


# if __name__ == "__main__":
#     examples = [
#         "can't believe it",
#         "prefer working alone",
#         "i'm unsure about that",
#         "maybe this will work",
#     ]

#     for ex in examples:
#         print("Original :", ex)
#         print("Normalized:", grammar_normalize(ex))
#         print("---")

