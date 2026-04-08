# ================================
# FILE STATUS: FROZEN
# Phase1_preprocessing : input_validation.py
# Verified on: 2026-01-22 by Dhanshri time: 11.45pm
# Do NOT modify this file
# ================================

"""
Input length validation utilities.

This module implements a single function `validate_user_input` which only
validates the word-count of a raw text input according to the Phase 1
documentation. It intentionally does NOT perform any cleaning, sentence
splitting, or external-library tokenization.
"""

def validate_user_input(text, min_words=150, max_words=500):
    if not isinstance(text, str):
        raise TypeError("Input must be a string.") 
    """
    Validate user input length.

    Rules (follow documentation strictly):
    - If number of words < min_words: raise ValueError
    - If number of words > max_words: truncate to max_words words
    - Do NOT clean the text or split sentences; use simple whitespace split.

    Args:
        text (str): Raw user input (unchanged by this function).
        min_words (int): Minimum allowed words (default 150).
        max_words (int): Maximum allowed words (default 500).

    Returns:
        str: The original text (possibly truncated to max_words).

    Raises:
        ValueError: If the input contains fewer than min_words words.
    """
    # Count words using simple whitespace splitting (do not perform cleaning).
    word_count = len(text.split())

    if word_count < min_words:
        # Message follows documentation guidance.
        raise ValueError(f"Input too short. Minimum {min_words} words required.")

    if word_count > max_words:
        # Truncate by taking the first max_words tokens from the whitespace split.
        text = " ".join(text.split()[:max_words])

    return text


# if __name__ == "__main__":
#     # Small sanity check (non-exhaustive). This intentionally avoids cleaning.
#     sample = "word " * 160
#     try:
#         validated = validate_user_input(sample)
#         print(f"Sanity check passed — returned text has {len(validated.split())} words.")
#     except ValueError as exc:
#         print(f"Sanity check failed: {exc}")

