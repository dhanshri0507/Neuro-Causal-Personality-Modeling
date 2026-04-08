# ================================
# FILE STATUS: FROZEN
# Phase1_preprocessing : basic_cleaning.py
# Verified on: 2026-01-23 by Dhanshri time: 00.16am
# Do NOT modify this file
# ================================
"""
Basic cleaning utilities (non-linguistic hygiene).

Implements `basic_clean(text)` according to Phase 1 documentation:
Operations (all must be applied):
- remove HTML / markup
- remove URLs
- remove Reddit artifacts (u/, r/)
- normalize repeated characters (3+ -> 2)
- normalize punctuation (collapse repeated punctuation like "!!!" -> "!")
- remove emojis
- normalize whitespace

Rules:
- Do NOT remove pronouns
- Do NOT lowercase
- Do NOT expand contractions

This module intentionally avoids tokenization, sentence splitting, or any
semantic changes.
"""
from typing import List
import re


def basic_clean(text: str) -> str:
    """
    Perform basic non-linguistic cleaning on `text`.

    The function follows the documented pipeline and does not perform any
    cleaning beyond the listed operations.
    """
    # 1) Remove HTML/markup (non-greedy)
    text = re.sub(r"<.*?>", " ", text)

    # 2) Remove URLs (http(s) and www.)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3) Remove Reddit artifacts like u/username or r/subreddit
    # Use word boundaries to avoid accidental removals inside words.
    text = re.sub(r"\bu/[\w\-]+|\br/[\w\-]+", " ", text)

    # 4) Normalize repeated characters (3 or more -> 2)
    # e.g., "soooo" -> "soo", "loooool" -> "lool"
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # 5) Normalize punctuation: collapse repeated punctuation to single one
    # e.g., "Really!!!" -> "Really!"
    text = re.sub(r"([!?.]){2,}", r"\1", text)

    # 6) Remove emojis and other pictographs.
    # Cover common emoji/codepoint ranges as well as miscellaneous symbols.
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\u2600-\u26FF"          # Misc symbols
        "\u2700-\u27BF"          # Dingbats
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub("", text)

    # 7) Normalize whitespace (collapse runs to single space) and strip ends
    text = re.sub(r"\s+", " ", text).strip()

    return text


# if __name__ == "__main__":
#     # Sanity check demonstrating operations (non-exhaustive).
#     sample = (
#         "<br>I read this https://example.com soooo tired!!! "
#         "u/user r/subreddit 😊😊   Extra   spaces"
#     )
#     cleaned = basic_clean(sample)
#     print("Original:")
#     print(sample)
#     print("\nCleaned:")
#     print(cleaned)

