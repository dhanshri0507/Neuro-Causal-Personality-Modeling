# ================================
# FILE STATUS: FROZEN
# Phase1_preprocessing : split_posts.py
# Verified on: 2026-01-22 by Dhanshri time: 11.54pm
# Do NOT modify this file
# ================================
"""
Split posts utility.

This module implements `split_posts(text)` according to the Phase 1
documentation:
- split on the exact separator "|||"
- strip surrounding whitespace from each segment
- remove empty segments
- do NOT clean, lowercase, tokenize, or alter text beyond trimming
"""

from typing import List


def split_posts(text) -> List[str]:
    """
    Split the provided text into posts using the literal separator "|||" and
    return a list of non-empty, stripped segments.

    Rules (strict):
    - Split on the exact string "|||"
    - Strip leading/trailing whitespace from each segment
    - Remove segments that are empty after stripping
    - Do not perform any cleaning, lowercasing, or tokenization
    """
    # Use literal split on the separator (do not normalize or change text).
    parts = text.split("|||")

    # Strip whitespace and filter out empty segments.
    posts = [segment.strip() for segment in parts if segment.strip()]

    return posts


# if __name__ == "__main__":
#     # Small sanity check (non-exhaustive) to demonstrate behavior.
#     sample = "First post||| Second post  |||   |||Third post|||"
#     result = split_posts(sample)
#     print("Sanity check — found segments:", len(result))
#     for i, seg in enumerate(result, 1):
#         print(f"{i}: '{seg}'")

