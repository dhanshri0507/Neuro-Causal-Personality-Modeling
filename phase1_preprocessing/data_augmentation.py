# ================================
# FILE STATUS: FROZEN 
# Phase1_preprocessing : data_augmentation.py
# Verified on: 2026-01-23 by Dhanshri time: 01.42pm
# Do NOT modify this file
# ================================
"""
Label-safe, rule-based data augmentation.

Provides:
- rule_paraphrase(text: str) -> str
- class_balance(df: pandas.DataFrame) -> pandas.DataFrame
- expand_sentence(sentence: str, min_words=6) -> str

Deterministic, rule-based logic only. No ML, no randomness (selection is
deterministic by hashing where variation is needed).
"""
from typing import Dict, List
import pandas as pd


# --- Paraphrase rules (exact string replacements) ---
PARAPHRASE_RULES: Dict[str, str] = {
    # belief / cognition    
    "I think": "I believe",
    "I prefer": "I tend to prefer",
    "I feel": "I experience",
    "I experience": "I feel",

    # intention / desire
    "I want to": "I intend to",
    "I intend to": "I want to",

    # uncertainty
    "I do not know": "I am uncertain",
    "I am uncertain": "I do not know",

    # reasoning connectors (bidirectional)
    "because": "since",
    "since": "because",

    # preference framing
    "I prefer": "I tend to prefer",
    "I tend to prefer": "I prefer",
}

# Expansion templates (neutral, reasoning-safe)
EXPANSION_TEMPLATES: List[str] = [
    "because it helps me focus better",
    "because it feels more comfortable to me",
    "because this approach is more effective",
    "because it allows me to think more clearly",
    "since this works better for me",
    "since it makes the process simpler",
    "as it helps me stay organized",
    "as this feels more manageable",
    "because it reduces unnecessary effort",
    "since it aligns better with my way of thinking",
    "as it leads to clearer outcomes",
    "because it supports better decision-making",
]


def rule_paraphrase(text: str) -> str:
    """
    Apply exact-string paraphrase replacements from PARAPHRASE_RULES.

    Deterministic and exact (case-sensitive) replacements are performed in
    the order of items in PARAPHRASE_RULES. This mirrors the documentation's
    simple rule-based replace approach.
    """
    for k, v in PARAPHRASE_RULES.items():
        text = text.replace(k, v)
    return text


def class_balance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Duplicate real samples from minority classes until target_count is reached.

    Rules:
    - Input df has columns ['posts', 'type'].
    - Target count = int(mean samples per class).
    - Identify minority classes as those with count < target_count.
    - Duplicate existing real samples (deterministically cycling through them) until the target count is reached for each minority class.
    - Apply rule_paraphrase ONLY to duplicated samples (not originals).
    - Preserve original samples and labels.
    - Do not rebalance majority classes.
    """
    counts = df["type"].value_counts()
    target_count = counts.max()


    # If target_count is less than or equal to current max, still proceed; only minority < target_count
    minority_types = counts[counts < target_count].index.tolist()

    new_rows = []
    for mbti in minority_types:
        subset = df[df["type"] == mbti].reset_index(drop=True)
        current_count = len(subset)
        if current_count == 0:
            continue
        # Deterministic duplication: cycle through existing rows by index order
        idx = 0
        while current_count < target_count:
            row = subset.iloc[idx % len(subset)]
            duplicated_post = rule_paraphrase(row["posts"])
            new_rows.append({"posts": duplicated_post, "type": row["type"]})
            current_count += 1
            idx += 1

    if new_rows:
        balanced_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        balanced_df = df.copy()

    return balanced_df


def expand_sentence(sentence: str, min_words: int = 6) -> str:
    """
    If sentence length < min_words and no causal connector exists, append one
    neutral template from EXPANSION_TEMPLATES deterministically.

    - Do not change casing of the sentence.
    - Do not introduce new meaning; templates are neutral.
    - Deterministic selection: choose template index by hashing sentence.
    """
    words = sentence.split()
    if len(words) >= min_words:
        return sentence

    lower = sentence.lower()
    if "because" in lower or "since" in lower:
        return sentence

    # Deterministic index selection
    idx = abs(hash(sentence)) % len(EXPANSION_TEMPLATES)
    template = EXPANSION_TEMPLATES[idx]

    # Append template preserving existing punctuation/casing.
    # If sentence already ends with punctuation, append with a space.
    # Otherwise also append with a space.
    return sentence + " " + template


# if __name__ == "__main__":
#     # Small sanity check
#     import pandas as pd

#     df = pd.DataFrame(
#         [
#             {"posts": "I think this works", "type": "INTJ"},
#             {"posts": "I prefer quiet places", "type": "INTJ"},
#             {"posts": "I like meeting new people", "type": "ENFP"},
#         ]
#     )

#     print("Original counts:")
#     print(df["type"].value_counts())

#     balanced = class_balance(df)
#     print("\nBalanced counts:")
#     print(balanced["type"].value_counts())

#     print("\nOriginal rows (unchanged):")
#     print(df)

#     print("\nNew rows added (duplicates paraphrased):")
#     # Show rows in balanced that were not in original (by index)
#     if len(balanced) > len(df):
#         added = balanced.iloc[len(df) :]
#         print(added)
#     else:
#         print("No duplicates needed.")

#     print("\nParaphrase example:")
#     sample = "I think"
#     print("Before:", sample)
#     print("After :", rule_paraphrase(sample))

#     print("\nSentence expansion examples:")
#     short = "I read"
#     print("Short before:", short)
#     print("Short after :", expand_sentence(short))
#     has_cause = "I do this because it helps"
#     print("Has cause before:", has_cause)
#     print("Has cause after :", expand_sentence(has_cause))

