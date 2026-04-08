
# ================================
# FILE STATUS: FROZEN (DUAL-PATH)
# Phase1_preprocessing : tokenization.py
# Verified on: 2026-01-23 by Dhanshri time: 01.16pm
# Do NOT modify this file
# ================================
"""
Tokenization utilities for dual-path processing.

Provides:
- spacy_parse(sentence: str) -> List[Tuple[str, str, str]]
- bert_tokenize(sentence: str) -> Dict[str, Tensor] (tokenizer output)
- selective_stopword_filter(tokens) -> filtered tokens (same structure)

Notes:
- This module does NOT perform any cleaning, lowercasing, or normalization.
- It uses spaCy `en_core_web_sm` and HuggingFace `bert-base-cased`.
"""
from typing import List, Tuple, Union, Dict, Any
import spacy
from transformers import BertTokenizer


# Load spaCy model once.
_nlp = spacy.load("en_core_web_sm")

# Load BERT tokenizer once.
_bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# Stopwords to remove on the cognitive path: articles and filler words only.
# Must keep pronouns, negation, modals, conditionals.
STOPWORDS_REMOVE = {"a", "an", "the", "uh", "um"}


def spacy_parse(sentence: str) -> List[Tuple[str, str, str]]:
    """
    Parse a sentence with spaCy and return a list of triples:
    (token.text, token.pos_, token.dep_)

    - Does NOT modify the input text.
    - Does NOT lowercase or remove punctuation.
    """
    doc = _nlp(sentence)
    return [(token.text, token.pos_, token.dep_) for token in doc]


def bert_tokenize(sentence: str) -> Dict[str, Any]:
    """
    Tokenize the sentence using HuggingFace BertTokenizer.from_pretrained("bert-base-cased").

    Returns the tokenizer output dictionary (e.g., input_ids, attention_mask).
    Uses PyTorch tensors via `return_tensors="pt"`.
    """
    return _bert_tokenizer(sentence, return_tensors="pt", truncation=True)


def selective_stopword_filter(tokens: List[Union[str, Tuple[str, ...]]]) -> List[Union[str, Tuple[str, ...]]]:
    """
    Filter out ONLY articles and filler words from a token sequence for the
    cognitive path. The function preserves the input token structure:
    - If tokens are strings, returns list of strings.
    - If tokens are tuples (token_text, ...), returns tuples unchanged except dropped.

    Rules:
    - Remove tokens whose lowercase text is in STOPWORDS_REMOVE.
    - Keep pronouns, negation tokens, modals, and conditionals (we do not remove them).
    - Do not modify POS tags or dependency labels.
    """
    if not tokens:
        return []

    filtered = []
    first = tokens[0]
    is_tuple = isinstance(first, tuple)

    if is_tuple:
        # Expect each token as (text, pos, dep, ...)
        for tok in tokens:
            text = tok[0]
            if text.lower() in STOPWORDS_REMOVE:
                continue
            filtered.append(tok)
    else:
        # tokens are plain strings
        for text in tokens:
            if text.lower() in STOPWORDS_REMOVE:
                continue
            filtered.append(text)

    return filtered


# if __name__ == "__main__":
#     # Small sanity check demonstrating all three functions.
#     example = "I read a book, uh I think it's great."

#     print("=== spaCy parse ===")
#     parsed = spacy_parse(example)
#     for t in parsed:
#         print(t)

#     print("\n=== selective stopword filter (on token texts) ===")
#     token_texts = [t[0] for t in parsed]
#     filtered_texts = selective_stopword_filter(token_texts)
#     print(filtered_texts)

#     print("\n=== selective stopword filter (on spaCy token tuples) ===")
#     filtered_tuples = selective_stopword_filter(parsed)
#     for t in filtered_tuples:
#         print(t)

#     print("\n=== BERT tokenization ===")
#     bert_out = bert_tokenize(example)
#     # Show input_ids shape and a small sample
#     input_ids = bert_out["input_ids"]
#     attention_mask = bert_out.get("attention_mask", None)
#     print("input_ids shape:", input_ids.shape)
#     if attention_mask is not None:
#         print("attention_mask shape:", attention_mask.shape)

