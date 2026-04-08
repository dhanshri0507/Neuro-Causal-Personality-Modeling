# ================================
# FILE STATUS: FROZEN (spaCy Sentencizer)
# Phase1_preprocessing : sentence_segmentation.py
# Verified on: 2026-01-23 by Dhanshri time: 12.51pm
# Do NOT modify this file
# ================================
"""
Sentence-level segmentation using spaCy `en_core_web_sm`.

This module provides `segment_sentences(text) -> List[str]` which uses the
spaCy sentencizer to produce sentence spans. Per requirements:
- Use spaCy sentencizer (ensure it's present)
- Do NOT modify sentence text (returned strings are verbatim spans)
- Do NOT perform any cleaning or tokenization outside spaCy
"""
from typing import List
import spacy


# Load the spaCy model once at import time for efficiency.
# This will raise a clear error if the model is not installed.
_nlp = spacy.load("en_core_web_sm")

# Ensure the sentencizer is available in the pipeline. If not present,
# add the lightweight Sentencizer component (does not require dependency parse).
if "sentencizer" not in _nlp.pipe_names:
    try:
        # in newer spaCy versions adding by name works
        _nlp.add_pipe("sentencizer")
    except Exception:
        # fallback: import and add Sentencizer class explicitly
        from spacy.pipeline import Sentencizer

        _nlp.add_pipe("sentencizer", first=True)


def segment_sentences(text: str) -> List[str]:
    """
    Segment `text` into sentences using spaCy sentencizer.

    Returns a list of sentence strings extracted from the spaCy Doc. Sentences
    are returned exactly as produced by spaCy (no additional stripping or
    modification is performed here).
    """
    doc = _nlp(text)
    return [sent.text for sent in doc.sents]


# if __name__ == "__main__":
#     examples = [
#         "I went to the store. Then I bought milk. It was fine.",
#         "Dr. Smith arrived at 5 p.m. He left soon after.",
#         "This is a single short sentence",
#     ]

#     for ex in examples:
#         print("Input:", ex)
#         segs = segment_sentences(ex)
#         print("Segments:", len(segs))
#         for i, s in enumerate(segs, 1):
#             print(f"{i}: {repr(s)}")
#         print("---")

