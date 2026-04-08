#!/usr/bin/env python3
# ================================
# FILE STATUS: FROZEN 
# Phase1_preprocessing : generate_phase1_cleaned.py
# Verified on: 2026-01-23 by Dhanshri 
# Do NOT modify this file
# ================================
"""
Generate Phase-1 cleaned dataset.

Loads `data/raw/mbti_kaggle.csv`, applies Phase-1 preprocessing pipeline,
and writes `data/processed/phase1_cleaned.json`.

This script is deterministic and performs no tokenization or shuffling.
"""
import os
import sys
import json
from typing import List, Dict

import pandas as pd


def _import_pipeline_modules(base_dir: str = "mbti-neuro-causal"):
    """Ensure phase1 preprocessing modules are importable and return them."""
    # Add mbti-neuro-causal directory to path so we can import phase1_preprocessing
    repo_path = os.path.abspath(base_dir)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    try:
        from phase1_preprocessing.basic_cleaning import basic_clean
        from phase1_preprocessing.grammar_normalization import grammar_normalize
        from phase1_preprocessing.sentence_segmentation import segment_sentences
    except Exception as e:
        raise ImportError(
            "Could not import Phase-1 preprocessing modules. Make sure 'mbti-neuro-causal/phase1_preprocessing' exists"
        ) from e

    return basic_clean, grammar_normalize, segment_sentences


def generate_cleaned(input_csv: str, output_json: str) -> int:
    """Load CSV, apply preprocessing, and save cleaned JSON. Returns number saved."""
    basic_clean, grammar_normalize, segment_sentences = _import_pipeline_modules()

    df = pd.read_csv(input_csv)

    # Determine columns: prefer Kaggle format ('posts','type')
    if "posts" in df.columns and "type" in df.columns:
        text_col = "posts"
        label_col = "type"
    elif "text" in df.columns and "label" in df.columns:
        text_col = "text"
        label_col = "label"
    else:
        # fallback to first two columns
        cols = list(df.columns)
        if len(cols) < 2:
            raise ValueError("Input CSV must have at least two columns (text, label)")
        text_col = cols[0]
        label_col = cols[1]

    out_dir = os.path.dirname(output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out: List[Dict] = []
    for idx, row in df.iterrows():
        raw_text = str(row[text_col])
        label = row[label_col]

        # Phase-1 pipeline (strict order)
        cleaned_1 = basic_clean(raw_text)
        cleaned_2 = grammar_normalize(cleaned_1)
        sentences = segment_sentences(cleaned_2)

        # Join sentences into one cleaned text
        cleaned_full = " ".join(s.strip() for s in sentences if s and s.strip())

        out.append({"id": int(idx) + 1, "text": cleaned_full, "mbti": label})

    # Save JSON list
    with open(output_json, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    return len(out)


if __name__ == "__main__":
    input_csv = os.path.join("data", "raw", "mbti_kaggle.csv")
    output_json = os.path.join("data", "processed", "phase1_cleaned.json")

    if not os.path.exists(input_csv):
        print(f"Input CSV not found: {input_csv}. Skipping generation.")
        sys.exit(0)

    n = generate_cleaned(input_csv, output_json)
    print(f"Saved {n} records to {output_json}")