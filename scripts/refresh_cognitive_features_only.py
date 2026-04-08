#!/usr/bin/env python3
"""
Refresh only cognitive vectors (C) and optional cheap J — reuse existing BERT semantics (S).

Full phase-2 regeneration is slow because it encodes every sentence with BERT. If you already
have `phase2_features.npz` with correct S and labels (same row order as phase1_cleaned.json),
this script:

  - Re-runs spaCy + psycholinguistic features (including NRC emotion if lexicon present)
  - Keeps S from the npz unchanged
  - Rebuilds J with the shared seeded GatedFusion (same as generate_phase2_features.py)

Typical runtime: orders of magnitude faster than full BERT re-encode (often ~1–3 hours vs 1–2 days
on CPU for large corpora — depends on doc count and hardware).

Usage (from repo root `mbti-neuro-causal`):

  python scripts/refresh_cognitive_features_only.py

  python scripts/refresh_cognitive_features_only.py --npz-in data/processed/phase2_features.npz \\
      --npz-out data/processed/phase2_refreshed.npz

Then train hybrid on the output npz:

  python -m training.train_hybrid_torch --npz data/processed/phase2_refreshed.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_path() -> None:
    r = _repo_root()
    if r not in sys.path:
        sys.path.insert(0, r)


def main() -> None:
    _ensure_path()
    import spacy

    from phase1_preprocessing.sentence_segmentation import segment_sentences
    from phase2_representation.aggregation.rule_based_aggregation import (
        RATIO_FEATURES,
        COUNT_FEATURES,
        BINARY_FEATURES,
        aggregate_cognitive_features,
    )
    from phase2_representation.cognitive_features.cognitive_markers import cognitive_markers
    from phase2_representation.cognitive_features.emotion_nrc import emotion_intensity, load_nrc_lexicon
    from phase2_representation.cognitive_features.lexical_diversity import lexical_diversity
    from phase2_representation.cognitive_features.modality_score import modality_score
    from phase2_representation.cognitive_features.negation_count import negation_count
    from phase2_representation.cognitive_features.pronoun_ratio import pronoun_ratio
    from phase2_representation.cognitive_features.readability_metrics import readability_score
    from phase2_representation.cognitive_features.sentence_length import sentence_length
    from phase2_representation.fusion.gated_fusion import GatedFusion

    parser = argparse.ArgumentParser(description="Recompute C only; keep S from npz.")
    parser.add_argument(
        "--input-json",
        default=os.path.join(_repo_root(), "data", "processed", "phase1_cleaned.json"),
    )
    parser.add_argument(
        "--npz-in",
        default=os.path.join(_repo_root(), "data", "processed", "phase2_features.npz"),
    )
    parser.add_argument(
        "--npz-out",
        default=None,
        help="Defaults to --npz-in (overwrite). Use another path to keep the original file.",
    )
    args = parser.parse_args()
    npz_out = args.npz_out or args.npz_in

    if not os.path.isfile(args.input_json):
        raise FileNotFoundError(args.input_json)
    if not os.path.isfile(args.npz_in):
        raise FileNotFoundError(args.npz_in)

    with open(args.input_json, "r", encoding="utf-8") as f:
        docs = json.load(f)

    data = np.load(args.npz_in, allow_pickle=True)
    S = np.asarray(data["S"], dtype=np.float32)
    labels_np = data["labels"]
    labels = list(labels_np)

    n = len(docs)
    if S.shape[0] != n or len(labels) != n:
        raise ValueError(
            f"Row count mismatch: json={n}, S.shape[0]={S.shape[0]}, len(labels)={len(labels)}"
        )

    json_labels = [d.get("mbti") for d in docs]
    mismatches = sum(1 for a, b in zip(json_labels, labels) if a != b)
    if mismatches:
        raise ValueError(
            f"Labels differ from npz in {mismatches}/{n} rows — same order as when S was built is required."
        )

    nrc_path = os.path.join(_repo_root(), "data", "nrc_lexicon.txt")
    nrc_set: set = load_nrc_lexicon(nrc_path) if os.path.isfile(nrc_path) else set()
    if not nrc_set:
        print("Warning: NRC lexicon missing or empty; emotion_intensity will be 0.")

    nlp = spacy.load("en_core_web_sm")
    torch.manual_seed(42)
    fusion = GatedFusion(cognitive_dim=10, projection_dim=512, semantic_dim=768)
    fusion.eval()

    C_list: List[np.ndarray] = []
    J_list: List[np.ndarray] = []

    for i, doc in enumerate(tqdm(docs, desc="C only (no BERT)")):
        text = doc.get("text", "")
        sentences = segment_sentences(text)
        sentence_feature_dicts: List[Dict[str, Any]] = []
        for sent in sentences:
            s_doc = nlp(sent)
            feats: Dict[str, Any] = {}
            feats["pronoun_ratio"] = pronoun_ratio(s_doc)
            feats["modality_score"] = modality_score(s_doc)
            feats["emotion_intensity"] = emotion_intensity(s_doc, nrc_set)
            feats["lexical_diversity"] = lexical_diversity(s_doc)
            feats["readability_score"] = readability_score(sent)
            feats["negation_count"] = negation_count(s_doc)
            feats["sentence_length"] = sentence_length(s_doc)
            feats.update(cognitive_markers(s_doc))
            sentence_feature_dicts.append(feats)

        aggregated = aggregate_cognitive_features(sentence_feature_dicts)
        C_values: List[float] = []
        for feat in RATIO_FEATURES:
            C_values.append(float(aggregated.get(f"{feat}_mean", 0.0)))
        for feat in COUNT_FEATURES:
            C_values.append(float(aggregated.get(f"{feat}_var", 0.0)))
        for feat in BINARY_FEATURES:
            C_values.append(float(aggregated.get(f"{feat}_prop", 0.0)))
        C_vec = np.array(C_values, dtype=np.float32)
        C_list.append(C_vec)

        S_vec = S[i]
        C_t = torch.tensor(C_vec, dtype=torch.float32)
        S_t = torch.tensor(S_vec, dtype=torch.float32)
        with torch.no_grad():
            J_t = fusion(C_t, S_t)
        J_list.append(J_t.cpu().numpy())

    C_arr = np.stack(C_list, axis=0)
    J_arr = np.stack(J_list, axis=0)

    os.makedirs(os.path.dirname(npz_out) or ".", exist_ok=True)
    np.savez(npz_out, C=C_arr, S=S, J=J_arr, labels=np.array(labels, dtype=object))
    print(f"Wrote {npz_out}  C{C_arr.shape} S{S.shape} J{J_arr.shape}")


if __name__ == "__main__":
    main()
