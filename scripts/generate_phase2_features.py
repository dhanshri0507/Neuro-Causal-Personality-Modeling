#!/usr/bin/env python3
# ================================
# FILE STATUS: FROZEN 
# Phase1_preprocessing : generate_phase1_cleaned.py
# Verified on: 2026-01-27 by Dhanshri 
# Do NOT modify this file
# ================================
"""
Generate Phase-2 feature arrays.

Loads data/processed/phase1_cleaned.json, runs Phase-2 components to produce
document-level cognitive (C), semantic (S), and joint (J) vectors, and saves
them as NumPy arrays in data/processed/phase2_features.npz.

Deterministic, no training. Uses tqdm for progress.

Faster alternative if you already have S in phase2_features.npz (same row order as this JSON):
see scripts/refresh_cognitive_features_only.py to recompute C only without re-running BERT.
"""
import os
import sys
import json
from typing import List, Dict, Any

import numpy as np
import torch
from tqdm import tqdm


def _ensure_repo_in_path():
    repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def generate_phase2(input_json: str, output_npz: str) -> Dict[str, Any]:
    """
    Load phase1 cleaned JSON, run Phase-2 pipeline components, and save arrays.
    Returns dict with shapes saved.
    """
    _ensure_repo_in_path()

    # Import Phase-2 pieces
    import spacy

    from phase2_representation.aggregation.rule_based_aggregation import aggregate_cognitive_features, RATIO_FEATURES, COUNT_FEATURES, BINARY_FEATURES
    from phase2_representation.semantic_encoder.bert_tokenizer import bert_tokenize
    from phase2_representation.semantic_encoder.bert_encoder import encode_sentence
    from phase2_representation.fusion.gated_fusion import GatedFusion
    from phase1_preprocessing.sentence_segmentation import segment_sentences
    from phase2_representation.cognitive_features.pronoun_ratio import pronoun_ratio
    from phase2_representation.cognitive_features.modality_score import modality_score
    from phase2_representation.cognitive_features.emotion_nrc import emotion_intensity
    from phase2_representation.cognitive_features.lexical_diversity import lexical_diversity
    from phase2_representation.cognitive_features.readability_metrics import readability_score
    from phase2_representation.cognitive_features.negation_count import negation_count
    from phase2_representation.cognitive_features.sentence_length import sentence_length
    from phase2_representation.cognitive_features.cognitive_markers import cognitive_markers

    # Read input JSON
    with open(input_json, "r", encoding="utf-8") as fh:
        docs = json.load(fh)

    n = len(docs)
    C_list = []
    S_list = []
    J_list = []
    labels: List[Any] = []

    nlp = spacy.load("en_core_web_sm")

    nrc_path = os.path.join(os.path.dirname(__file__), "..", "data", "nrc_lexicon.txt")
    nrc_set = set()
    if os.path.isfile(nrc_path):
        from phase2_representation.cognitive_features.emotion_nrc import load_nrc_lexicon

        nrc_set = load_nrc_lexicon(nrc_path)
    else:
        print(f"Warning: NRC lexicon not found at {nrc_path}; emotion_intensity will be 0.")

    # One GatedFusion for all docs (reproducible). Per-document fusion used different random
    # weights and broke J. Run `python -m training.train_hybrid_torch` to overwrite J with trained fusion.
    torch.manual_seed(42)
    fusion = GatedFusion(cognitive_dim=10, projection_dim=512, semantic_dim=768)
    fusion.eval()

    for doc in tqdm(docs, desc="Processing documents"):
        text = doc.get("text", "")
        label = doc.get("mbti", None)
        labels.append(label)

        # Segment sentences
        sentences = segment_sentences(text)

        # Per-sentence cognitive features
        sentence_feature_dicts = []
        sentence_embeddings = []
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
            markers = cognitive_markers(s_doc)
            feats.update(markers)
            sentence_feature_dicts.append(feats)

            # Semantic embedding
            tokenized = bert_tokenize(sent)
            emb = encode_sentence(tokenized)  # (768,)
            sentence_embeddings.append(emb)

        # Aggregate cognitive features -> aggregated dict -> build C vector
        aggregated = aggregate_cognitive_features(sentence_feature_dicts)
        C_values = []
        for feat in RATIO_FEATURES:
            key = f"{feat}_mean"
            C_values.append(float(aggregated.get(key, 0.0)))
        for feat in COUNT_FEATURES:
            key = f"{feat}_var"
            C_values.append(float(aggregated.get(key, 0.0)))
        for feat in BINARY_FEATURES:
            key = f"{feat}_prop"
            C_values.append(float(aggregated.get(key, 0.0)))
        C_vec = np.array(C_values, dtype=np.float32)
        C_list.append(C_vec)

        # Document semantic vector: mean pool of sentence BERT embeddings (stable vs random attention w per doc)
        if sentence_embeddings:
            H = torch.stack(sentence_embeddings, dim=0)
            with torch.no_grad():
                S_vec = H.mean(dim=0).detach().cpu().numpy()
        else:
            S_vec = np.zeros(768, dtype=np.float32)
        S_list.append(S_vec)

        C_t = torch.tensor(C_vec, dtype=torch.float32)
        S_t = torch.tensor(S_vec, dtype=torch.float32)
        with torch.no_grad():
            J_t = fusion(C_t, S_t)
        J_list.append(J_t.detach().cpu().numpy())

    C_arr = np.stack(C_list, axis=0) if C_list else np.empty((0, len(C_values)), dtype=np.float32)
    S_arr = np.stack(S_list, axis=0) if S_list else np.empty((0, 768), dtype=np.float32)
    J_arr = np.stack(J_list, axis=0) if J_list else np.empty((0, 512), dtype=np.float32)

    os.makedirs(os.path.dirname(output_npz), exist_ok=True)
    np.savez(output_npz, C=C_arr, S=S_arr, J=J_arr, labels=np.array(labels, dtype=object))

    return {"n": n, "C_shape": C_arr.shape, "S_shape": S_arr.shape, "J_shape": J_arr.shape}


if __name__ == "__main__":
    input_json = os.path.join("data", "processed", "phase1_cleaned.json")
    output_npz = os.path.join("data", "processed", "phase2_features.npz")

    if not os.path.exists(input_json):
        print(f"Input JSON not found: {input_json}. Skipping generation.")
        sys.exit(0)

    info = generate_phase2(input_json, output_npz)
    print(f"Processed {info['n']} documents")
    print("C shape:", info["C_shape"])
    print("S shape:", info["S_shape"])
    print("J shape:", info["J_shape"])
