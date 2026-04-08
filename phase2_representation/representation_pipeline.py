# ================================
# FILE STATUS: FROZEN
# Phase2_representation : representation_pipeline.py
# Verified on: 2026-01-23 by dhanshri time: 11.20pm
# Do NOT modify this file
# ================================
"""
Phase-2 Representation Pipeline

Orchestrates:
1) sentence-level cognitive feature extraction
2) aggregation -> C_doc
3) BERT encoding per sentence -> H (T, 768)
4) attention aggregation -> S_doc (768,)
5) gated fusion -> J (512,)

Pure forward pipeline; no training or parameter updates.
"""
from typing import List, Dict, Any
import torch
import spacy

# Cognitive features
from .cognitive_features.pronoun_ratio import pronoun_ratio
from .cognitive_features.modality_score import modality_score
from .cognitive_features.emotion_nrc import emotion_intensity
from .cognitive_features.lexical_diversity import lexical_diversity
from .cognitive_features.readability_metrics import readability_score
from .cognitive_features.negation_count import negation_count
from .cognitive_features.sentence_length import sentence_length
from .cognitive_features.cognitive_markers import cognitive_markers

# Aggregation
from .aggregation.rule_based_aggregation import (
    aggregate_cognitive_features,
    RATIO_FEATURES,
    COUNT_FEATURES,
    BINARY_FEATURES,
)

# Semantic encoder pieces
from .semantic_encoder.bert_tokenizer import bert_tokenize
from .semantic_encoder.bert_encoder import encode_sentence
from .semantic_encoder.attention_aggregator import AttentionAggregator

# Fusion
from .fusion.gated_fusion import GatedFusion


def representation_pipeline(sentences: List[str]) -> torch.Tensor:
    """
    Run the forward representation pipeline and return joint vector J (512,).
    Also prints intermediate shapes.
    """
    if not isinstance(sentences, list):
        raise TypeError("sentences must be a list of strings")

    # 1. Prepare spaCy for token-level cognitive features
    nlp = spacy.load("en_core_web_sm")

    # Use a small mock NRC set for emotion matching (production should load full lexicon)
    nrc_mock = {"happy", "sad", "angry"}

    sentence_feature_dicts: List[Dict[str, Any]] = []
    sentence_embeddings = []

    for sent in sentences:
        doc = nlp(sent)

        # Cognitive features per sentence
        feats: Dict[str, Any] = {}
        feats["pronoun_ratio"] = pronoun_ratio(doc)
        feats["modality_score"] = modality_score(doc)
        feats["emotion_intensity"] = emotion_intensity(doc, nrc_mock)
        feats["lexical_diversity"] = lexical_diversity(doc)
        feats["readability_score"] = readability_score(sent)
        feats["negation_count"] = negation_count(doc)
        feats["sentence_length"] = sentence_length(doc)
        markers = cognitive_markers(doc)
        feats.update(markers)

        sentence_feature_dicts.append(feats)

        # Semantic encoding
        tokenized = bert_tokenize(sent)
        emb = encode_sentence(tokenized)  # (768,)
        sentence_embeddings.append(emb)

    # Stack sentence embeddings -> H (T, 768)
    if sentence_embeddings:
        H = torch.stack(sentence_embeddings, dim=0)
    else:
        H = torch.empty(0, 768)

    print(f"Stage: encoded H shape: {tuple(H.shape)}")

    # 3. Aggregate cognitive features -> aggregated dict
    aggregated = aggregate_cognitive_features(sentence_feature_dicts)

    # Build C_doc vector in deterministic order:
    # ratio_mean (in RATIO_FEATURES order), count_var (COUNT_FEATURES), binary_prop (BINARY_FEATURES)
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

    C_doc = torch.tensor(C_values, dtype=torch.float32)
    print(f"Stage: aggregated C_doc shape: {tuple(C_doc.shape)}")

    # 4. Attention aggregation over H -> S_doc
    if H.numel() == 0:
        # empty: zero semantic vector
        S_doc = torch.zeros(768)
    else:
        att = AttentionAggregator(hidden_size=768)
        S_doc = att(H)  # (768,)
    print(f"Stage: semantic S_doc shape: {tuple(S_doc.shape)}")

    # 5. Gated fusion
    fusion = GatedFusion(cognitive_dim=C_doc.shape[0], projection_dim=512, semantic_dim=768)
    J = fusion(C_doc, S_doc)  # (512,)
    print(f"Stage: fused J shape: {tuple(J.shape)}")

    return J


# if __name__ == "__main__":
#     examples = [
#         "I am happy with this project because it helps me focus.",
#         "I might prefer to work alone sometimes.",
#         "The methodology is complex and requires careful consideration.",
#     ]

#     J = representation_pipeline(examples)
#     print("Final joint vector shape:", tuple(J.shape))

