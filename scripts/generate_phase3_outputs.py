#!/usr/bin/env python3
"""
Generate Phase-3 outputs (predictions, counterfactuals, explanations).

Loads data/processed/phase2_features.npz and runs causal reasoning pipeline
to produce per-document factual and counterfactual predictions and template
explanations.
"""
import os
import sys
import json
from typing import Dict, Any

import numpy as np
from tqdm import tqdm


def _ensure_repo_in_path(base_dir: str = "mbti-neuro-causal"):
    repo_path = os.path.abspath(base_dir)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def generate_phase3(input_npz: str, output_json: str, intervention_feature: str = "pronoun_ratio", lam: float = 0.3) -> int:
    """
    Run Phase-3 causal reasoning for each document and save outputs.
    Returns number of documents processed.
    """
    _ensure_repo_in_path()

    import torch

    from phase3_causal_reasoning.classifier.mbti_classifier import MBTIClassifier
    from phase2_representation.fusion.gated_fusion import GatedFusion
    from phase3_causal_reasoning.counterfactual.counterfactual_predictor import counterfactual_predictor
    from phase3_causal_reasoning.counterfactual.do_intervention import do_intervention
    from phase3_causal_reasoning.counterfactual.decision_logic import trait_flip
    from phase3_causal_reasoning.causal_analysis.probability_shift import probability_shift
    from phase3_causal_reasoning.causal_analysis.sensitivity_score import sensitivity_score
    from phase3_causal_reasoning.explanation.template_generator import generate_explanation

    data = np.load(input_npz, allow_pickle=True)
    C_arr = data["C"]
    S_arr = data["S"]
    J_arr = data["J"]
    labels = data["labels"] if "labels" in data.files else [None] * C_arr.shape[0]

    n = C_arr.shape[0]

    # Compute reference means for cognitive features (use across-dataset mean)
    # This will be used as x_ref for interventions.
    ref_means = np.mean(C_arr, axis=0) if n > 0 else np.zeros((C_arr.shape[1],), dtype=float)

    # Map intervention_feature to index in C vector.
    # Ordering: ratio means (5), count vars (2), binary props (3) => total len = 10
    base_order = [
        "pronoun_ratio",
        "modality_score",
        "emotion_intensity",
        "lexical_diversity",
        "readability_score",
        "negation_count",
        "sentence_length",
        "reasoning",
        "planning",
        "uncertainty",
    ]
    if intervention_feature not in base_order:
        raise ValueError(f"Intervention feature '{intervention_feature}' not recognized")
    feat_idx = base_order.index(intervention_feature)

    # Initialize models once (shared)
    classifier = MBTIClassifier(input_dim=512)
    fusion_model = GatedFusion(cognitive_dim=C_arr.shape[1], projection_dim=512, semantic_dim=768)

    outputs = []

    for i in tqdm(range(n), desc="Phase-3 documents"):
        C_vec = C_arr[i]
        S_vec = S_arr[i]
        J_vec = J_arr[i]
        label = labels[i] if i < len(labels) else None

        # Factual prediction
        J_t = torch.tensor(J_vec, dtype=torch.float32)
        factual = classifier(J_t)
        factual_probs = factual["probabilities"]
        factual_type = factual["type"]

        # Intervention: compute x_ref from ref_means for the selected feature
        x = float(C_vec[feat_idx])
        x_ref = float(ref_means[feat_idx])
        x_cf = do_intervention(x, x_ref, lam)

        # Build interventions dict for counterfactual_predictor
        interventions = {intervention_feature: (x_ref, lam)}

        # Use counterfactual_predictor to get counterfactual outputs (reuses fusion and classifier)
        C_tensor = torch.tensor(C_vec, dtype=torch.float32)
        cf_out = counterfactual_predictor(C_tensor, torch.tensor(S_vec, dtype=torch.float32), classifier, fusion_model, interventions)
        counterfactual_probs = cf_out["counterfactual_probabilities"]
        counterfactual_type = cf_out["counterfactual_type"]

        # Compute probability shifts per dimension
        prob_shifts = {}
        sensitivity_scores = {}
        flip_flags = {}
        for dim in ["IE", "NS", "TF", "JP"]:
            p0 = float(factual_probs.get(dim, 0.0))
            p1 = float(counterfactual_probs.get(dim, 0.0))
            dp = probability_shift(p0, p1)
            prob_shifts[dim] = dp
            # delta_x is x_cf - x (change in the single intervened feature)
            dx = x_cf - x
            sens = sensitivity_score(dp, dx)
            sensitivity_scores[dim] = sens
            flip_flags[dim] = trait_flip(p0, p1)

        explanation = generate_explanation(
            factual_type,
            factual_probs,
            counterfactual_type,
            counterfactual_probs,
            prob_shifts,
            sensitivity_scores,
            flip_flags,
        )

        outputs.append(
            {
                "id": int(i) + 1,
                "label": label,
                "factual_probs": factual_probs,
                "factual_type": factual_type,
                "counterfactual_probs": counterfactual_probs,
                "counterfactual_type": counterfactual_type,
                "probability_shifts": prob_shifts,
                "sensitivity_scores": sensitivity_scores,
                "trait_flip_flags": flip_flags,
                "explanation": explanation,
            }
        )

    # Save outputs
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as fh:
        json.dump(outputs, fh, ensure_ascii=False, indent=2)

    return n


if __name__ == "__main__":
    input_npz = os.path.join("data", "processed", "phase2_features.npz")
    output_json = os.path.join("data", "processed", "phase3_outputs.json")

    if not os.path.exists(input_npz):
        print(f"Input NPZ not found: {input_npz}. Skipping generation.")
        sys.exit(0)

    n = generate_phase3(input_npz, output_json)
    print(f"Processed {n} documents")
    # print one example
    with open(output_json, "r", encoding="utf-8") as fh:
        data = json.load(fh)
        if data:
            print("Example output (first document):")
            print(json.dumps(data[0], indent=2))

