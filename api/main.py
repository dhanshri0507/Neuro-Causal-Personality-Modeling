# ================================
# FILE STATUS: FROZEN
# Do NOT modify unless API changes
# ================================

import random
import numpy as np
import torch

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import TextInput, MBTIResponse, CounterfactualResponse
from .config_loader import load_intervention_limits

# Phase-1 imports (preprocessing)
from phase1_preprocessing.basic_cleaning import basic_clean
from phase1_preprocessing.grammar_normalization import grammar_normalize
from phase1_preprocessing.sentence_segmentation import segment_sentences

# Phase-2 imports (representation components)
from phase2_representation.aggregation.rule_based_aggregation import (
    aggregate_cognitive_features,
    RATIO_FEATURES,
    COUNT_FEATURES,
    BINARY_FEATURES,
)
from phase2_representation.semantic_encoder.bert_tokenizer import bert_tokenize
from phase2_representation.semantic_encoder.bert_encoder import encode_sentence
from phase2_representation.semantic_encoder.attention_aggregator import AttentionAggregator
from phase2_representation.fusion.gated_fusion import GatedFusion
from phase2_representation.cognitive_features.pronoun_ratio import pronoun_ratio
from phase2_representation.cognitive_features.modality_score import modality_score
from phase2_representation.cognitive_features.lexical_diversity import lexical_diversity
from phase2_representation.cognitive_features.readability_metrics import readability_score
from phase2_representation.cognitive_features.negation_count import negation_count
from phase2_representation.cognitive_features.sentence_length import sentence_length
from phase2_representation.cognitive_features.cognitive_markers import cognitive_markers
from phase2_representation.cognitive_features.emotion_nrc import (load_nrc_lexicon, emotion_intensity, )


# Phase-3 imports (classifier / counterfactual / explanation)
from phase3_causal_reasoning.classifier.mbti_classifier import MBTIClassifier
from phase3_causal_reasoning.counterfactual.counterfactual_predictor import counterfactual_predictor

app = FastAPI(title="MBTI Neuro-Causal API")

# Global models (initialized at startup)
_classifier: MBTIClassifier = None
_fusion_model: GatedFusion = None
_attention_aggregator: AttentionAggregator = None
_nlp = None
_nrc_lexicon = None


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    global _classifier, _fusion_model, _attention_aggregator, _nlp, _nrc_lexicon

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    _classifier = MBTIClassifier(input_dim=512)
    _fusion_model = GatedFusion(
        cognitive_dim=10,
        projection_dim=512,
        semantic_dim=768
    )
    _attention_aggregator = AttentionAggregator(hidden_size=768)

    import spacy
    _nlp = spacy.load("en_core_web_sm")
    
    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    NRC_PATH = os.path.join(BASE_DIR, "data", "nrc_lexicon.txt")

    _nrc_lexicon = load_nrc_lexicon(NRC_PATH)

    print(f"✅ NRC lexicon loaded: {len(_nrc_lexicon)} entries")
    print("✅ Startup complete")
    print(f"   spaCy loaded: {_nlp is not None}")
    print(f"   NRC size: {len(_nrc_lexicon)}")
    

@app.get("/health")
def health():
    return {"status": "ok"}

def _compute_representation_from_text(text: str):
    """Run Phase-1 and Phase-2 steps to compute C_doc, S_doc, J tensor."""
    # Phase-1
    cleaned = basic_clean(text)
    normalized = grammar_normalize(cleaned)
    sentences = segment_sentences(normalized)

    # Phase-2: per-sentence features and embeddings
    sentence_feature_dicts = []
    sentence_embeddings = []

    for sent in sentences:
        s_doc = _nlp(sent)

        emotion_val = (
            emotion_intensity(s_doc, _nrc_lexicon)
            if _nrc_lexicon is not None
            else 0.0
        )

        # cognitive features (per sentence)
        feats = {
            "pronoun_ratio": pronoun_ratio(s_doc),
            "modality_score": modality_score(s_doc),
            "emotion_intensity": emotion_val,
            "lexical_diversity": lexical_diversity(s_doc),
            "readability_score": readability_score(sent),
            "negation_count": negation_count(s_doc),
            "sentence_length": sentence_length(s_doc),
        }
        markers = cognitive_markers(s_doc)
        feats.update(markers)
        sentence_feature_dicts.append(feats)

        # semantic embedding
        tokenized = bert_tokenize(sent)
        emb = encode_sentence(tokenized)  # torch.Tensor (768,)
        sentence_embeddings.append(emb)

    # Aggregate cognitive features -> deterministic C vector
    aggregated = aggregate_cognitive_features(sentence_feature_dicts)

    C_values = []
    for feat in RATIO_FEATURES:
        C_values.append(float(aggregated.get(f"{feat}_mean", 0.0)))
    for feat in COUNT_FEATURES:
        C_values.append(float(aggregated.get(f"{feat}_var", 0.0)))
    for feat in BINARY_FEATURES:
        C_values.append(float(aggregated.get(f"{feat}_prop", 0.0)))

    C_doc = torch.tensor(C_values, dtype=torch.float32)

    # Semantic aggregation via attention
    sentence_attribution = []

    if sentence_embeddings:
        H = torch.stack(sentence_embeddings, dim=0)  # (T, 768)
        S_doc, alphas = _attention_aggregator(H, return_weights=True)
        # build sentence attribution (order preserved)
        sentence_attribution = [
            {
                "sentence": sent,
                "weight": float(alpha)
            }
            for sent, alpha in zip(sentences, alphas)
        ]
    else:
        S_doc = torch.zeros(768)

    # Joint fused vector
    J = _fusion_model(C_doc, S_doc)  # (512,)
    
    def _safe_mean(key):
        vals = [f.get(key, 0.0) for f in sentence_feature_dicts]
        return float(np.mean(vals)) if vals else 0.0

    cognitive_features = {
        "pronoun_ratio": _safe_mean("pronoun_ratio"),
        "modality_score": _safe_mean("modality_score"),
        "negation_count": _safe_mean("negation_count"),
        "emotion_intensity": _safe_mean("emotion_intensity"),
        "reasoning_prop": _safe_mean("reasoning"),
        "planning_prop": _safe_mean("planning"),
        "uncertainty_prop": _safe_mean("uncertainty"),
        "lexical_diversity": _safe_mean("lexical_diversity"),
    }


    return C_doc, S_doc, J, cognitive_features, sentence_attribution

@app.post("/predict", response_model=MBTIResponse)
def predict(payload: TextInput):
    """
    Main MBTI prediction endpoint.
    Accepts text input and returns MBTI type with probabilities and explanation.
    """
    try:
        # Phase-1 and Phase-2: compute representation
        C_doc, S_doc, J, cognitive_features, sentence_attribution = \
            _compute_representation_from_text(payload.text)
        
        # Phase-3: classify
        result = _classifier(J)
        
        # Extract results
        mbti_type = result["type"]
        confidence = result["confidence"]
        probabilities = result["probabilities"]
        
        # Generate explanation
        explanation = (
            f"Based on cognitive and semantic analysis, your text suggests "
            f"an {mbti_type} personality type with {confidence:.1%} confidence. "
            f"Key indicators include cognitive patterns and linguistic features."
        )
        
        print(f"✅ Prediction successful: {mbti_type} (confidence: {confidence:.2f})")
        
        # -------- Normalize cognitive features for radar visualization --------
        feature_ranges = {
            "pronoun_ratio": (0.0, 0.15),
            "modality_score": (0.0, 0.20),
            "negation_count": (0.0, 0.10),
            "emotion_intensity": (0.0, 1.0),
            "reasoning_prop": (0.0, 0.30),
            "planning_prop": (0.0, 0.30),
            "uncertainty_prop": (0.0, 0.30),
            "lexical_diversity": (0.4, 1.0),
        }

        normalized_cognitive_features = {}
        for k, v in cognitive_features.items():
            lo, hi = feature_ranges.get(k, (0.0, 1.0))
            if hi > lo:
                normalized_cognitive_features[k] = max(
                    0.0, min(1.0, (v - lo) / (hi - lo))
                )
            else:
                normalized_cognitive_features[k] = 0.0


        return MBTIResponse(
            mbti=mbti_type,
            confidence=confidence,
            probabilities=probabilities,
            explanation=explanation,
            cognitive_features=normalized_cognitive_features,
            sentence_attribution=sentence_attribution,
        )
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/counterfactual", response_model=CounterfactualResponse)
def counterfactual(payload: TextInput):
    try:
        # Load intervention config
        limits = load_intervention_limits()
        
        # Phase-1 and Phase-2
        C_doc, S_doc, J, cognitive_features, _ = \
            _compute_representation_from_text(payload.text)
        
        # Factual prediction
        factual_out = _classifier(J)
        factual_probs = factual_out["probabilities"]
        factual_type = factual_out["type"]

        # Prepare intervention parameters from config
        if "reference_means" not in limits:
            raise HTTPException(status_code=500, detail="intervention reference_means not found in config")
            
        ref_means = limits["reference_means"]
        # Determine feature to intervene
        feature = (
            payload.intervention_feature
            if payload.intervention_feature
            else limits.get("default_feature", "pronoun_ratio")
        )

        lam = (
            float(payload.intervention_lambda)
            if payload.intervention_lambda is not None
            else float(limits.get("default_lambda", 0.3))
        )

        # Convert cognitive_features dict to proper format for predictor
        C_doc_dict = {
            "pronoun_ratio": cognitive_features.get("pronoun_ratio", 0.0),
            "modality_score": cognitive_features.get("modality_score", 0.0),
            "negation_count": cognitive_features.get("negation_count", 0.0),
            "emotion_intensity": cognitive_features.get("emotion_intensity", 0.0),
            "lexical_diversity": cognitive_features.get("lexical_diversity", 0.0),
            "readability_score": 0.0,  # Not in cognitive_features
            "sentence_length": 0.0,    # Not in cognitive_features
            "reasoning": cognitive_features.get("reasoning_prop", 0.0),
            "planning": cognitive_features.get("planning_prop", 0.0),
            "uncertainty": cognitive_features.get("uncertainty_prop", 0.0),
        }   
        
        # DEBUG: Print what we're sending
        print(f"🔍 Counterfactual Debug:")
        print(f"  - Input text length: {len(payload.text)}")
        print(f"  - Feature to intervene: {feature}")
        print(f"  - Lambda: {lam}")
        print(f"  - Reference mean: {ref_means.get(feature, 0.0)}")
        
        # IMPORTANT: Use base feature name (without suffix) for intervention
        # The counterfactual_predictor expects base names like "pronoun_ratio"
        # Get reference statistics
        current_val = C_doc_dict.get(feature, 0.0)
        ref_mean = ref_means.get(feature, 0.0)

        # Clamp lam strictly for causal operator
        lam_cf = max(0.2, min(0.4, lam))

        # Move reference value instead of lam
        target_val = current_val + (ref_mean - current_val) * lam_cf

        # Optional: extra amplification (safe)
        target_val = current_val + 3.0 * (target_val - current_val)

        # Clamp feature range if needed
        if feature in {
            "pronoun_ratio",
            "modality_score",
            "emotion_intensity",
            "lexical_diversity",
            "reasoning",
            "planning",
            "uncertainty",
        }:
            target_val = max(0.0, min(1.0, target_val))

        interventions = {
            feature: (
                target_val,
                lam_cf  # ✅ ALWAYS within [0.2, 0.4]
            )
        }
        print(
            f"[CF DEBUG] {feature}: current={current_val:.3f}, "
            f"ref={ref_mean:.3f}, target={target_val:.3f}"
        )

        
        # Call counterfactual predictor with the dict format
        cf_out = counterfactual_predictor(
            C_doc=C_doc_dict,
            S_doc=S_doc,
            classifier=_classifier,
            fusion_model=_fusion_model,
            interventions=interventions,
        )
        
        if cf_out is None:
            # Log the issue
            print("⚠️ counterfactual_predictor returned None")
            
            # 1. Pick most uncertain dimension
            dim_name, dim_prob = min(
                factual_probs.items(),
                key=lambda x: abs(x[1] - 0.5)
            )

            # 2. Flip probability
            cf_probs = factual_probs.copy()
            cf_probs[dim_name] = 1.0 - dim_prob

            # 3. Flip MBTI letter
            type_chars = list(factual_type)
            if dim_name == "IE":
                type_chars[0] = "E" if dim_prob < 0.5 else "I"
            elif dim_name == "NS":
                type_chars[1] = "S" if dim_prob < 0.5 else "N"
            elif dim_name == "TF":
                type_chars[2] = "F" if dim_prob < 0.5 else "T"
            elif dim_name == "JP":
                type_chars[3] = "P" if dim_prob < 0.5 else "J"

            return CounterfactualResponse(
                factual_type=factual_type,
                counterfactual_type="".join(type_chars),
                factual_probabilities=factual_probs,
                counterfactual_probabilities=cf_probs,
                counterfactual_explanation="Counterfactual analysis could not be generated for this input (optimization failed). Falling back to heuristic flip.",
            )
            
        # --------- Δ Probability computation ----------
        delta_probabilities = {
            dim: cf_out["counterfactual_probabilities"].get(dim, p) - p
            for dim, p in factual_probs.items()
        }
        
        # --------- Sensitivity computation ----------
        feature_delta = abs(target_val - current_val)
        sensitivity = {
            dim: abs(delta) / feature_delta if feature_delta > 1e-6 else 0.0
            for dim, delta in delta_probabilities.items()
        }
        
        # --------- Decision Logic: Trait Flip ----------
        trait_flip = {}

        for dim, p_f in factual_probs.items():
            p_cf = cf_out["counterfactual_probabilities"].get(dim, p_f)

            flip = (
                (p_f >= 0.6 and p_cf <= 0.4) or
                (p_f <= 0.4 and p_cf >= 0.6)
            )   

            trait_flip[dim] = flip

        # --------- Counterfactual Explanation (Natural Language) ----------

# Identify most affected dimension
        most_changed_dim = max(
            delta_probabilities.items(),
            key=lambda x: abs(x[1])
        )[0]

        delta_val = delta_probabilities[most_changed_dim]
        sens_val = sensitivity.get(most_changed_dim, 0.0)
        flipped = trait_flip.get(most_changed_dim, False)

        if flipped:
            flip_text = (
                f"This intervention caused a decisive shift in the {most_changed_dim} dimension, "
                f"crossing the decision boundary and resulting in a personality trait flip."
            )
        else:
            flip_text = (
                f"Although the dominant letter in the {most_changed_dim} dimension shifted, "
                f"the probability change did not cross the calibrated stability boundary. "
                f"This indicates a marginal shift rather than a robust personality transformation."
            )


        counterfactual_explanation = (
            f"When intervening on the feature '{feature}', the model observed a change "
            f"of Δ = {delta_val:.3f} in the {most_changed_dim} probability. "
            f"The sensitivity score for this dimension was {sens_val:.3f}. "
            f"{flip_text} "
            f"This reflects the model’s design choice to treat personality traits "
            f"as continuous spectra rather than rigid categories."
        )

        
        # ✅ RETURN ONCE, AFTER ALL COMPUTATION
        response = CounterfactualResponse(
            factual_type=factual_type,
            counterfactual_type=cf_out["counterfactual_type"],
            factual_probabilities=factual_probs,
            counterfactual_probabilities=cf_out["counterfactual_probabilities"],
            delta_probabilities=delta_probabilities,
            sensitivity=sensitivity,
            trait_flip=trait_flip,
            counterfactual_explanation=counterfactual_explanation,
        )
        return response
        
    except Exception as e:
        # Log error for monitoring
        print(f"❌ Counterfactual error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Fallback to factual results
        try:
            _, _, J, _, _ = _compute_representation_from_text(payload.text)
            factual_out = _classifier(J)
            factual_type = factual_out["type"]
            factual_probs = factual_out["probabilities"]
            
            # Create a simple counterfactual
            cf_probs = factual_probs.copy()
            cf_probs["IE"] = 1.0 - cf_probs["IE"]
            
            # Flip I/E
            cf_type_chars = list(factual_type)
            cf_type_chars[0] = "E" if cf_type_chars[0] == "I" else "I"
            counterfactual_type = "".join(cf_type_chars)
            
            return CounterfactualResponse(
                factual_type=factual_type,
                counterfactual_type=counterfactual_type,
                factual_probabilities=factual_probs,
                counterfactual_probabilities=cf_probs,
                counterfactual_explanation=f"Error during counterfactual generation: {str(e)}. Returning heuristic flip.",
            )
        except:
            # Ultimate fallback
            return CounterfactualResponse(
                factual_type="ENFJ",
                counterfactual_type="INFP",
                factual_probabilities={"IE": 0.7, "NS": 0.6, "TF": 0.8, "JP": 0.5},
                counterfactual_probabilities={"IE": 0.3, "NS": 0.6, "TF": 0.8, "JP": 0.5},
                counterfactual_explanation="Critical error in counterfactual service. Returning default fallback.",
            )