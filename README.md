# 🧠 Neuro-Causal Personality Prediction

> **A three-phase neuro-causal framework for explainable MBTI personality type prediction from free-form text**, combining cognitive linguistic features with transformer-based semantics via a gated fusion architecture.

---

## 📌 Overview

This project presents a novel **neuro-causal personality modeling system** that predicts Myers–Briggs Type Indicator (MBTI) personality types from unstructured natural language text. Unlike traditional deep-learning–only approaches, the framework integrates **cognitive feature engineering** (psycholinguistic signals) with **BERT-based semantic embeddings** through a learned **Gated Fusion** module, and then applies **causal reasoning** to produce human-interpretable counterfactual explanations.

The system is deployed as a full-stack web application with a React/TypeScript frontend and a FastAPI backend.

---

## 🗺️ System Architecture

```
Raw Text Input
      │
      ▼
┌─────────────────────────────────┐
│  PHASE 1 – Preprocessing        │
│  • Basic cleaning               │
│  • Grammar normalization        │
│  • Sentence segmentation        │
└──────────────┬──────────────────┘
               │
      ┌────────┴────────┐
      ▼                 ▼
┌──────────────┐  ┌───────────────────────┐
│ Cognitive    │  │ Semantic Encoder       │
│ Feature      │  │ (BERT + Attention      │
│ Extraction   │  │  Aggregator)           │
│ C_doc ∈ R¹⁰ │  │ S_doc ∈ R⁷⁶⁸          │
└──────┬───────┘  └──────────┬────────────┘
       │                     │
       └──────────┬──────────┘
                  ▼
      ┌───────────────────────┐
      │  PHASE 2 – Gated      │
      │  Fusion Module        │
      │  J = g⊙C' + (1-g)⊙S' │
      │  J ∈ R⁵¹²             │
      └───────────┬───────────┘
                  │
                  ▼
      ┌───────────────────────┐
      │  PHASE 3 – Causal     │
      │  Reasoning            │
      │  • 4-head Classifier  │
      │  • Counterfactual      │
      │  • Causal DAG analysis│
      └───────────┬───────────┘
                  │
                  ▼
        MBTI Type + Explanation
```

---

## 🔬 Research Contributions

This work is documented in three companion papers:

| Paper | Venue | File |
|-------|-------|------|
| *Neuro-Causal Personality Prediction* | Master's Project (Full) | `MP.tex` |
| *Neuro-Causal Personality Prediction* | IEEE Transactions on Affective Computing | `ieee_tac.tex` |
| *Neuro-Causal Personality Prediction* | arXiv Preprint | `arxiv.tex` |

**Key Results** (Hybrid Gated Fusion model):

| MBTI Dimension | F1 Score |
|----------------|----------|
| I/E            | 0.5146   |
| N/S            | 0.5119   |
| T/F            | 0.5867   |
| J/P            | 0.5271   |
| **Average**    | **0.535** |

> Outperforms BERT-only (0.519) and simple concatenation (0.533) baselines.

---

## 📁 Project Structure

```
Major_Project/
│
├── mbti-neuro-causal/              # Main source code
│   ├── phase1_preprocessing/       # Phase 1: Text cleaning & normalisation
│   │   ├── basic_cleaning.py
│   │   ├── grammar_normalization.py
│   │   ├── sentence_segmentation.py
│   │   ├── tokenization.py
│   │   ├── data_augmentation.py
│   │   ├── input_validation.py
│   │   └── split_posts.py
│   │
│   ├── phase2_representation/      # Phase 2: Feature extraction & fusion
│   │   ├── cognitive_features/     # Psycholinguistic feature extractors
│   │   │   ├── pronoun_ratio.py
│   │   │   ├── modality_score.py
│   │   │   ├── lexical_diversity.py
│   │   │   ├── readability_metrics.py
│   │   │   ├── negation_count.py
│   │   │   ├── sentence_length.py
│   │   │   ├── cognitive_markers.py
│   │   │   ├── emotion_nrc.py
│   │   │   └── sentiment_vader.py
│   │   ├── semantic_encoder/       # BERT encoder + attention aggregation
│   │   │   ├── bert_tokenizer.py
│   │   │   ├── bert_encoder.py
│   │   │   └── attention_aggregator.py
│   │   ├── fusion/
│   │   │   └── gated_fusion.py     # GatedFusion neural module (FROZEN)
│   │   ├── aggregation/
│   │   └── representation_pipeline.py
│   │
│   ├── phase3_causal_reasoning/    # Phase 3: Classification & explanations
│   │   ├── classifier/
│   │   │   └── mbti_classifier.py  # 4-head sigmoid classifier (FROZEN)
│   │   ├── counterfactual/
│   │   │   └── counterfactual_predictor.py
│   │   ├── causal_analysis/
│   │   │   ├── probability_shift.py
│   │   │   └── sensitivity_score.py
│   │   ├── explanation/
│   │   └── visualization/
│   │
│   ├── api/                        # FastAPI backend
│   │   ├── main.py                 # API entry point (FROZEN)
│   │   ├── schemas.py
│   │   ├── config_loader.py
│   │   └── configs/
│   │
│   ├── training/                   # Training & evaluation scripts
│   │   ├── experiments.py
│   │   ├── baseline_comparison.py
│   │   ├── hybrid_torch_model.py
│   │   ├── train_hybrid_torch.py
│   │   ├── optimize_joint_framework.py
│   │   ├── stability_analysis.py
│   │   └── run_all_experiments.py
│   │
│   ├── scripts/                    # Data pipeline & visualisation scripts
│   │   ├── generate_phase1_cleaned.py
│   │   ├── generate_phase2_features.py
│   │   ├── generate_phase3_outputs.py
│   │   ├── generate_visualizations.py
│   │   ├── compute_feature_reference_means.py
│   │   ├── refresh_cognitive_features_only.py
│   │   └── visualize_latent_trajectory.py
│   │
│   ├── frontend/                   # React + TypeScript web UI
│   │   ├── src/
│   │   │   ├── pages/
│   │   │   ├── components/
│   │   │   ├── constants/
│   │   │   ├── App.tsx
│   │   │   └── main.tsx
│   │   ├── package.json
│   │   └── vite.config.ts
│   │
│   ├── data/                       # Datasets and lexicons
│   │   ├── raw/                    # Raw MBTI Kaggle dataset
│   │   ├── processed/              # Cleaned & tokenised data
│   │   ├── stats/                  # Feature statistics
│   │   └── nrc_lexicon.txt         # NRC Emotion Lexicon
│   │
│   ├── models/                     # Trained model checkpoints
│   │   ├── hybrid_fusion_trained.pt
│   │   └── hybrid_train_report.json
│   │
│   ├── results/                    # Experimental outputs
│   │   ├── baselines/
│   │   ├── figures/
│   │   ├── stability/
│   │   ├── stability_comprehensive/
│   │   ├── optimization/
│   │   └── Prototype_Interface_images/
│   │
│   └── requirements.txt
│
├── ArXIV/                          # ArXiv submission bundle
├── Documentation_Files/            # Project documentation & reports
│
├── MP.tex                          # Full project paper (Springer svproc)
├── ieee_tac.tex                    # IEEE TAC formatted paper
├── arxiv.tex                       # ArXiv preprint
├── sample_input.txt                # Example input text
│
└── *.png                           # Publication figures
    ├── system_architecture.png
    ├── causal_dag.png
    ├── data_flow.png
    ├── confusion_matrices_grid.png
    ├── roc_curves_grid.png
    ├── fig4_sensitivity_heatmap.png
    └── fig5_stability_curve.png
```

---

## 🗂️ Dataset

The dataset used in this project is the MBTI dataset from Kaggle. 

**Due to size constraints, the full dataset and large generated features (e.g., `phase2_features_backup_random_J.npz`) are not included in this repository.**

A small sample dataset is provided in `mbti-neuro-causal/Dataset/Sample/` for demonstration purposes.

To run the complete pipeline, please download the full dataset from:
[Kaggle MBTI Type Dataset](https://www.kaggle.com/datasets/datasnaek/mbti-type)

---

## ⚙️ Prerequisites

### Python Backend
- Python 3.9+
- PyTorch 2.0.1
- CUDA (optional, CPU inference is supported)

### Frontend
- Node.js 18+
- npm / yarn

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd Major_Project/mbti-neuro-causal
```

### 2. Set Up the Python Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Required NLP Models

```bash
# spaCy English model
python -m spacy download en_core_web_sm

# NLTK data (for VADER, tokenisation)
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

### 5. Prepare the Data

Place the [MBTI Kaggle dataset](https://www.kaggle.com/datasnaek/mbti-type) (`mbti_1.csv`) in `data/raw/`, then run:

```bash
python scripts/generate_phase1_cleaned.py
python scripts/generate_phase2_features.py
python scripts/compute_feature_reference_means.py
```

---

## 🏃 Running the Application

### Start the Backend API

```bash
# From mbti-neuro-causal/
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: `http://localhost:8000/docs`

### Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend available at: `http://localhost:5173`

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/predict` | MBTI personality prediction |
| `POST` | `/counterfactual` | Counterfactual explanation generation |

### Example: Predict Personality

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love deep conversations and solving complex problems..."}'
```

**Response:**
```json
{
  "mbti": "INTJ",
  "confidence": 0.72,
  "probabilities": {"IE": 0.28, "NS": 0.31, "TF": 0.78, "JP": 0.35},
  "explanation": "...",
  "cognitive_features": { ... },
  "sentence_attribution": [ ... ]
}
```

---

## 🧪 Running Experiments

```bash
# Run all training/evaluation experiments
python training/run_all_experiments.py

# Baseline comparisons (BERT-only, concat, hybrid)
python training/baseline_comparison.py

# Stability analysis
python training/stability_analysis.py

# Generate publication figures
python scripts/generate_visualizations.py
```

---

## 🧠 Cognitive Features

The system extracts **10 psycholinguistic features** per sentence:

| Feature | Description |
|---------|-------------|
| Pronoun Ratio | I/we vs. total tokens |
| Modality Score | Modal verb frequency |
| Emotion Intensity | NRC Emotion Lexicon score |
| Lexical Diversity | Type-token ratio (TTR) |
| Readability | Flesch–Kincaid score |
| Negation Count | Negation token frequency |
| Sentence Length | Mean token count |
| Reasoning Proportion | Causal/logical connectives |
| Planning Proportion | Future-oriented language |
| Uncertainty Proportion | Hedging language |

---

## 🔍 Causal Reasoning

The Phase-3 module provides:

- **MBTI Classification**: Four independent sigmoid heads over J ∈ ℝ⁵¹²
- **Counterfactual Prediction**: Interventions on cognitive features to produce alternate personality profiles
- **Sensitivity Scores**: Measures how much each dimension probability changes per unit of feature intervention
- **Trait Flip Detection**: Identifies whether an intervention crosses the decision boundary (probability shift > 0.2 → 0.4 threshold)

---

## 📊 Results Summary

| Model | Avg F1 | IE F1 | NS F1 | TF F1 | JP F1 |
|-------|--------|-------|-------|-------|-------|
| BERT-only (semantic) | 0.519 | 0.510 | 0.478 | 0.578 | 0.512 |
| Concatenation | 0.533 | 0.504 | 0.486 | 0.619 | 0.522 |
| **Hybrid Gated Fusion** | **0.535** | **0.515** | **0.512** | **0.587** | **0.527** |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Language model | `bert-base-uncased` (HuggingFace Transformers) |
| Deep learning | PyTorch 2.0.1 |
| NLP pipeline | spaCy 3.7, NLTK, VADER, textstat |
| Backend API | FastAPI + Uvicorn |
| Frontend | React 19 + TypeScript + Vite |
| Data viz | Recharts (UI), Matplotlib / Seaborn / Plotly (research) |
| Dataset | [Kaggle MBTI-500](https://www.kaggle.com/datasnaek/mbti-type) |
| Emotion lexicon | NRC Emotion Lexicon |


## 🔒 License

This project is for academic research purposes. All rights reserved © 2026.
