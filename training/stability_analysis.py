#!/usr/bin/env python3
"""
Comprehensive Stability & Sensitivity Analysis for MBTI Classification

Performs systematic causal intervention analysis:
1. Trains Joint Model (Cognitive + Semantic) once.
2. Iterates over ALL 10 cognitive features.
3. Tests multiple intervention magnitudes λ ∈ {0.3, 0.5, 1.0}.
4. Computes APS (Average Probability Shift) and TFR (Trait Flip Rate).
5. Generates Sensitivity Ranking Matrix.

Author: Dhanshri
Date: 2026-02-15
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from tqdm import tqdm

# Ensure project imports work
sys.path.insert(0, os.path.abspath('.'))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Import phase3 logical components for metrics (keeping shifts/flips logic)
from phase3_causal_reasoning.causal_analysis.probability_shift import probability_shift
from phase3_causal_reasoning.counterfactual.decision_logic import trait_flip

# Import experiments utilities
from training.experiments import (
    labels_to_targets,
    DIMENSIONS,
    RANDOM_STATE
)

# Feature definitions aligned with generation script
ALL_FEATURES = [
    "pronoun_ratio",      # 0
    "modality_score",     # 1
    "emotion_intensity",  # 2
    "lexical_diversity",  # 3
    "readability_score",  # 4
    "negation_count",     # 5
    "sentence_length",    # 6
    "reasoning",          # 7
    "planning",           # 8
    "uncertainty",        # 9
]

INTERVENTION_LAMBDAS = [0.3, 0.5, 1.0]

def do_intervention_unbounded(x: Any, x_ref: float, lam: float) -> Any:
    """
    Apply relative do() intervention without safety bounds.
    Supports scalar or vectorized input.
    x_cf = x + lam * (x_ref - x)
    """
    return x + lam * (x_ref - x)

def predict_probabilities(clf: LogisticRegression, X: np.ndarray) -> float:
    """Get probability of class 1."""
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return clf.predict_proba(X)[0, 1]

def compute_metrics(prob_shifts: List[Dict[str, float]], 
                   flip_flags: List[Dict[str, bool]]) -> Dict[str, Dict[str, float]]:
    """Compute APS, TFR, SR per dimension."""
    metrics = {}
    for dim in DIMENSIONS:
        shifts = [abs(s[dim]) for s in prob_shifts]
        flips = [f[dim] for f in flip_flags]
        
        aps = np.mean(shifts)
        tfr = (sum(flips) / len(flips)) * 100.0
        sr = 1.0 - aps
        
        metrics[dim] = {
            'APS': float(aps),
            'TFR': float(tfr),
            'SR': float(sr)
        }
    return metrics

def run_comprehensive_analysis(npz_path: str, output_dir: str):
    print("\n" + "="*90)
    print("COMPREHENSIVE STABILITY & SENSITIVITY ANALYSIS")
    print("="*90)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load and Prepare Data
    print("Loading data...")
    data = np.load(npz_path, allow_pickle=True)
    C = data['C']
    S = data['S']
    labels = list(data['labels'])
    y = labels_to_targets(labels)
    
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )
    
    # 2. Scale Features (Step 1a Strategy: Separate Scaling)
    print("Scaling features...")
    scaler_C = StandardScaler()
    scaler_S = StandardScaler()
    
    C_train = scaler_C.fit_transform(C[train_idx])
    C_test = scaler_C.transform(C[test_idx])
    
    S_train = scaler_S.fit_transform(S[train_idx])
    S_test = scaler_S.transform(S[test_idx])
    
    J_train = np.concatenate([C_train, S_train], axis=1)
    J_test = np.concatenate([C_test, S_test], axis=1)
    y_train = y[train_idx]
    
    # Raw C for intervention logic
    C_test_raw = C[test_idx]
    
    print(f"Test Set: {len(test_idx)} samples")
    
    # 3. Train Models Once
    print("\nTraining joint models...")
    classifiers = {}
    for dim_idx, dim_name in enumerate(DIMENSIONS):
        clf = LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000, 
            class_weight='balanced', solver='lbfgs'
        )
        clf.fit(J_train, y_train[:, dim_idx])
        classifiers[dim_name] = clf
        print(f"  ✓ {dim_name} model trained")

    # 4. Compute Reference Values (Mean of test set)
    ref_values = np.mean(C_test_raw, axis=0)
    
    # 5. Iterative Analysis
    full_results = {}  # {feature: {lambda: {metrics}}}
    
    print("\nStarting sensitivity sweep...")
    total_steps = len(ALL_FEATURES) * len(INTERVENTION_LAMBDAS)
    pbar = tqdm(total=total_steps, desc="Analyzing Interventions")
    
    for feat_name in ALL_FEATURES:
        feat_idx = ALL_FEATURES.index(feat_name)
        feat_ref = ref_values[feat_idx]
        full_results[feat_name] = {}
        
        for lam in INTERVENTION_LAMBDAS:
            # Run intervention on whole test set
            prob_shifts_all = []
            flip_flags_all = []
            
            # Efficient vectorized intervention? 
            # Logic: x_new = x + lam*(ref - x) = (1-lam)*x + lam*ref
            # This is a linear transform. We can do it on the whole batch C_test_raw column.
            
            # Vectorized intervention on raw feature
            col_raw = C_test_raw[:, feat_idx]
            col_intervened = do_intervention_unbounded(col_raw, feat_ref, lam) # Broadcasting works? No, do_intervention is scalar
            # Vectorized logic:
            col_intervened = col_raw + lam * (feat_ref - col_raw)
            
            # Create full C matrix with intervened column
            C_test_raw_cf = C_test_raw.copy()
            C_test_raw_cf[:, feat_idx] = col_intervened
            
            # Scale
            C_test_cf = scaler_C.transform(C_test_raw_cf)
            
            # Concatenate with S (S does not change)
            J_test_cf = np.concatenate([C_test_cf, S_test], axis=1)
            
            # Predict batch
            shifts_batch = {}
            flips_batch = {}
            
            for dim_name in DIMENSIONS:
                clf = classifiers[dim_name]
                # Factual predictions (can be cached, but fast enough to recompute)
                p_factual = clf.predict_proba(J_test)[:, 1]
                p_cf = clf.predict_proba(J_test_cf)[:, 1]
                
                # Compute shifts
                diffs = np.abs(p_factual - p_cf)
                aps = np.mean(diffs)
                
                # Compute flips
                # Flip defined as: (p < 0.5 and p_cf >= 0.5) or (p >= 0.5 and p_cf < 0.5)
                # i.e. sign(p-0.5) != sign(p_cf-0.5)
                flipped = (p_factual >= 0.5) != (p_cf >= 0.5)
                tfr = np.mean(flipped) * 100.0
                sr = 1.0 - aps
                
                shifts_batch[dim_name] = {'APS': float(aps), 'TFR': float(tfr), 'SR': float(sr)}
            
            full_results[feat_name][lam] = shifts_batch
            pbar.update(1)
            
    pbar.close()
    
    # 6. Save Results
    json_path = os.path.join(output_dir, 'comprehensive_stability.json')
    with open(json_path, 'w') as f:
        # Key must be string
        # Convert lam keys to strings
        serializable = {f: {str(l): m for l, m in v.items()} for f, v in full_results.items()}
        json.dump(serializable, f, indent=2)
    print(f"\n✓ Saved comprehensive results to {json_path}")
    
    # 7. Generate Summary Tables
    generate_sensitivity_matrix(full_results, output_dir)

def generate_sensitivity_matrix(results: Dict, output_dir: str):
    """Generate Rankings and Heatmap Data."""
    
    # Flatten data for DataFrame
    rows = []
    for feat, lams in results.items():
        # Focus on max intervention (lam=1.0) for ranking sensitivity
        if 1.0 in lams:
            data_10 = lams[1.0] # or lams['1.0'] if loaded from json
        else:
            data_10 = lams[max(lams.keys())]
            
        row = {'Feature': feat}
        avg_aps = 0
        for dim in DIMENSIONS:
            aps = data_10[dim]['APS']
            row[f'APS_{dim}'] = aps
            avg_aps += aps
        row['Mean_APS'] = avg_aps / 4
        rows.append(row)
        
    df = pd.DataFrame(rows)
    df = df.sort_values('Mean_APS', ascending=False)
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'sensitivity_ranking.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved sensitivity ranking to {csv_path}")
    
    # Generate LaTeX
    tex_path = os.path.join(output_dir, 'sensitivity_table.tex')
    with open(tex_path, 'w') as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Cognitive Feature Sensitivity Ranking ($\\lambda=1.0$)}\n")
        f.write("\\label{tab:sensitivity}\n")
        f.write("\\begin{tabular}{lccccc}\n\\hline\n")
        f.write("Feature & IE & NS & TF & JP & \\textbf{Mean APS} \\\\\n\\hline\n")
        
        for _, r in df.iterrows():
            f.write(f"{r['Feature'].replace('_', ' ')} & "
                    f"{r['APS_IE']:.4f} & {r['APS_NS']:.4f} & "
                    f"{r['APS_TF']:.4f} & {r['APS_JP']:.4f} & "
                    f"\\textbf{{{r['Mean_APS']:.4f}}} \\\\\n")
        
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"✓ Saved ranking table to {tex_path}")

if __name__ == '__main__':
    npz_path = os.path.join('data', 'processed', 'phase2_features.npz')
    output_dir = os.path.join('results', 'stability_comprehensive')
    
    if os.path.exists(npz_path):
        run_comprehensive_analysis(npz_path, output_dir)
    else:
        print("Data file not found!")
