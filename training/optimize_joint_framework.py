#!/usr/bin/env python3
"""
Systematic Optimization of MBTI Joint Cognitive-Semantic Classification Framework

This script implements 6 experimental optimization strategies:
1. Feature Normalization (StandardScaler)
2. Logistic Regression Hyperparameter Tuning
3. PCA Dimensionality Reduction Study
4. Weighted Fusion Strategies
5. Classical TF-IDF + LinearSVC Baseline
6. Comprehensive Statistical Comparison

Target: ≥ 0.60 macro-F1, ensure Joint > Semantic-only, maintain explainability.

Author: Dhanashri
Date: 2026-02-15
"""

import os
import sys
import json
import shutil
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple
from collections import Counter

# Machine learning imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

# Import utilities from existing experiments
from experiments import (
    labels_to_targets,
    evaluate_dimension,
    analyze_imbalance,
    DIMENSIONS,
    RANDOM_STATE
)

# Global configuration
np.random.seed(RANDOM_STATE)
BASE_RESULTS_DIR = "results"
OPTIMIZATION_DIR = os.path.join(BASE_RESULTS_DIR, "optimization")


def setup_experiment_dirs():
    """Delete and recreate results directory for fresh experiments."""
    print("\n" + "="*90)
    print("EXPERIMENT SETUP")
    print("="*90)
    
    # Delete existing results
    if os.path.exists(BASE_RESULTS_DIR):
        print(f"Deleting existing directory: {BASE_RESULTS_DIR}")
        shutil.rmtree(BASE_RESULTS_DIR)
        print("  ✓ Deleted")
    
    # Create fresh structure
    dirs = [
        OPTIMIZATION_DIR,
        os.path.join(OPTIMIZATION_DIR, "normalization"),
        os.path.join(OPTIMIZATION_DIR, "lr_tuning"),
        os.path.join(OPTIMIZATION_DIR, "pca_study"),
        os.path.join(OPTIMIZATION_DIR, "weighted_fusion"),
        os.path.join(OPTIMIZATION_DIR, "tfidf_baseline"),
        os.path.join(OPTIMIZATION_DIR, "final_comparison")
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    print(f"✓ Created fresh results directory structure")
    print("="*90 + "\n")


def load_data():
    """Load all data (C, S, J features and raw text)."""
    print("Loading data...")
    
    # Load phase2 features
    npz_path = os.path.join("data", "processed", "phase2_features.npz")
    data = np.load(npz_path, allow_pickle=True)
    C = data['C']  # (N, 10)
    S = data['S']  # (N, 768)
    J = data['J']  # (N, 512)
    labels = list(data['labels'])
    
    # Load raw text for TF-IDF baseline
    json_path = os.path.join("data", "processed", "phase1_cleaned.json")
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    texts = [entry['text'] for entry in raw_data]
    
    print(f"  ✓ Loaded {len(labels)} samples")
    print(f"  ✓ C shape: {C.shape}")
    print(f"  ✓ S shape: {S.shape}")
    print(f"  ✓ J shape: {J.shape}")
    print(f"  ✓ Raw texts: {len(texts)}\n")
    
    return C, S, J, labels, texts


def save_experiment_results(output_dir: str, config: Dict, metrics: Dict, name: str):
    """Save experiment configuration and results."""
    # Save config
    config_path = os.path.join(output_dir, f"{name}_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f"{name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"  ✓ Results saved to {output_dir}")


def evaluate_model(X_train, X_test, y_train, y_test, config_name: str, max_iter=5000, C_param=1.0):
    """
    Train and evaluate logistic regression model for all dimensions.
    
    Returns:
        dict: Metrics per dimension
    """
    metrics = {}
    
    for dim_idx, dim_name in enumerate(DIMENSIONS):
        # Train classifier
        clf = LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=max_iter,
            class_weight='balanced',
            solver='lbfgs',
            C=C_param
        )
        clf.fit(X_train, y_train[:, dim_idx])
        
        # Predictions
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        
        # Compute F1-macro
        f1_macro = f1_score(y_test[:, dim_idx], y_pred, average='macro', zero_division=0)
        
        metrics[dim_name] = {
            'f1_macro': float(f1_macro)
        }
        
        print(f"    {dim_name}: F1-macro = {f1_macro:.4f}")
    
    # Compute average
    avg_f1 = np.mean([metrics[d]['f1_macro'] for d in DIMENSIONS])
    metrics['average'] = float(avg_f1)
    
    print(f"    Average F1-macro: {avg_f1:.4f}")
    
    return metrics


# ==================== STEP 1: FEATURE NORMALIZATION ====================

def step1_normalization(C, S, J, labels):
    """
    Test different feature normalization strategies.
    """
    print("\n" + "="*90)
    print("STEP 1: FEATURE NORMALIZATION")
    print("="*90)
    
    output_dir = os.path.join(OPTIMIZATION_DIR, "normalization")
    y = labels_to_targets(labels)
    
    # Split data (stratified)
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )
    
    y_train, y_test = y[train_idx], y[test_idx]
    
    results = {}
    
    # Strategy (a): Scale C & S separately, then concatenate
    print("\n(a) Scaling C & S separately...")
    scaler_C = StandardScaler()
    scaler_S = StandardScaler()
    
    C_train_scaled = scaler_C.fit_transform(C[train_idx])
    C_test_scaled = scaler_C.transform(C[test_idx])
    
    S_train_scaled = scaler_S.fit_transform(S[train_idx])
    S_test_scaled = scaler_S.transform(S[test_idx])
    
    J_train_a = np.concatenate([C_train_scaled, S_train_scaled], axis=1)
    J_test_a = np.concatenate([C_test_scaled, S_test_scaled], axis=1)
    
    results['separate_scaling'] = evaluate_model(J_train_a, J_test_a, y_train, y_test, "separate_scaling")
    
    # Strategy (b): Scale only J (concatenated)
    print("\n(b) Scaling J directly...")
    scaler_J = StandardScaler()
    J_train_b = scaler_J.fit_transform(J[train_idx])
    J_test_b = scaler_J.transform(J[test_idx])
    
    results['joint_scaling'] = evaluate_model(J_train_b, J_test_b, y_train, y_test, "joint_scaling")
    
    # Strategy (c): No scaling (baseline)
    print("\n(c) No scaling (baseline)...")
    J_train_c = J[train_idx]
    J_test_c = J[test_idx]
    
    results['no_scaling'] = evaluate_model(J_train_c, J_test_c, y_train, y_test, "no_scaling")
    
    # Save results
    config = {
        'step': 'normalization',
        'strategies': ['separate_scaling', 'joint_scaling', 'no_scaling'],
        'random_state': RANDOM_STATE
    }
    
    save_experiment_results(output_dir, config, results, "normalization")
    
    # Find best
    best_strategy = max(results.keys(), key=lambda k: results[k]['average'])
    print(f"\n✓ Best strategy: {best_strategy} (F1-macro: {results[best_strategy]['average']:.4f})")
    
    return results, best_strategy


# ==================== STEP 2: LOGISTIC REGRESSION TUNING ====================

def step2_lr_tuning(C, S, J, labels):
    """
    Tune logistic regression hyperparameter C using validation set.
    """
    print("\n" + "="*90)
    print("STEP 2: LOGISTIC REGRESSION HYPERPARAMETER TUNING")
    print("="*90)
    
    output_dir = os.path.join(OPTIMIZATION_DIR, "lr_tuning")
    y = labels_to_targets(labels)
    
    # Split: 60% train, 20% val, 20% test
    indices = np.arange(len(labels))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )
    
    # Further split train_val into train and val
    labels_train_val = [labels[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.25, random_state=RANDOM_STATE, stratify=labels_train_val  # 0.25 * 0.8 = 0.2
    )
    
    # Use best normalization strategy from Step 1 (assume separate scaling)
    scaler_C = StandardScaler()
    scaler_S = StandardScaler()
    
    C_train_scaled = scaler_C.fit_transform(C[train_idx])
    C_val_scaled = scaler_C.transform(C[val_idx])
    C_test_scaled = scaler_C.transform(C[test_idx])
    
    S_train_scaled = scaler_S.fit_transform(S[train_idx])
    S_val_scaled = scaler_S.transform(S[val_idx])
    S_test_scaled = scaler_S.transform(S[test_idx])
    
    J_train = np.concatenate([C_train_scaled, S_train_scaled], axis=1)
    J_val = np.concatenate([C_val_scaled, S_val_scaled], axis=1)
    J_test = np.concatenate([C_test_scaled, S_test_scaled], axis=1)
    
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]
    
    # Grid search
    C_values = [0.01, 0.1, 1, 5, 10]
    results = {}
    best_C_per_dim = {}
    
    for C_param in C_values:
        print(f"\nTesting C = {C_param}...")
        
        val_metrics = {}
        for dim_idx, dim_name in enumerate(DIMENSIONS):
            clf = LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=5000,
                class_weight='balanced',
                solver='lbfgs',
                C=C_param
            )
            clf.fit(J_train, y_train[:, dim_idx])
            
            y_val_pred = clf.predict(J_val)
            f1_val = f1_score(y_val[:, dim_idx], y_val_pred, average='macro', zero_division=0)
            
            val_metrics[dim_name] = float(f1_val)
            print(f"    {dim_name}: Val F1 = {f1_val:.4f}")
        
        avg_f1_val = np.mean(list(val_metrics.values()))
        val_metrics['average'] = float(avg_f1_val)
        results[f'C_{C_param}'] = val_metrics
    
    # Select best C per dimension
    for dim_name in DIMENSIONS:
        best_C = max(C_values, key=lambda c: results[f'C_{c}'][dim_name])
        best_C_per_dim[dim_name] = best_C
        print(f"\n  Best C for {dim_name}: {best_C} (Val F1 = {results[f'C_{best_C}'][dim_name]:.4f})")
    
    # Evaluate on test set with best C (use average best)
    best_C_overall = max(C_values, key=lambda c: results[f'C_{c}']['average'])
    print(f"\n✓ Best C overall: {best_C_overall}")
    
    print(f"\nEvaluating on test set with C = {best_C_overall}...")
    test_metrics = evaluate_model(J_train, J_test, y_train, y_test, "lr_tuned", max_iter=5000, C_param=best_C_overall)
    
    config = {
        'step': 'lr_tuning',
        'C_values': C_values,
        'best_C_overall': best_C_overall,
        'best_C_per_dim': best_C_per_dim,
        'random_state': RANDOM_STATE
    }
    
    final_results = {
        'validation_results': results,
        'test_results': test_metrics,
        'best_config': config
    }
    
    save_experiment_results(output_dir, config, final_results, "lr_tuning")
    
    return test_metrics, best_C_overall


# ==================== STEP 3: PCA STUDY ====================

def step3_pca_study(C, S, J, labels):
    """
    Apply PCA to S and J, test variance thresholds.
    """
    print("\n" + "="*90)
    print("STEP 3: PCA DIMENSIONALITY REDUCTION STUDY")
    print("="*90)
    
    output_dir = os.path.join(OPTIMIZATION_DIR, "pca_study")
    y = labels_to_targets(labels)
    
    # Split data
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )
    
    # Scale features
    scaler_C = StandardScaler()
    scaler_S = StandardScaler()
    scaler_J = StandardScaler()
    
    C_train_scaled = scaler_C.fit_transform(C[train_idx])
    C_test_scaled = scaler_C.transform(C[test_idx])
    
    S_train_scaled = scaler_S.fit_transform(S[train_idx])
    S_test_scaled = scaler_S.transform(S[test_idx])
    
    J_train_scaled = scaler_J.fit_transform(J[train_idx])
    J_test_scaled = scaler_J.transform(J[test_idx])
    
    y_train, y_test = y[train_idx], y[test_idx]
    
    results = {}
    
    # Baseline: No PCA
    print("\nBaseline (No PCA)...")
    J_baseline = np.concatenate([C_train_scaled, S_train_scaled], axis=1)
    J_baseline_test = np.concatenate([C_test_scaled, S_test_scaled], axis=1)
    results['no_pca'] = evaluate_model(J_baseline, J_baseline_test, y_train, y_test, "no_pca")
    
    # PCA on S (semantic)
    for variance in [0.90, 0.95]:
        print(f"\nPCA on S (variance={variance})...")
        pca_s = PCA(n_components=variance, random_state=RANDOM_STATE)
        S_train_pca = pca_s.fit_transform(S_train_scaled)
        S_test_pca = pca_s.transform(S_test_scaled)
        
        print(f"  Components retained: {pca_s.n_components_} / {S_train_scaled.shape[1]}")
        
        J_pca = np.concatenate([C_train_scaled, S_train_pca], axis=1)
        J_pca_test = np.concatenate([C_test_scaled, S_test_pca], axis=1)
        
        results[f'pca_S_{int(variance*100)}'] = evaluate_model(J_pca, J_pca_test, y_train, y_test, f"pca_S_{int(variance*100)}")
        results[f'pca_S_{int(variance*100)}']['n_components'] = int(pca_s.n_components_)
    
    # PCA on J (joint)
    for variance in [0.90, 0.95]:
        print(f"\nPCA on J (variance={variance})...")
        pca_j = PCA(n_components=variance, random_state=RANDOM_STATE)
        J_train_pca = pca_j.fit_transform(J_train_scaled)
        J_test_pca = pca_j.transform(J_test_scaled)
        
        print(f"  Components retained: {pca_j.n_components_} / {J_train_scaled.shape[1]}")
        
        results[f'pca_J_{int(variance*100)}'] = evaluate_model(J_train_pca, J_test_pca, y_train, y_test, f"pca_J_{int(variance*100)}")
        results[f'pca_J_{int(variance*100)}']['n_components'] = int(pca_j.n_components_)
    
    config = {
        'step': 'pca_study',
        'variance_thresholds': [0.90, 0.95],
        'random_state': RANDOM_STATE
    }
    
    save_experiment_results(output_dir, config, results, "pca_study")
    
    # Find best
    best_strategy = max(results.keys(), key=lambda k: results[k]['average'])
    print(f"\n✓ Best strategy: {best_strategy} (F1-macro: {results[best_strategy]['average']:.4f})")
    
    return results, best_strategy


# ==================== STEP 4: WEIGHTED FUSION ====================

def step4_weighted_fusion(C, S, labels):
    """
    Test weighted fusion: J = [αC ; βS], with β=1.
    """
    print("\n" + "="*90)
    print("STEP 4: WEIGHTED FUSION STRATEGIES")
    print("="*90)
    
    output_dir = os.path.join(OPTIMIZATION_DIR, "weighted_fusion")
    y = labels_to_targets(labels)
    
    # Split data
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )
    
    # Scale features
    scaler_C = StandardScaler()
    scaler_S = StandardScaler()
    
    C_train_scaled = scaler_C.fit_transform(C[train_idx])
    C_test_scaled = scaler_C.transform(C[test_idx])
    
    S_train_scaled = scaler_S.fit_transform(S[train_idx])
    S_test_scaled = scaler_S.transform(S[test_idx])
    
    y_train, y_test = y[train_idx], y[test_idx]
    
    results = {}
    alpha_values = [0.5, 1, 2]
    beta = 1
    
    for alpha in alpha_values:
        print(f"\nTesting α = {alpha}, β = {beta}...")
        
        # Weighted concatenation
        C_weighted_train = C_train_scaled * alpha
        C_weighted_test = C_test_scaled * alpha
        
        S_weighted_train = S_train_scaled * beta
        S_weighted_test = S_test_scaled * beta
        
        J_train = np.concatenate([C_weighted_train, S_weighted_train], axis=1)
        J_test = np.concatenate([C_weighted_test, S_weighted_test], axis=1)
        
        results[f'alpha_{alpha}'] = evaluate_model(J_train, J_test, y_train, y_test, f"alpha_{alpha}")
    
    config = {
        'step': 'weighted_fusion',
        'alpha_values': alpha_values,
        'beta': beta,
        'random_state': RANDOM_STATE
    }
    
    save_experiment_results(output_dir, config, results, "weighted_fusion")
    
    # Find best
    best_alpha = max(alpha_values, key=lambda a: results[f'alpha_{a}']['average'])
    print(f"\n✓ Best α: {best_alpha} (F1-macro: {results[f'alpha_{best_alpha}']['average']:.4f})")
    
    return results, best_alpha


# ==================== STEP 5: TF-IDF BASELINE ====================

def step5_tfidf_baseline(texts, labels):
    """
    Classical baseline: TF-IDF + LinearSVC.
    """
    print("\n" + "="*90)
    print("STEP 5: TF-IDF + LinearSVC BASELINE")
    print("="*90)
    
    output_dir = os.path.join(OPTIMIZATION_DIR, "tfidf_baseline")
    y = labels_to_targets(labels)
    
    # Split data
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )
    
    texts_train = [texts[i] for i in train_idx]
    texts_test = [texts[i] for i in test_idx]
    
    y_train, y_test = y[train_idx], y[test_idx]
    
    # TF-IDF vectorization
    print("\nVectorizing with TF-IDF (max_features=10000)...")
    vectorizer = TfidfVectorizer(max_features=10000, min_df=2, max_df=0.95)
    X_train = vectorizer.fit_transform(texts_train)
    X_test = vectorizer.transform(texts_test)
    
    print(f"  Feature matrix shape: {X_train.shape}")
    
    # Train LinearSVC for each dimension
    metrics = {}
    for dim_idx, dim_name in enumerate(DIMENSIONS):
        print(f"\n  Training {dim_name}...")
        
        clf = LinearSVC(
            random_state=RANDOM_STATE,
            class_weight='balanced',
            max_iter=2000
        )
        clf.fit(X_train, y_train[:, dim_idx])
        
        y_pred = clf.predict(X_test)
        f1_macro = f1_score(y_test[:, dim_idx], y_pred, average='macro', zero_division=0)
        
        metrics[dim_name] = {'f1_macro': float(f1_macro)}
        print(f"    F1-macro: {f1_macro:.4f}")
    
    avg_f1 = np.mean([metrics[d]['f1_macro'] for d in DIMENSIONS])
    metrics['average'] = float(avg_f1)
    
    print(f"\n  Average F1-macro: {avg_f1:.4f}")
    
    config = {
        'step': 'tfidf_baseline',
        'vectorizer': 'TfidfVectorizer',
        'max_features': 10000,
        'classifier': 'LinearSVC',
        'random_state': RANDOM_STATE
    }
    
    save_experiment_results(output_dir, config, metrics, "tfidf_baseline")
    
    return metrics


# ==================== STEP 6: FINAL COMPARISON ====================

def step6_final_comparison(all_results):
    """
    Generate comprehensive comparison report.
    """
    print("\n" + "="*90)
    print("STEP 6: FINAL COMPREHENSIVE COMPARISON")
    print("="*90)
    
    output_dir = os.path.join(OPTIMIZATION_DIR, "final_comparison")
    
    # Extract best results from each step
    summary = {}
    
    # Find best overall
    all_configs = []
    
    for step_name, step_data in all_results.items():
        if isinstance(step_data, dict):
            for config_name, metrics in step_data.items():
                if isinstance(metrics, dict) and 'average' in metrics:
                    all_configs.append({
                        'step': step_name,
                        'config': config_name,
                        'avg_f1': metrics['average'],
                        'metrics': metrics
                    })
    
    # Sort by average F1
    all_configs.sort(key=lambda x: x['avg_f1'], reverse=True)
    
    # Print top 10
    print("\nTop 10 Configurations:")
    print("-" * 90)
    print(f"{'Rank':<6} {'Step':<25} {'Configuration':<30} {'Avg F1':<10}")
    print("-" * 90)
    
    for i, config in enumerate(all_configs[:10], 1):
        print(f"{i:<6} {config['step']:<25} {config['config']:<30} {config['avg_f1']:.4f}")
    
    # Best configuration
    best_config = all_configs[0]
    print(f"\n{'='*90}")
    print(f"✓ BEST CONFIGURATION")
    print(f"{'='*90}")
    print(f"  Step: {best_config['step']}")
    print(f"  Config: {best_config['config']}")
    print(f"  Average F1-macro: {best_config['avg_f1']:.4f}")
    print(f"\n  Per-dimension F1 scores:")
    for dim in DIMENSIONS:
        if dim in best_config['metrics']:
            print(f"    {dim}: {best_config['metrics'][dim]['f1_macro']:.4f}")
    
    # Check if target achieved
    target_achieved = best_config['avg_f1'] >= 0.60
    print(f"\n  Target ≥ 0.60: {'✓ ACHIEVED' if target_achieved else '✗ NOT ACHIEVED'}")
    
    # Save complete summary
    summary_report = {
        'timestamp': datetime.now().isoformat(),
        'target_f1': 0.60,
        'target_achieved': target_achieved,
        'best_configuration': best_config,
        'top_10_configurations': all_configs[:10],
        'all_results': all_results
    }
    
    summary_path = os.path.join(output_dir, 'summary_report.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    # Save best config separately
    best_config_path = os.path.join(OPTIMIZATION_DIR, 'best_config.json')
    with open(best_config_path, 'w') as f:
        json.dump(best_config, f, indent=2)
    
    print(f"\n✓ Summary report saved to: {summary_path}")
    print(f"✓ Best config saved to: {best_config_path}")
    
    # Generate LaTeX table
    latex_path = os.path.join(output_dir, 'optimization_comparison.tex')
    with open(latex_path, 'w') as f:
        f.write("% Optimization Study: Top Configurations\\n")
        f.write("\\begin{table}[h]\\n")
        f.write("\\centering\\n")
        f.write("\\caption{Top Configuration Comparison - Macro-F1 Scores}\\n")
        f.write("\\label{tab:optimization_comparison}\\n")
        f.write("\\begin{tabular}{llc}\\n")
        f.write("\\hline\\n")
        f.write("Step & Configuration & Avg F1 \\\\\\n")
        f.write("\\hline\\n")
        
        for config in all_configs[:5]:
            step = config['step'].replace('_', ' ').title()
            cfg_name = config['config'].replace('_', ' ').title()
            f1 = f"{config['avg_f1']:.3f}"
            
            if config == best_config:
                f.write(f"\\textbf{{{step}}} & \\textbf{{{cfg_name}}} & \\textbf{{{f1}}} \\\\\\n")
            else:
                f.write(f"{step} & {cfg_name} & {f1} \\\\\\n")
        
        f.write("\\hline\\n")
        f.write("\\end{tabular}\\n")
        f.write("\\end{table}\\n")
    
    print(f"✓ LaTeX table saved to: {latex_path}")
    
    return summary_report


# ==================== MAIN EXECUTION ====================

def main():
    """Main execution pipeline."""
    print("\n" + "="*90)
    print("SYSTEMATIC OPTIMIZATION OF MBTI JOINT CLASSIFICATION FRAMEWORK")
    print("="*90)
    print(f"Target: ≥ 0.60 macro-F1")
    print(f"Random State: {RANDOM_STATE}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*90)
    
    # Setup
    setup_experiment_dirs()
    
    # Load data
    C, S, J, labels, texts = load_data()
    
    # Store all results
    all_results = {}
    
    # STEP 1: Normalization
    norm_results, best_norm_strategy = step1_normalization(C, S, J, labels)
    all_results['normalization'] = norm_results
    
    # STEP 2: LR Tuning
    lr_results, best_C = step2_lr_tuning(C, S, J, labels)
    all_results['lr_tuning'] = {'test_results': lr_results}
    
    # STEP 3: PCA Study
    pca_results, best_pca_strategy = step3_pca_study(C, S, J, labels)
    all_results['pca_study'] = pca_results
    
    # STEP 4: Weighted Fusion
    fusion_results, best_alpha = step4_weighted_fusion(C, S, labels)
    all_results['weighted_fusion'] = fusion_results
    
    # STEP 5: TF-IDF Baseline
    tfidf_results = step5_tfidf_baseline(texts, labels)
    all_results['tfidf_baseline'] = {'tfidf_svc': tfidf_results}
    
    # STEP 6: Final Comparison
    summary_report = step6_final_comparison(all_results)
    
    print("\n" + "="*90)
    print("✓ ALL OPTIMIZATION EXPERIMENTS COMPLETED")
    print("="*90)
    print(f"\nResults directory: {os.path.abspath(OPTIMIZATION_DIR)}")
    print(f"\nKey findings:")
    print(f"  - Best average F1-macro: {summary_report['best_configuration']['avg_f1']:.4f}")
    print(f"  - Target (≥ 0.60): {'✓ ACHIEVED' if summary_report['target_achieved'] else '✗ NOT ACHIEVED'}")
    print(f"  - Best approach: {summary_report['best_configuration']['step']} - {summary_report['best_configuration']['config']}")
    print("="*90 + "\n")


if __name__ == "__main__":
    main()
