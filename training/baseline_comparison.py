#!/usr/bin/env python3
"""
Baseline Comparison for MBTI Classification

Compares three feature representations:
1. Cognitive-only (C) - 10 dimensions
2. Semantic-only (S) - 768 dimensions (BERT)
3. Joint fused (J) - 512 dimensions (Gated fusion of C + S)

All baselines use identical:
- Train/test split (stratified, random_state=42)
- Classifier (LogisticRegression with balanced weights)
- Evaluation metrics

Author: Dhanshri
Date: 2026-02-14
"""

import os
import sys
import json
import numpy as np
from typing import Dict, Any, List
from datetime import datetime

# Import from experiments.py
from .experiments import (
    load_phase2,
    labels_to_targets,
    analyze_imbalance,
    evaluate_dimension,
    save_metrics,
    print_summary_table,
    generate_combined_figures,
    DIMENSIONS,
    RANDOM_STATE
)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def evaluate_baseline(features: np.ndarray,
                      labels: List[str],
                      baseline_name: str,
                      output_dir: str) -> Dict[str, Any]:
    """
    Evaluate a single baseline with complete metrics.
    
    Args:
        features: Feature matrix (N, D)
        labels: MBTI labels (N,)
        baseline_name: 'cognitive_only', 'semantic_only', or 'joint_fused'
        output_dir: Output directory for results
    
    Returns:
        dict: All metrics and results
    """
    print(f"\n{'='*70}")
    print(f"BASELINE: {baseline_name.upper().replace('_', ' ')}")
    print(f"{'='*70}")
    print(f"Feature dimensionality: {features.shape[1]}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert labels to binary targets
    y = labels_to_targets(labels)
    
    # Stratified train/test split
    print(f"Performing stratified train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,  # Stratify on 16 MBTI classes
        shuffle=True
    )
    
    print(f"  Train samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    
    # Configuration
    config = {
        'baseline_name': baseline_name,
        'feature_dim': features.shape[1],
        'n_samples': features.shape[0],
        'n_train': X_train.shape[0],
        'n_test': X_test.shape[0],
        'random_state': RANDOM_STATE,
        'test_size': 0.2,
        'stratified': True,
        'classifier': 'LogisticRegression',
        'max_iter': 1000,
        'class_weight': 'balanced'
    }
    
    # Train and evaluate each dimension
    all_metrics = {}
    
    print(f"\nTraining and evaluating {len(DIMENSIONS)} dimensions...\n")
    
    for dim_idx, dim_name in enumerate(DIMENSIONS):
        print(f"[{dim_idx+1}/{len(DIMENSIONS)}] Evaluating dimension: {dim_name}")
        
        # Train balanced classifier
        clf = LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000,
            class_weight='balanced',
            solver='lbfgs'
        )
        clf.fit(X_train, y_train[:, dim_idx])
        
        # Predictions
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        
        # Comprehensive evaluation
        metrics = evaluate_dimension(
            y_test[:, dim_idx],
            y_pred,
            y_proba,
            dim_name,
            output_dir
        )
        
        all_metrics[dim_name] = metrics
        
        print(f"  [OK] F1-macro: {metrics['f1_macro']:.4f}, AUC: {metrics['auc']:.4f}")
    
    # Save metrics
    imbalance_stats = analyze_imbalance(labels)
    metrics_path = os.path.join(output_dir, 'metrics.json')
    save_metrics(all_metrics, config, imbalance_stats, metrics_path)
    
    # Print summary
    print_summary_table(all_metrics)
    
    return all_metrics


def generate_comparison_table(results: Dict[str, Dict[str, Dict[str, Any]]],
                              output_dir: str) -> Dict[str, Any]:
    """
    Generate baseline comparison table and LaTeX output.
    
    Args:
        results: Nested dict of {baseline_name: {dimension: metrics}}
        output_dir: Output directory
    
    Returns:
        dict: Comparison data
    """
    baselines = ['cognitive_only', 'semantic_only', 'joint_fused']
    
    # Collect F1 scores
    comparison = {}
    for baseline in baselines:
        comparison[baseline] = {}
        for dim in DIMENSIONS:
            comparison[baseline][dim] = results[baseline][dim]['f1_macro']
        # Average across dimensions
        comparison[baseline]['avg'] = np.mean([comparison[baseline][d] for d in DIMENSIONS])
    
    # Also collect other metrics for comprehensive comparison
    full_comparison = {}
    for baseline in baselines:
        full_comparison[baseline] = {
            'f1_macro': {dim: results[baseline][dim]['f1_macro'] for dim in DIMENSIONS},
            'precision_macro': {dim: results[baseline][dim]['precision_macro'] for dim in DIMENSIONS},
            'recall_macro': {dim: results[baseline][dim]['recall_macro'] for dim in DIMENSIONS},
            'auc': {dim: results[baseline][dim]['auc'] for dim in DIMENSIONS},
            'averages': {
                'f1': comparison[baseline]['avg'],
                'precision': np.mean([results[baseline][d]['precision_macro'] for d in DIMENSIONS]),
                'recall': np.mean([results[baseline][d]['recall_macro'] for d in DIMENSIONS]),
                'auc': np.mean([results[baseline][d]['auc'] for d in DIMENSIONS])
            }
        }
    
    # Save as JSON
    comparison_path = os.path.join(output_dir, 'baseline_comparison.json')
    with open(comparison_path, 'w') as f:
        json.dump(full_comparison, f, indent=2)
    
    print(f"\n[OK] Comparison saved to: {comparison_path}")
    
    # Print console table
    print("\n" + "="*90)
    print("BASELINE COMPARISON - MACRO-F1 SCORES")
    print("="*90)
    print(f"{'Baseline':<20} {'IE':<10} {'NS':<10} {'TF':<10} {'JP':<10} {'Average':<10}")
    print("-"*90)
    
    for baseline in baselines:
        name_formatted = baseline.replace('_', ' ').title()
        row = [name_formatted]
        for dim in DIMENSIONS:
            row.append(f"{comparison[baseline][dim]:.4f}")
        row.append(f"{comparison[baseline]['avg']:.4f}")
        
        # Format as string
        formatted_row = f"{row[0]:<20} {row[1]:<10} {row[2]:<10} {row[3]:<10} {row[4]:<10} {row[5]:<10}"
        
        # Highlight best baseline
        if baseline == 'joint_fused':
            print(f"-> {formatted_row}")
        else:
            print(f"  {formatted_row}")
    
    print("="*90 + "\n")
    
    # Generate LaTeX table
    latex_path = os.path.join(output_dir, 'baseline_comparison.tex')
    with open(latex_path, 'w') as f:
        f.write("% Baseline Comparison Table - Macro-F1 Scores\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Baseline Comparison: Macro-F1 Scores Across MBTI Dimensions}\n")
        f.write("\\label{tab:baseline_comparison}\n")
        f.write("\\begin{tabular}{lcccc|c}\n")
        f.write("\\hline\n")
        f.write("Baseline & IE & NS & TF & JP & Average \\\\\n")
        f.write("\\hline\n")
        
        for baseline in baselines:
            name_formatted = baseline.replace('_', ' ').title()
            row_values = [f"{comparison[baseline][dim]:.3f}" for dim in DIMENSIONS]
            avg_value = f"{comparison[baseline]['avg']:.3f}"
            
            if baseline == 'joint_fused':
                f.write(f"\\textbf{{{name_formatted}}} & ")
                f.write(" & ".join([f"\\textbf{{{v}}}" for v in row_values]))
                f.write(f" & \\textbf{{{avg_value}}} \\\\\n")
            else:
                f.write(f"{name_formatted} & ")
                f.write(" & ".join(row_values))
                f.write(f" & {avg_value} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"[OK] LaTeX table saved to: {latex_path}")
    
    return full_comparison


def compute_ablation_statistics(results: Dict[str, Dict[str, Dict[str, Any]]],
                                output_dir: str) -> Dict[str, Any]:
    """
    Compute ablation study statistics.
    
    Args:
        results: Baseline results
        output_dir: Output directory
    
    Returns:
        dict: Ablation statistics
    """
    # Extract F1 scores
    joint_f1 = np.array([results['joint_fused'][dim]['f1_macro'] for dim in DIMENSIONS])
    cognitive_f1 = np.array([results['cognitive_only'][dim]['f1_macro'] for dim in DIMENSIONS])
    semantic_f1 = np.array([results['semantic_only'][dim]['f1_macro'] for dim in DIMENSIONS])
    
    # Compute improvements
    ablation_stats = {}
    
    # Joint vs. Cognitive
    delta_cog = joint_f1 - cognitive_f1
    ablation_stats['joint_vs_cognitive'] = {
        'mean_improvement': float(np.mean(delta_cog)),
        'std_improvement': float(np.std(delta_cog)),
        'per_dimension': dict(zip(DIMENSIONS, delta_cog.tolist())),
        'num_improved': int(np.sum(delta_cog > 0)),
        'num_degraded': int(np.sum(delta_cog < 0)),
        'num_dimensions': len(DIMENSIONS)
    }
    
    # Joint vs. Semantic
    delta_sem = joint_f1 - semantic_f1
    ablation_stats['joint_vs_semantic'] = {
        'mean_improvement': float(np.mean(delta_sem)),
        'std_improvement': float(np.std(delta_sem)),
        'per_dimension': dict(zip(DIMENSIONS, delta_sem.tolist())),
        'num_improved': int(np.sum(delta_sem > 0)),
        'num_degraded': int(np.sum(delta_sem < 0)),
        'num_dimensions': len(DIMENSIONS)
    }
    
    # Save ablation statistics
    ablation_path = os.path.join(output_dir, 'ablation_statistics.json')
    with open(ablation_path, 'w') as f:
        json.dump(ablation_stats, f, indent=2)
    
    # Print ablation report
    print("\n" + "="*90)
    print("ABLATION STUDY ANALYSIS")
    print("="*90)
    
    print("\n1. Joint vs. Cognitive-Only")
    print("-" * 90)
    stats_cog = ablation_stats['joint_vs_cognitive']
    print(f"  Mean improvement: {stats_cog['mean_improvement']:+.4f} +/- {stats_cog['std_improvement']:.4f}")
    print(f"  Dimensions improved: {stats_cog['num_improved']}/{stats_cog['num_dimensions']}")
    print(f"  Per-dimension deltas:")
    for dim, delta in stats_cog['per_dimension'].items():
        print(f"    {dim}: {delta:+.4f}")
    
    print("\n2. Joint vs. Semantic-Only")
    print("-" * 90)
    stats_sem = ablation_stats['joint_vs_semantic']
    print(f"  Mean improvement: {stats_sem['mean_improvement']:+.4f} +/- {stats_sem['std_improvement']:.4f}")
    print(f"  Dimensions improved: {stats_sem['num_improved']}/{stats_sem['num_dimensions']}")
    print(f"  Per-dimension deltas:")
    for dim, delta in stats_sem['per_dimension'].items():
        print(f"    {dim}: {delta:+.4f}")
    
    print("\n3. Interpretation")
    print("-" * 90)
    if stats_cog['mean_improvement'] > 0 and stats_sem['mean_improvement'] > 0:
        print("  -> Joint fusion provides complementary information from both modalities")
    elif stats_sem['mean_improvement'] < 0.01:
        print("  -> Semantic features dominate; cognitive features provide modest regularization")
    else:
        print("  -> Mixed results; fusion effectiveness varies by dimension")
    
    print("="*90 + "\n")
    
    print(f"[OK] Ablation statistics saved to: {ablation_path}")
    
    return ablation_stats


def run_complete_baseline_experiments(npz_path: str, output_parent_dir: str) -> Dict[str, Any]:
    """
    Run all three baseline experiments with comprehensive evaluation.
    
    Args:
        npz_path: Path to phase2_features.npz
        output_parent_dir: Parent output directory
    
    Returns:
        dict: All baseline results
    """
    # Load data
    data = np.load(npz_path, allow_pickle=True)
    C = data['C']  # (N, 10) - Cognitive features
    S = data['S']  # (N, 768) - Semantic features (BERT)
    J = data['J']  # (N, 512) - Joint fused features
    labels = list(data['labels'])
    
    print(f"\n{'='*90}")
    print("COMPREHENSIVE BASELINE EVALUATION FOR MBTI CLASSIFICATION")
    print(f"{'='*90}")
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(labels)}")
    print(f"  Cognitive features (C): {C.shape}")
    print(f"  Semantic features (S): {S.shape}")
    print(f"  Joint features (J): {J.shape}")
    
    # Analyze imbalance once
    imbalance_stats = analyze_imbalance(labels)
    
    # Run baselines
    baselines_config = {
        'cognitive_only': C,
        'semantic_only': S,
        'joint_fused': J
    }
    
    results = {}
    
    for name, features in baselines_config.items():
        output_dir = os.path.join(output_parent_dir, name)
        results[name] = evaluate_baseline(features, labels, name, output_dir)
        
        # Generate combined figures
        print(f"\nGenerating combined visualizations for {name}...")
        generate_combined_figures(results[name], output_dir)
    
    # Generate comparison table
    print("\n" + "="*90)
    print("GENERATING COMPARATIVE ANALYSIS")
    print("="*90)
    
    comparison = generate_comparison_table(results, output_parent_dir)
    
    # Compute ablation statistics
    ablation = compute_ablation_statistics(results, output_parent_dir)
    
    # Save complete results summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'dataset': {
            'n_samples': len(labels),
            'n_classes': 16,
            'imbalance_ratio': imbalance_stats['imbalance_ratio']
        },
        'baselines': comparison,
        'ablation': ablation
    }
    
    summary_path = os.path.join(output_parent_dir, 'complete_results_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[OK] Complete summary saved to: {summary_path}")
    
    return results


if __name__ == '__main__':
    # Paths
    npz_path = os.path.join('data', 'processed', 'phase2_features.npz')
    output_dir = os.path.join('results', 'baselines')
    
    if not os.path.exists(npz_path):
        print(f"Error: Input file not found: {npz_path}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run complete baseline experiments
    results = run_complete_baseline_experiments(npz_path, output_dir)
    
    print("\n" + "="*90)
    print("[OK] ALL BASELINE EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("="*90)
    print(f"\nResults location: {os.path.abspath(output_dir)}")
    print("\nGenerated outputs:")
    print("  Baseline-specific:")
    print("    - cognitive_only/metrics.json")
    print("    - semantic_only/metrics.json")
    print("    - joint_fused/metrics.json")
    print("    - */confusion_matrix_*.png (per dimension)")
    print("    - */roc_curve_*.png (per dimension)")
    print("    - */confusion_matrices_grid.png (combined)")
    print("    - */roc_curves_grid.png (combined)")
    print("\n  Comparative analysis:")
    print("    - baseline_comparison.json")
    print("    - baseline_comparison.tex (LaTeX table)")
    print("    - ablation_statistics.json")
    print("    - complete_results_summary.json")
    print("="*90 + "\n")
