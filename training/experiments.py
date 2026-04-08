#!/usr/bin/env python3
"""
Comprehensive Logistic Regression Evaluation for MBTI Classification

This script provides Springer conference-ready experimental evaluation:
- Stratified train/test splitting
- Balanced class weights
- Comprehensive metrics: Precision, Recall, F1, Confusion Matrix, ROC/AUC
- Publication-quality visualizations
- Structured JSON output

Author: Dhanshri
Date: 2026-02-14
"""
from typing import Tuple, Dict, List, Any
import os
import sys
import json
from datetime import datetime
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

DIMENSIONS = ["IE", "NS", "TF", "JP"]
RANDOM_STATE = 42

# Set seeds for reproducibility
np.random.seed(RANDOM_STATE)


def load_phase2(npz_path: str) -> Tuple[np.ndarray, List[str]]:
    """Load Phase-2 features and labels."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Phase-2 file not found: {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    
    if "J" not in data:
        raise KeyError("Input .npz must contain 'J' array (joint vectors)")
    if "labels" not in data:
        raise KeyError("Input .npz must contain 'labels' array")
    
    J = data["J"]
    labels = list(data["labels"])
    
    return J, labels


def labels_to_targets(labels: List[str]) -> np.ndarray:
    """
    Convert MBTI labels (e.g., 'INFJ') into binary targets per dimension.
    
    Returns array shape (N, 4) with binary values:
      IE: I=0, E=1
      NS: N=0, S=1
      TF: T=0, F=1
      JP: J=0, P=1
    """
    N = len(labels)
    y = np.zeros((N, 4), dtype=int)
    
    for i, lbl in enumerate(labels):
        if not isinstance(lbl, str) or len(lbl) < 4:
            raise ValueError(f"Invalid MBTI label at index {i}: {lbl!r}")
        
        lbl = lbl.strip().upper()
        y[i, 0] = 1 if lbl[0] == "E" else 0
        y[i, 1] = 1 if lbl[1] == "S" else 0
        y[i, 2] = 1 if lbl[2] == "F" else 0
        y[i, 3] = 1 if lbl[3] == "P" else 0
    
    return y


def analyze_imbalance(labels: List[str]) -> Dict[str, Any]:
    """
    Analyze class distribution and imbalance statistics.
    
    Returns:
        dict: Imbalance statistics
    """
    dist = Counter(labels)
    total = len(labels)
    counts = np.array(list(dist.values()))
    
    stats = {
        'total_samples': total,
        'num_classes': len(dist),
        'distribution': dict(sorted(dist.items(), key=lambda x: x[1], reverse=True)),
        'max_samples': int(counts.max()),
        'min_samples': int(counts.min()),
        'imbalance_ratio': float(counts.max() / counts.min()),
        'mean_samples_per_class': float(counts.mean()),
        'std_samples_per_class': float(counts.std())
    }
    
    print("\n" + "="*70)
    print("CLASS IMBALANCE ANALYSIS")
    print("="*70)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Number of classes: {stats['num_classes']}")
    print(f"\nClass Distribution:")
    
    for cls, count in stats['distribution'].items():
        percentage = (count / total) * 100
        print(f"  {cls}: {count:5d} ({percentage:5.2f}%)")
    
    print(f"\nImbalance Statistics:")
    print(f"  Max samples: {stats['max_samples']}")
    print(f"  Min samples: {stats['min_samples']}")
    print(f"  Imbalance ratio: {stats['imbalance_ratio']:.2f}:1")
    print(f"  Mean +/- Std: {stats['mean_samples_per_class']:.1f} +/- {stats['std_samples_per_class']:.1f}")
    print("="*70 + "\n")
    
    return stats


def evaluate_dimension(y_true: np.ndarray, 
                       y_pred: np.ndarray, 
                       y_proba: np.ndarray,
                       dim_name: str,
                       output_dir: str) -> Dict[str, Any]:
    """
    Comprehensive evaluation for a single MBTI dimension.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        y_proba: Predicted probabilities (continuous [0,1])
        dim_name: Dimension name (IE, NS, TF, JP)
        output_dir: Directory to save plots
    
    Returns:
        dict: Complete metrics dictionary
    """
    metrics = {}
    
    # 1. Classification metrics
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics['precision_per_class'] = {
        'class_0': float(precision_per_class[0]),
        'class_1': float(precision_per_class[1])
    }
    metrics['recall_per_class'] = {
        'class_0': float(recall_per_class[0]),
        'class_1': float(recall_per_class[1])
    }
    metrics['f1_per_class'] = {
        'class_0': float(f1_per_class[0]),
        'class_1': float(f1_per_class[1])
    }
    
    # 2. Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'],
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {dim_name} Dimension', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=11)
    plt.xlabel('Predicted Label', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix_{dim_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    metrics['auc'] = float(roc_auc)
    metrics['roc_curve'] = {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist()
    }
    
    # Plot ROC curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate', fontsize=11)
    plt.title(f'ROC Curve - {dim_name} Dimension', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curve_{dim_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return metrics


def save_metrics(all_metrics: Dict[str, Dict[str, Any]], 
                 config: Dict[str, Any],
                 imbalance_stats: Dict[str, Any],
                 output_path: str) -> None:
    """Save all metrics in structured JSON format."""
    
    summary = {
        'macro_avg_accuracy': float(np.mean([m['accuracy'] for m in all_metrics.values()])),
        'macro_avg_precision': float(np.mean([m['precision_macro'] for m in all_metrics.values()])),
        'macro_avg_recall': float(np.mean([m['recall_macro'] for m in all_metrics.values()])),
        'macro_avg_f1': float(np.mean([m['f1_macro'] for m in all_metrics.values()])),
        'macro_avg_auc': float(np.mean([m['auc'] for m in all_metrics.values()]))
    }
    
    output = {
        'experiment_config': config,
        'imbalance_statistics': imbalance_stats,
        'timestamp': datetime.now().isoformat(),
        'dimensions': all_metrics,
        'summary': summary
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n[OK] Metrics saved to: {output_path}")


def print_summary_table(all_metrics: Dict[str, Dict[str, Any]]) -> None:
    """Print formatted summary table."""
    
    print("\n" + "="*90)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*90)
    print(f"{'Dimension':<12} {'Accuracy':<10} {'Precision':<12} {'Recall':<10} {'F1-Score':<10} {'AUC':<10}")
    print("-"*90)
    
    for dim in DIMENSIONS:
        m = all_metrics[dim]
        print(f"{dim:<12} {m['accuracy']:<10.4f} {m['precision_macro']:<12.4f} "
              f"{m['recall_macro']:<10.4f} {m['f1_macro']:<10.4f} {m['auc']:<10.4f}")
    
    print("-"*90)
    
    # Summary row
    avg_acc = np.mean([m['accuracy'] for m in all_metrics.values()])
    avg_prec = np.mean([m['precision_macro'] for m in all_metrics.values()])
    avg_rec = np.mean([m['recall_macro'] for m in all_metrics.values()])
    avg_f1 = np.mean([m['f1_macro'] for m in all_metrics.values()])
    avg_auc = np.mean([m['auc'] for m in all_metrics.values()])
    
    print(f"{'AVERAGE':<12} {avg_acc:<10.4f} {avg_prec:<12.4f} "
          f"{avg_rec:<10.4f} {avg_f1:<10.4f} {avg_auc:<10.4f}")
    print("="*90 + "\n")


def evaluate_joint_baseline(J: np.ndarray, 
                            labels: List[str],
                            output_dir: str = 'results/joint_baseline') -> Dict[str, Any]:
    """
    Comprehensive evaluation of joint representation baseline.
    
    Args:
        J: Joint feature vectors (N, 512)
        labels: MBTI labels (N,)
        output_dir: Output directory for results
    
    Returns:
        dict: All metrics and results
    """
    print("\n" + "="*70)
    print("JOINT REPRESENTATION BASELINE EVALUATION")
    print("="*70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze imbalance
    imbalance_stats = analyze_imbalance(labels)
    
    # Convert labels to binary targets
    y = labels_to_targets(labels)
    
    # Stratified train/test split
    print(f"Performing stratified train/test split (test_size=0.2, random_state={RANDOM_STATE})...")
    X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(
        J, y, labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,  # Stratify on 16 MBTI classes
        shuffle=True
    )
    
    print(f"  Train samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    
    # Verify stratification
    train_dist = Counter(labels_train)
    test_dist = Counter(labels_test)
    print(f"\n[OK] Stratification verified:")
    print(f"  Train classes: {len(train_dist)}")
    print(f"  Test classes: {len(test_dist)}")
    
    # Configuration
    config = {
        'baseline_name': 'joint_fused',
        'feature_dim': J.shape[1],
        'n_samples': J.shape[0],
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
            class_weight='balanced',  # Handles class imbalance
            solver='lbfgs'
        )
        clf.fit(X_train, y_train[:, dim_idx])
        
        # Predictions
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]  # Probability of class 1
        
        # Comprehensive evaluation
        metrics = evaluate_dimension(
            y_test[:, dim_idx],
            y_pred,
            y_proba,
            dim_name,
            output_dir
        )
        
        all_metrics[dim_name] = metrics
        
        print(f"  [OK] Accuracy: {metrics['accuracy']:.4f}, "
              f"F1-macro: {metrics['f1_macro']:.4f}, "
              f"AUC: {metrics['auc']:.4f}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    save_metrics(all_metrics, config, imbalance_stats, metrics_path)
    
    # Print summary
    print_summary_table(all_metrics)
    
    # Generate combined visualizations
    print("Generating combined visualization grids...")
    generate_combined_figures(all_metrics, output_dir)
    
    return all_metrics


def generate_combined_figures(results: Dict[str, Dict[str, Any]], output_dir: str) -> None:
    """Generate combined confusion matrix and ROC curve grids."""
    
    # Confusion Matrix Grid (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.flatten()
    
    for idx, dim in enumerate(DIMENSIONS):
        cm = np.array(results[dim]['confusion_matrix'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Class 0', 'Class 1'],
                    yticklabels=['Class 0', 'Class 1'],
                    ax=axes[idx], cbar=True,
                    cbar_kws={'label': 'Count'})
        
        axes[idx].set_title(f'{dim} Dimension', fontsize=13, fontweight='bold')
        axes[idx].set_xlabel('Predicted Label', fontsize=10)
        axes[idx].set_ylabel('True Label', fontsize=10)
    
    plt.suptitle('Confusion Matrices - All MBTI Dimensions', 
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrices_grid.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Confusion matrix grid saved")
    
    # ROC Curve Grid (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.flatten()
    
    for idx, dim in enumerate(DIMENSIONS):
        roc_data = results[dim]['roc_curve']
        auc_score = results[dim]['auc']
        
        fpr = roc_data['fpr']
        tpr = roc_data['tpr']
        
        axes[idx].plot(fpr, tpr, color='darkorange', lw=2,
                      label=f'ROC curve (AUC = {auc_score:.3f})')
        axes[idx].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
        
        axes[idx].set_xlim([0.0, 1.0])
        axes[idx].set_ylim([0.0, 1.05])
        axes[idx].set_xlabel('False Positive Rate', fontsize=10)
        axes[idx].set_ylabel('True Positive Rate', fontsize=10)
        axes[idx].set_title(f'{dim} Dimension', fontsize=13, fontweight='bold')
        axes[idx].legend(loc='lower right', fontsize=9)
        axes[idx].grid(alpha=0.3, linestyle='--')
    
    plt.suptitle('ROC Curves - All MBTI Dimensions',
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curves_grid.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] ROC curve grid saved")


if __name__ == "__main__":
    # Input/output paths
    input_npz = os.path.join("data", "processed", "phase2_features.npz")
    output_dir = os.path.join("results", "joint_baseline")
    
    if not os.path.exists(input_npz):
        print(f"Error: Input file not found: {input_npz}")
        sys.exit(1)
    
    # Load data
    print("Loading Phase-2 features...")
    J, labels = load_phase2(input_npz)
    print(f"[OK] Loaded {len(labels)} samples with {J.shape[1]}-dimensional features")
    
    # Run evaluation
    results = evaluate_joint_baseline(J, labels, output_dir)
    
    print("\n" + "="*70)
    print("[OK] EVALUATION COMPLETE")
    print("="*70)
    print(f"Results saved to: {os.path.abspath(output_dir)}")
    print("\nGenerated files:")
    print("  - metrics.json              (Complete metrics)")
    print("  - confusion_matrix_*.png    (Per-dimension confusion matrices)")
    print("  - roc_curve_*.png           (Per-dimension ROC curves)")
    print("  - confusion_matrices_grid.png (Combined grid)")
    print("  - roc_curves_grid.png       (Combined grid)")
    print("="*70 + "\n")
