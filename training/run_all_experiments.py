#!/usr/bin/env python3
"""
Run Complete Experimental Pipeline

This master script runs all three evaluation steps in sequence:
1. Joint baseline evaluation
2. Complete baseline comparison (C, S, J)
3. Stability under intervention analysis

All results are saved to the results/ directory with standardized structure.

Author: Dhanshri
Date: 2026-02-14
"""

import os
import sys
import time
from datetime import datetime

def print_header(title):
    """Print formatted section header."""
    print("\n" + "="*100)
    print(f"  {title}")
    print("="*100 + "\n")

def run_experiments():
    """Run complete experimental pipeline."""
    
    start_time = time.time()
    
    print("\n" + "="*100)
    print("  MBTI CLASSIFICATION - COMPLETE EXPERIMENTAL PIPELINE")
    print("  Springer Conference Paper Evaluation")
    print("="*100)
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis will run:")
    print("  1. Joint baseline evaluation (experiments.py)")
    print("  2. Complete baseline comparison (baseline_comparison.py)")
    print("  3. Stability analysis (stability_analysis.py)")
    print("\n" + "-"*100)
    
    # Check if data exists
    npz_path = os.path.join('data', 'processed', 'phase2_features.npz')
    if not os.path.exists(npz_path):
        print(f"\n❌ ERROR: Required data file not found: {npz_path}")
        print("Please ensure Phase-2 features are generated first.")
        sys.exit(1)
    
    print(f"\n✓ Data file verified: {npz_path}\n")
    
    # Step 1: Joint Baseline Evaluation
    print_header("STEP 1/3: Joint Baseline Evaluation")
    print("Running experiments.py...")
    try:
        import training.experiments as exp
        J, labels = exp.load_phase2(npz_path)
        results_joint = exp.evaluate_joint_baseline(
            J, labels,
            output_dir='results/joint_baseline'
        )
        print("\n✓ Joint baseline evaluation completed successfully")
    except Exception as e:
        print(f"\n❌ ERROR in joint baseline evaluation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Complete Baseline Comparison
    print_header("STEP 2/3: Complete Baseline Comparison")
    print("Running baseline_comparison.py...")
    try:
        import training.baseline_comparison as bc
        results_baselines = bc.run_complete_baseline_experiments(
            npz_path,
            output_parent_dir='results/baselines'
        )
        print("\n✓ Baseline comparison completed successfully")
    except Exception as e:
        print(f"\n❌ ERROR in baseline comparison: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Stability Analysis
    print_header("STEP 3/3: Stability Under Intervention Analysis")
    print("Running stability_analysis.py...")
    try:
        import training.stability_analysis as sa
        stability_metrics = sa.analyze_stability_on_joint_baseline(
            npz_path,
            intervention_feature='pronoun_ratio',
            intervention_lambda=0.3,
            output_dir='results/stability'
        )
        print("\n✓ Stability analysis completed successfully")
    except Exception as e:
        print(f"\n❌ ERROR in stability analysis: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Summary
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    print_header("PIPELINE COMPLETE - SUMMARY")
    
    print("✓ All experiments completed successfully!")
    print(f"\nTotal execution time: {minutes} min {seconds} sec")
    print(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "-"*100)
    print("RESULTS DIRECTORY STRUCTURE:")
    print("-"*100)
    print("""
results/
├── joint_baseline/
│   ├── metrics.json
│   ├── confusion_matrix_*.png (4 files)
│   ├── roc_curve_*.png (4 files)
│   ├── confusion_matrices_grid.png
│   └── roc_curves_grid.png
│
├── baselines/
│   ├── cognitive_only/
│   │   └── (metrics.json + 10 figures)
│   ├── semantic_only/
│   │   └── (metrics.json + 10 figures)
│   ├── joint_fused/
│   │   └── (metrics.json + 10 figures)
│   ├── baseline_comparison.json
│   ├── baseline_comparison.tex         ← Use in paper
│   ├── ablation_statistics.json
│   └── complete_results_summary.json
│
└── stability/
    ├── stability_metrics.json
    └── stability_table.tex              ← Use in paper
    """)
    
    print("-"*100)
    print("\nFOR YOUR PAPER:")
    print("-"*100)
    print("""
1. LaTeX Tables (ready to copy-paste):
   - results/baselines/baseline_comparison.tex
   - results/stability/stability_table.tex

2. Figures (publication-quality, 300 DPI):
   - results/baselines/joint_fused/confusion_matrices_grid.png
   - results/baselines/joint_fused/roc_curves_grid.png

3. Metrics (JSON format for further analysis):
   - results/baselines/complete_results_summary.json
   - results/baselines/ablation_statistics.json
   - results/stability/stability_metrics.json
    """)
    
    print("="*100 + "\n")

if __name__ == '__main__':
    try:
        run_experiments()
    except KeyboardInterrupt:
        print("\n\n❌ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
