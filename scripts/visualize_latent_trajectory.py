#!/usr/bin/env python3
"""
Visualize Latent Space & Intervention Trajectory
Projects the joint representation J into 2D (PCA) and visualizes the movement
caused by counterfactual interventions on cognitive features.
"""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple

# Ensure repo in path
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

from phase2_representation.fusion.gated_fusion import GatedFusion
from phase3_causal_reasoning.counterfactual.do_intervention import do_intervention

# Constants
OUTPUT_DIR = os.path.join(repo_path, "results", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

def visualize_trajectory():
    # 1. Load Data
    npz_path = os.path.join(repo_path, "data", "processed", "phase2_features.npz")
    if not os.path.exists(npz_path):
        print(f"Error: {npz_path} not found.")
        return

    data = np.load(npz_path, allow_pickle=True)
    C_arr = data['C']  # (N, 10)
    S_arr = data['S']  # (N, 768)
    J_arr = data['J']  # (N, 512)
    labels = data['labels']

    print(f"Loaded {len(labels)} samples.")

    # 2. PCA Projection of the existing Latent Space J
    print("Computing PCA projection...")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    J_2d = pca.fit_transform(J_arr)

    # 3. Select Representative Samples for Trajectory
    # We'll pick a few samples from different quadrants or classes if possible
    # For simplicity, let's pick 5 random indices
    sample_indices = np.random.choice(len(labels), 8, replace=False)
    
    # 4. Initialize a consistent GatedFusion model
    # We need this to compute the shift J -> J_cf
    fusion_model = GatedFusion(cognitive_dim=10, projection_dim=512, semantic_dim=768)
    # We want to use the factual J from the npz if possible, but to show a relative shift
    # we need to re-compute J_fact using THE SAME fusion model as J_cf.
    
    # 5. Compute Interventions
    # Strategy: Intervene on 'pronoun_ratio' (index 0) with lambda = 1.0 (max shift)
    # to make the arrow visible.
    feat_idx = 0 
    x_ref = np.mean(C_arr[:, feat_idx])
    lam = 1.0 # High lambda for visual clarity of the vector shift

    trajectories = []
    for idx in sample_indices:
        C_orig = torch.tensor(C_arr[idx], dtype=torch.float32)
        S_orig = torch.tensor(S_arr[idx], dtype=torch.float32)
        
        # Factual J (re-computed with fixed model for consistent traj)
        with torch.no_grad():
            J_fact = fusion_model(C_orig, S_orig)
            
            # Counterfactual C
            C_cf = C_orig.clone()
            x = float(C_orig[feat_idx].item())
            x_ref_val = float(x_ref)
            x_cf = do_intervention(x, x_ref_val, 0.4) # limit to max valid lam
            C_cf[feat_idx] = x_cf
            
            # Counterfactual J
            J_cf = fusion_model(C_cf, S_orig)
            
        # Project both to the same PCA space
        pts = pca.transform(torch.stack([J_fact, J_cf]).numpy())
        trajectories.append(pts)

    # 6. Plotting
    plt.figure(figsize=(12, 10))
    
    # Background: Plot all data points
    # Color by the first letter of MBTI (E vs I) to show some clustering
    colors = ['#1f77b4' if l[0] == 'E' else '#ff7f0e' for l in labels]
    plt.scatter(J_2d[:, 0], J_2d[:, 1], c=colors, alpha=0.15, s=20, label='Latent Space (J)')

    # Add centroids (conceptual)
    centroid_e = np.mean(J_2d[[l[0] == 'E' for l in labels]], axis=0)
    centroid_i = np.mean(J_2d[[l[0] == 'I' for l in labels]], axis=0)
    plt.scatter(centroid_e[0], centroid_e[1], c='#1f77b4', s=200, marker='X', edgecolor='black', label='Extrovert Centroid')
    plt.scatter(centroid_i[0], centroid_i[1], c='#ff7f0e', s=200, marker='X', edgecolor='black', label='Introvert Centroid')

    # Plot Trajectories
    for i, traj in enumerate(trajectories):
        plt.arrow(traj[0, 0], traj[0, 1], 
                  traj[1, 0] - traj[0, 0], traj[1, 1] - traj[0, 1],
                  head_width=0.05, head_length=0.08, fc='red', ec='red', 
                  linewidth=2, alpha=0.8, length_includes_head=True,
                  label='Intervention Trajectory' if i == 0 else "")
        plt.scatter(traj[0, 0], traj[0, 1], c='black', s=40, zorder=5)

    plt.title("Latent Space Intervention Trajectory (PCA Projection)", fontsize=16, fontweight='bold')
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} var)", fontsize=12)
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} var)", fontsize=12)
    plt.legend(loc='best', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    save_path = os.path.join(OUTPUT_DIR, "latent_trajectory_pca.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved to {save_path}")

if __name__ == "__main__":
    visualize_trajectory()
