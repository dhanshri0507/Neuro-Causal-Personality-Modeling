#!/usr/bin/env python3
"""
Train the gated cognitive–semantic fusion end-to-end (frozen BERT features in S).

The Phase-2 npz previously built J with a *new* random GatedFusion per document, which
makes J meaningless. This script:

1. Fits StandardScaler on C and S (train split only).
2. Trains HybridMBTIModel (fusion + four linear heads) with BCEWithLogitsLoss.
3. Compares test Macro-F1 to sklearn LogisticRegression on semantic-only (S) and
   concatenation [C; S] (strong sanity baseline).
4. Exports updated J for the full dataset to `data/processed/phase2_features.npz`
   (backup original to `phase2_features_backup_random_J.npz` on first run).

Run from repo root `mbti-neuro-causal`:

  python -m training.train_hybrid_torch

Requires: phase2_features.npz with C, S, labels (J is overwritten after training).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Dict, List, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from training.experiments import DIMENSIONS, RANDOM_STATE, labels_to_targets
from training.hybrid_torch_model import HybridMBTIModel


def _sklearn_macro_f1_per_dim(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    max_iter: int = 5000,
    C_param: float = 1.0,
) -> Tuple[Dict[str, float], float]:
    f1s = {}
    for i, name in enumerate(DIMENSIONS):
        clf = LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=max_iter,
            class_weight="balanced",
            solver="lbfgs",
            C=C_param,
        )
        clf.fit(X_train, y_train[:, i])
        pred = clf.predict(X_test)
        f1s[name] = float(f1_score(y_test[:, i], pred, average="macro", zero_division=0))
    return f1s, float(np.mean(list(f1s.values())))


def _torch_eval_macro_f1(
    model: HybridMBTIModel,
    c_t: torch.Tensor,
    s_t: torch.Tensor,
    y: np.ndarray,
    device: torch.device,
) -> Tuple[Dict[str, float], float]:
    model.eval()
    f1s: Dict[str, float] = {}
    with torch.no_grad():
        logits = model(c_t.to(device), s_t.to(device)).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        pred = (probs >= 0.5).astype(np.int64)
    for i, name in enumerate(DIMENSIONS):
        f1s[name] = float(f1_score(y[:, i], pred[:, i], average="macro", zero_division=0))
    return f1s, float(np.mean(list(f1s.values())))


def train_and_export(
    npz_path: str,
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 15,
    device: str | None = None,
) -> Dict:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing {npz_path}. Generate phase2 features first.")

    data = np.load(npz_path, allow_pickle=True)
    C = np.asarray(data["C"], dtype=np.float32)
    S = np.asarray(data["S"], dtype=np.float32)
    labels = list(data["labels"])
    y = labels_to_targets(labels)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    idx = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )
    # validation from train
    tr_labels = [labels[i] for i in train_idx]
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.125, random_state=RANDOM_STATE, stratify=tr_labels
    )

    sc_c = StandardScaler()
    sc_s = StandardScaler()
    C_train = sc_c.fit_transform(C[train_idx])
    C_val = sc_c.transform(C[val_idx])
    C_test = sc_c.transform(C[test_idx])
    S_train = sc_s.fit_transform(S[train_idx])
    S_val = sc_s.transform(S[val_idx])
    S_test = sc_s.transform(S[test_idx])

    # Sklearn baselines (same split)
    sem_f1_dim, sem_avg = _sklearn_macro_f1_per_dim(S_train, S_test, y[train_idx], y[test_idx])
    concat_train = np.concatenate([C_train, S_train], axis=1)
    concat_test = np.concatenate([C_test, S_test], axis=1)
    cat_f1_dim, cat_avg = _sklearn_macro_f1_per_dim(
        concat_train, concat_test, y[train_idx], y[test_idx]
    )

    print("\n=== Sklearn baselines (test) ===")
    print(f"Semantic-only avg Macro-F1: {sem_avg:.4f}  {sem_f1_dim}")
    print(f"Concat [C;S] avg Macro-F1:  {cat_avg:.4f}  {cat_f1_dim}")

    model = HybridMBTIModel(cognitive_dim=C.shape[1], projection_dim=512, semantic_dim=S.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    c_tr = torch.from_numpy(C_train).float()
    s_tr = torch.from_numpy(S_train).float()
    c_va = torch.from_numpy(C_val).float()
    s_va = torch.from_numpy(S_val).float()
    y_tr = y[train_idx]
    y_va = y[val_idx]

    pos_weights = []
    for i in range(4):
        pos = (y_tr[:, i] == 1).sum()
        neg = (y_tr[:, i] == 0).sum()
        pos_weights.append(float(neg) / max(float(pos), 1.0))
    pos_w = torch.tensor(pos_weights, dtype=torch.float32, device=dev)

    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    n = c_tr.size(0)
    best_state = None
    best_val = -1.0
    stale = 0

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx_b = perm[start : start + batch_size]
            cb = c_tr[idx_b].to(dev)
            sb = s_tr[idx_b].to(dev)
            yb = torch.from_numpy(y_tr[idx_b]).float().to(dev)
            opt.zero_grad()
            logits = model(cb, sb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * cb.size(0)

        model.eval()
        with torch.no_grad():
            lv = model(c_va.to(dev), s_va.to(dev))
            yv = torch.from_numpy(y_va).float().to(dev)
            vloss = float(crit(lv, yv).item())
        probs = torch.sigmoid(lv).cpu().numpy()
        pred = (probs >= 0.5).astype(np.int64)
        val_f1 = np.mean(
            [f1_score(y_va[:, i], pred[:, i], average="macro", zero_division=0) for i in range(4)]
        )

        if val_f1 > best_val:
            best_val = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        print(f"Epoch {ep+1}/{epochs}  train_loss={total_loss/n:.4f}  val_loss={vloss:.4f}  val_macro_f1={val_f1:.4f}")
        if stale >= patience:
            print(f"Early stop at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    c_te = torch.from_numpy(C_test).float()
    s_te = torch.from_numpy(S_test).float()
    hy_f1_dim, hy_avg = _torch_eval_macro_f1(model, c_te, s_te, y[test_idx], dev)

    print("\n=== Trained hybrid (test) ===")
    print(f"Hybrid (trained fusion) avg Macro-F1: {hy_avg:.4f}  {hy_f1_dim}")
    print(f"Delta vs semantic-only: {hy_avg - sem_avg:+.4f}")
    print(f"Delta vs concat [C;S]:  {hy_avg - cat_avg:+.4f}")

    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    ckpt_path = os.path.join(models_dir, "hybrid_fusion_trained.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "scaler_c_mean": sc_c.mean_,
            "scaler_c_scale": sc_c.scale_,
            "scaler_s_mean": sc_s.mean_,
            "scaler_s_scale": sc_s.scale_,
            "cognitive_dim": int(C.shape[1]),
            "semantic_dim": int(S.shape[1]),
            "projection_dim": 512,
            "random_state": RANDOM_STATE,
        },
        ckpt_path,
    )
    print(f"\nSaved checkpoint: {ckpt_path}")

    # Export J for full dataset
    model.eval()
    C_all = sc_c.transform(C)
    S_all = sc_s.transform(S)
    J_list = []
    with torch.no_grad():
        for start in range(0, len(C_all), batch_size):
            cb = torch.from_numpy(C_all[start : start + batch_size]).float().to(dev)
            sb = torch.from_numpy(S_all[start : start + batch_size]).float().to(dev)
            jb = model.encode_joint(cb, sb).cpu().numpy()
            J_list.append(jb)
    J_new = np.vstack(J_list).astype(np.float32)

    backup_path = npz_path.replace(".npz", "_backup_random_J.npz")
    if not os.path.exists(backup_path):
        shutil.copy2(npz_path, backup_path)
        print(f"Backed up original npz to {backup_path}")

    np.savez(npz_path, C=C, S=S, J=J_new, labels=np.array(labels, dtype=object))
    print(f"Updated {npz_path} with trained J (shape {J_new.shape})")

    summary = {
        "semantic_only_avg_f1": sem_avg,
        "semantic_per_dim": sem_f1_dim,
        "concat_avg_f1": cat_avg,
        "concat_per_dim": cat_f1_dim,
        "hybrid_trained_avg_f1": hy_avg,
        "hybrid_per_dim": hy_f1_dim,
        "checkpoint": ckpt_path,
    }
    rep = os.path.join(models_dir, "hybrid_train_report.json")
    with open(rep, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {rep}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz",
        default=os.path.join(_REPO_ROOT, "data", "processed", "phase2_features.npz"),
        help="Path to phase2_features.npz",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

    train_and_export(
        npz_path=args.npz,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
