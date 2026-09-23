"""
train.py -- Model Training Pipeline
====================================

Trains both models (Isolation Forest + XGBoost) on the preprocessed
feature matrices, tunes hyperparameters on validation, and saves trained
models under the 5MB edge-deployment limit.

Usage:
    cd D:\\groundwatch-demo\\backend
    python src/training/train.py

Prerequisites:
    1. Run generate_dataset.py first (creates raw data + labels).
    2. Run preprocessing/pipeline.py first (creates Parquet features).
"""

import os
import sys
import time
import yaml
import numpy as np
import pandas as pd
import joblib

_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _BACKEND_ROOT)

from src.models.isolation_forest import IsolationForestModel
from src.models.xgboost_classifier import XGBoostModel, collapse_labels
from src.preprocessing.pipeline import FEATURE_COLS


def load_config():
    cfg_path = os.path.join(_BACKEND_ROOT, "configs", "config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def load_splits():
    """Load train/val/test Parquet files."""
    proc_dir = os.path.join(_BACKEND_ROOT, "data", "processed")
    train = pd.read_parquet(os.path.join(proc_dir, "features_train.parquet"))
    val = pd.read_parquet(os.path.join(proc_dir, "features_val.parquet"))
    test = pd.read_parquet(os.path.join(proc_dir, "features_test.parquet"))
    return train, val, test


def get_feature_matrix(df, feature_cols):
    """Extract feature matrix, handling missing columns gracefully."""
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].values.astype(float)
    # Replace any remaining NaN/inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, available


def check_model_size(path, max_mb=5):
    """Assert model file is under the edge-deployment size limit.

    This is a HARD REQUIREMENT -- models run on i3-class edge hardware
    with limited RAM. A 50MB model would be unusable; 5MB is the
    practical ceiling for fast joblib deserialization + inference.
    """
    size_bytes = os.path.getsize(path)
    size_mb = size_bytes / (1024 * 1024)
    status = "OK" if size_mb <= max_mb else "EXCEEDED"
    print(f"    Model size: {size_mb:.2f} MB  [{status}, limit={max_mb} MB]")
    assert size_mb <= max_mb, (
        f"Model at {path} is {size_mb:.2f} MB, exceeding the {max_mb} MB "
        f"edge-deployment limit. Reduce n_estimators or max_depth."
    )


def main():
    cfg = load_config()
    model_cfg = cfg["models"]
    models_dir = os.path.join(_BACKEND_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)

    pipeline_start = time.time()

    print("Loading preprocessed data...")
    train, val, test = load_splits()

    X_train, feat_cols = get_feature_matrix(train, FEATURE_COLS)
    X_val, _ = get_feature_matrix(val, feat_cols)
    y_train = train["label"].values
    y_val = val["label"].values

    print(f"  Train samples: {len(X_train)}")
    print(f"  Val samples:   {len(X_val)}")
    print(f"  Features used: {len(feat_cols)}")

    # ==============================================================
    #  ISOLATION FOREST -- train on normal-only data
    # ==============================================================
    print("\n" + "=" * 60)
    print("TRAINING: Isolation Forest (anomaly detection)")
    print("=" * 60)

    if_start = time.time()

    # Step 1: Tune contamination on validation set
    if_cfg = model_cfg["isolation_forest"]
    if_model = IsolationForestModel(
        n_estimators=if_cfg["n_estimators"],
        max_features=if_cfg["max_features"],
        contamination=if_cfg.get("contamination", "auto"),
        random_state=if_cfg["random_state"],
    )

    optimal_contamination = if_model.tune_contamination(X_val, y_val)
    print(f"  Tuned contamination: {optimal_contamination:.4f}")
    print(f"  (= actual anomaly rate on validation set)")

    # Step 2: Re-create with tuned contamination and train on normal-only
    if_model = IsolationForestModel(
        n_estimators=if_cfg["n_estimators"],
        max_features=if_cfg["max_features"],
        contamination=optimal_contamination,
        random_state=if_cfg["random_state"],
    )

    # CRITICAL: train ONLY on normal windows -- the IF learns what
    # "normal" looks like and flags everything else as anomalous.
    normal_mask = y_train == "normal"
    X_train_normal = X_train[normal_mask]
    print(f"  Training on {len(X_train_normal)} normal-only windows...")
    if_model.fit(X_train_normal)

    # Validate
    val_scores = if_model.anomaly_scores(X_val)
    val_anomaly = y_val != "normal"
    mean_score_normal = val_scores[~val_anomaly].mean()
    mean_score_anomaly = val_scores[val_anomaly].mean()
    print(f"  Val mean anomaly score - normal: {mean_score_normal:.4f}, "
          f"anomalous: {mean_score_anomaly:.4f}")
    print(f"  Separation: {mean_score_anomaly - mean_score_normal:.4f}")

    if_elapsed = time.time() - if_start
    print(f"  Training time: {if_elapsed:.1f}s")

    # Save
    if_path = os.path.join(models_dir, "isolation_forest.pkl")
    joblib.dump(if_model, if_path)
    print(f"  Saved to: {if_path}")
    check_model_size(if_path, model_cfg["max_model_size_mb"])

    # ==============================================================
    #  XGBOOST -- train on all labelled classes
    # ==============================================================
    print("\n" + "=" * 60)
    print("TRAINING: XGBoost (multi-class classifier)")
    print("=" * 60)

    xgb_start = time.time()

    xgb_cfg = model_cfg["xgboost"]
    xgb_model = XGBoostModel(
        n_estimators=xgb_cfg["n_estimators"],
        max_depth=xgb_cfg["max_depth"],
        learning_rate=xgb_cfg["learning_rate"],
        tree_method=xgb_cfg["tree_method"],
        early_stopping_rounds=xgb_cfg.get("early_stopping_rounds", 20),
        random_state=xgb_cfg["random_state"],
    )

    # Collapse subsidence tiers for training
    y_train_collapsed = collapse_labels(y_train)
    y_val_collapsed = collapse_labels(y_val)

    # Print class distribution
    unique, counts = np.unique(y_train_collapsed, return_counts=True)
    print("  Training class distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"    {cls:20s}  {cnt:>7d}  ({100*cnt/len(y_train_collapsed):5.1f}%)")

    print(f"\n  Training on {len(X_train)} windows across {len(unique)} classes...")
    print(f"  (with early stopping on validation set, patience={xgb_cfg.get('early_stopping_rounds', 20)})")

    # Train with validation set for early stopping
    xgb_model.fit(X_train, y_train_collapsed, X_val=X_val, y_val=y_val_collapsed)

    xgb_elapsed = time.time() - xgb_start

    # Validate
    val_pred = xgb_model.predict(X_val)
    val_acc = np.mean(val_pred == y_val_collapsed)
    print(f"  Validation accuracy: {val_acc:.4f}")

    # Per-class recall
    for cls in sorted(unique):
        mask = y_val_collapsed == cls
        if mask.sum() > 0:
            cls_acc = np.mean(val_pred[mask] == cls)
            print(f"    {cls:20s}  recall={cls_acc:.4f}  (n={mask.sum()})")

    # Feature importance (top 10)
    importances = xgb_model.model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    print("\n  Top 10 feature importances:")
    for rank, idx in enumerate(top_idx, 1):
        if idx < len(feat_cols):
            print(f"    {rank:2d}. {feat_cols[idx]:25s}  {importances[idx]:.4f}")

    print(f"\n  Training time: {xgb_elapsed:.1f}s")

    # Save model + label encoder (both needed at inference time)
    xgb_path = os.path.join(models_dir, "xgboost_classifier.pkl")
    encoder_path = os.path.join(models_dir, "label_encoder.pkl")
    joblib.dump(xgb_model, xgb_path)
    joblib.dump(xgb_model.label_encoder, encoder_path)
    print(f"  Model saved to: {xgb_path}")
    print(f"  Label encoder saved to: {encoder_path}")
    check_model_size(xgb_path, model_cfg["max_model_size_mb"])

    total_elapsed = time.time() - pipeline_start
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Models saved to: {models_dir}")
    print(f"Total wall-clock time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
