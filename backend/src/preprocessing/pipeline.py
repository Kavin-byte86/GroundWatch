"""
pipeline.py — Full Preprocessing Pipeline Orchestrator
=======================================================

Orchestrates: filtering → feature engineering → normalization → output.

Reads raw CSVs from data/raw/synthetic/, applies all preprocessing,
splits by panel ID, normalises features (z-score fit on TRAIN only),
and writes final feature matrices as Parquet files.

Usage:
    cd D:\\groundwatch-demo\\backend
    python src/preprocessing/pipeline.py
"""

import os
import sys
import json
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _BACKEND_ROOT)

from src.preprocessing.filtering import (
    resample_to_uniform_grid, remove_invalid_values,
    compensate_temperature_drift,
)
from src.preprocessing.feature_engineering import (
    build_features_for_node, compute_cross_node_correlation,
)


def load_config():
    cfg_path = os.path.join(_BACKEND_ROOT, "configs", "config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


# Feature columns used by the ML models (excludes metadata)
FEATURE_COLS = [
    "tilt_current", "tilt_rate_1h",
    "tilt_mean_6h", "tilt_std_6h",
    "tilt_mean_24h", "tilt_std_24h",
    "vib_rms", "vib_peak", "vib_dom_freq", "vib_spectral_ratio",
    "strain_current", "strain_rate_1h", "strain_cumulative",
    "crack_proxy_current", "crack_proxy_cumulative",
    "cross_node_corr",
    "temp_c",
]

META_COLS = ["timestamp", "node_id", "panel_id", "lat", "lng"]


def process_panel(panel_dir, panel_id, cfg):
    """Process all nodes for a single panel: filter → engineer features."""
    meta_path = os.path.join(panel_dir, "panel_meta.json")
    with open(meta_path, "r") as f:
        panel_meta = json.load(f)

    node_infos = panel_meta["nodes"]
    node_ids = [n["node_id"] for n in node_infos]

    # Load and filter all node DataFrames
    filtered_dfs = []
    for ninfo in node_infos:
        csv_path = os.path.join(panel_dir, f"{ninfo['node_id'].lower()}.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)

        # Apply filtering pipeline
        df = resample_to_uniform_grid(df, interval_hours=1, max_gap_hours=6)
        df = remove_invalid_values(df)
        df = compensate_temperature_drift(df, slope_per_c=cfg["noise"]["temp_drift_slope_per_c"])

        filtered_dfs.append(df)

    if not filtered_dfs:
        return pd.DataFrame()

    # Compute cross-node correlation (requires all nodes in the panel)
    cross_corrs = compute_cross_node_correlation(
        filtered_dfs, node_ids[:len(filtered_dfs)], window_size=24
    )

    # Build features for each node
    all_features = []
    for i, df in enumerate(filtered_dfs):
        nid = node_ids[i] if i < len(node_ids) else f"N{i+1:02d}"
        corr = cross_corrs.get(nid, None)
        feat = build_features_for_node(df, corr)
        feat["panel_id"] = panel_id
        all_features.append(feat)

    return pd.concat(all_features, ignore_index=True)


def split_by_panel(all_features, labels_df, cfg):
    """Split dataset by panel ID into train/val/test.

    WHY PANEL-LEVEL SPLIT: If we split by individual rows, the model
    would see other time-windows from the SAME panel geometry it's
    being tested on. Since all windows from a panel share the same
    underlying subsidence profile (depth, width, S_max), this would
    leak information about the panel's physics into the test set,
    inflating reported accuracy. Splitting by panel ensures the model
    is evaluated on entirely unseen panel geometries.
    """
    split_cfg = cfg["split"]
    panel_ids = sorted(all_features["panel_id"].unique())

    # Deterministic split based on config counts
    n_train = split_cfg["train_panels"]
    n_val = split_cfg["val_panels"]

    train_panels = panel_ids[:n_train]
    val_panels = panel_ids[n_train:n_train + n_val]
    test_panels = panel_ids[n_train + n_val:]

    print(f"  Train panels ({len(train_panels)}): {train_panels}")
    print(f"  Val panels   ({len(val_panels)}):   {val_panels}")
    print(f"  Test panels  ({len(test_panels)}):  {test_panels}")

    # Merge labels into features
    labels_df = labels_df.rename(columns={"window_timestamp": "timestamp"})
    labels_df["timestamp"] = pd.to_datetime(labels_df["timestamp"])
    all_features["timestamp"] = pd.to_datetime(all_features["timestamp"])

    merged = all_features.merge(
        labels_df[["panel_id", "node_id", "timestamp", "label"]],
        on=["panel_id", "node_id", "timestamp"],
        how="left"
    )
    merged["label"] = merged["label"].fillna("normal")

    train = merged[merged["panel_id"].isin(train_panels)].copy()
    val = merged[merged["panel_id"].isin(val_panels)].copy()
    test = merged[merged["panel_id"].isin(test_panels)].copy()

    return train, val, test


def normalize_features(train, val, test, feature_cols):
    """Z-score normalization, fit ONLY on training split.

    The scaler is fit on train data and applied to val/test to prevent
    data leakage. The fitted scaler is saved for use in production
    inference.
    """
    scaler = StandardScaler()

    # Handle NaN/inf in feature columns
    for df in [train, val, test]:
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        df[feature_cols] = df[feature_cols].fillna(0)

    scaler.fit(train[feature_cols])

    train[feature_cols] = scaler.transform(train[feature_cols])
    val[feature_cols] = scaler.transform(val[feature_cols])
    test[feature_cols] = scaler.transform(test[feature_cols])

    # Save scaler for production use
    scaler_path = os.path.join(_BACKEND_ROOT, "models", "scaler.pkl")
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved to: {scaler_path}")

    return train, val, test


def main():
    cfg = load_config()
    synthetic_dir = os.path.join(_BACKEND_ROOT, "data", "raw", "synthetic")
    labels_path = os.path.join(_BACKEND_ROOT, "data", "labels", "window_labels.csv")

    if not os.path.exists(labels_path):
        print("ERROR: Labels file not found. Run generate_dataset.py first.")
        sys.exit(1)

    labels_df = pd.read_csv(labels_path)

    # Process all panels
    panel_dirs = sorted([
        d for d in os.listdir(synthetic_dir)
        if os.path.isdir(os.path.join(synthetic_dir, d))
    ])

    print(f"Processing {len(panel_dirs)} panels...")
    all_features = []
    for pdir in panel_dirs:
        print(f"  Processing {pdir}...", end=" ", flush=True)
        feat = process_panel(os.path.join(synthetic_dir, pdir), pdir, cfg)
        if len(feat) > 0:
            all_features.append(feat)
            print(f"{len(feat)} windows")
        else:
            print("SKIP (no data)")

    all_features = pd.concat(all_features, ignore_index=True)
    print(f"\nTotal feature windows: {len(all_features)}")

    # Split by panel
    print("\nSplitting by panel ID...")
    train, val, test = split_by_panel(all_features, labels_df, cfg)
    print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # Normalize
    print("\nNormalizing features (z-score, fit on train)...")
    available_features = [c for c in FEATURE_COLS if c in train.columns]
    train, val, test = normalize_features(train, val, test, available_features)

    # Save as Parquet
    processed_dir = os.path.join(_BACKEND_ROOT, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # Select columns to save
    save_cols = META_COLS + available_features + ["label"]
    save_cols = [c for c in save_cols if c in train.columns]

    train[save_cols].to_parquet(os.path.join(processed_dir, "features_train.parquet"), index=False)
    val[save_cols].to_parquet(os.path.join(processed_dir, "features_val.parquet"), index=False)
    test[save_cols].to_parquet(os.path.join(processed_dir, "features_test.parquet"), index=False)

    print(f"\nParquet files written to: {processed_dir}")

    # Print label distribution per split
    for name, df in [("Train", train), ("Val", val), ("Test", test)]:
        print(f"\n{name} label distribution:")
        dist = df["label"].value_counts()
        for label, count in dist.items():
            pct = 100 * count / len(df)
            print(f"    {label:25s}  {count:>7d}  ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
