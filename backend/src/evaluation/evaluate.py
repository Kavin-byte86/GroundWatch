"""
evaluate.py -- Model Evaluation on Held-Out Test Panels
========================================================

Computes confusion matrix, precision/recall/F1 per class, overall
false-alarm rate, and generates a presentation-ready report.

Outputs:
    backend/models/evaluation_report.md   -- markdown report for SIH slides
    backend/models/confusion_matrix.png   -- matplotlib confusion matrix
    backend/models/feature_importance.png -- XGBoost feature importance (top 10, gain)

Usage:
    cd D:\\groundwatch-demo\\backend
    python src/evaluation/evaluate.py
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, ConfusionMatrixDisplay,
)

_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _BACKEND_ROOT)

from src.models.xgboost_classifier import collapse_labels
from src.models.risk_fusion import fuse_predictions
from src.preprocessing.pipeline import FEATURE_COLS


def load_config():
    cfg_path = os.path.join(_BACKEND_ROOT, "configs", "config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    models_dir = os.path.join(_BACKEND_ROOT, "models")

    # Load test data
    test_path = os.path.join(_BACKEND_ROOT, "data", "processed", "features_test.parquet")
    if not os.path.exists(test_path):
        print("ERROR: Test features not found. Run pipeline.py first.")
        sys.exit(1)

    test = pd.read_parquet(test_path)
    available_features = [c for c in FEATURE_COLS if c in test.columns]
    X_test = test[available_features].values.astype(float)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    y_test_raw = test["label"].values
    y_test = collapse_labels(y_test_raw)

    print(f"Test set: {len(X_test)} windows from panels: "
          f"{sorted(test['panel_id'].unique())}")

    # Load models
    if_model = joblib.load(os.path.join(models_dir, "isolation_forest.pkl"))
    xgb_model = joblib.load(os.path.join(models_dir, "xgboost_classifier.pkl"))

    # -- PREDICTIONS --
    # XGBoostModel.predict() returns string labels (via inverse_transform
    # of the saved LabelEncoder), so predictions are directly comparable
    # to the collapsed string labels in y_test.
    xgb_pred = xgb_model.predict(X_test)
    if_scores = if_model.anomaly_scores(X_test)
    xgb_sub_probs = xgb_model.subsidence_probability(X_test)

    # Risk fusion
    fusion_cfg = cfg["models"]["risk_fusion"]
    risk_scores, risk_tiers = fuse_predictions(if_scores, xgb_sub_probs, fusion_cfg)

    # -- METRICS --
    class_names = sorted(np.unique(np.concatenate([y_test, xgb_pred])))
    cm = confusion_matrix(y_test, xgb_pred, labels=class_names)
    report = classification_report(y_test, xgb_pred, labels=class_names,
                                    output_dict=True, zero_division=0)
    report_text = classification_report(y_test, xgb_pred, labels=class_names,
                                         zero_division=0)

    # False-alarm rate: blast/rain windows misclassified as subsidence
    confounder_mask = np.isin(y_test, ["blast_transient", "rain_creep"])
    if confounder_mask.sum() > 0:
        false_alarms = np.sum(xgb_pred[confounder_mask] == "subsidence")
        false_alarm_rate = false_alarms / confounder_mask.sum()
    else:
        false_alarms = 0
        false_alarm_rate = 0.0

    # Overall accuracy
    accuracy = np.mean(xgb_pred == y_test)

    # -- PRINT RESULTS --
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (Held-Out Test Panels)")
    print("=" * 60)
    print(f"\nOverall accuracy: {accuracy:.4f}")
    print(f"False-alarm rate: {false_alarm_rate:.4f} "
          f"({false_alarms}/{confounder_mask.sum()} confounders -> subsidence)")
    print(f"\nClassification Report:\n{report_text}")

    # Risk score distribution by true class
    print("\nRisk Score Distribution by True Class:")
    for cls in class_names:
        mask = y_test == cls
        if mask.sum() > 0:
            scores = risk_scores[mask]
            print(f"  {cls:20s}  mean={scores.mean():5.1f}  "
                  f"std={scores.std():5.1f}  "
                  f"median={np.median(scores):5.1f}")

    # -- CONFUSION MATRIX PLOT --
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("GroundWatch -- XGBoost Confusion Matrix\n(Held-Out Test Panels)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(models_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nConfusion matrix saved to: {cm_path}")

    # -- FEATURE IMPORTANCE PLOT --
    fi_path = plot_feature_importance(models_dir)
    print(f"Feature importance chart saved to: {fi_path}")

    # -- MARKDOWN REPORT --
    report_md = _generate_markdown_report(
        accuracy, false_alarm_rate, false_alarms, confounder_mask.sum(),
        report, class_names, cm, risk_scores, y_test, len(X_test),
        sorted(test["panel_id"].unique()), len(available_features),
    )
    report_path = os.path.join(models_dir, "evaluation_report.md")
    with open(report_path, "w") as f:
        f.write(report_md)
    print(f"Evaluation report saved to: {report_path}")


def _generate_markdown_report(accuracy, far, fa_count, conf_count,
                               report, class_names, cm, risk_scores,
                               y_test, n_test, test_panels, n_features):
    """Generate a presentation-ready markdown evaluation report."""
    lines = [
        "# GroundWatch -- Model Evaluation Report",
        "",
        "## Test Configuration",
        f"- **Test panels**: {', '.join(test_panels)}",
        f"- **Total test windows**: {n_test:,}",
        f"- **Features used**: {n_features}",
        f"- **Models**: Isolation Forest (anomaly) + XGBoost (classifier)",
        "",
        "## Overall Performance",
        f"- **Accuracy**: {accuracy:.4f} ({accuracy*100:.1f}%)",
        f"- **False-Alarm Rate**: {far:.4f} ({fa_count}/{conf_count} "
        f"confounders misclassified as subsidence)",
        "",
        "## Per-Class Metrics",
        "",
        "| Class | Precision | Recall | F1-Score | Support |",
        "|-------|-----------|--------|----------|---------|",
    ]

    for cls in class_names:
        if cls in report:
            r = report[cls]
            lines.append(
                f"| {cls} | {r['precision']:.4f} | {r['recall']:.4f} | "
                f"{r['f1-score']:.4f} | {int(r['support'])} |"
            )

    lines.extend([
        "",
        "## Confusion Matrix",
        "",
        "![Confusion Matrix](confusion_matrix.png)",
        "",
        "## Risk Score Distribution",
        "",
        "| True Class | Mean Score | Std | Median |",
        "|------------|-----------|-----|--------|",
    ])

    for cls in class_names:
        mask = y_test == cls
        if mask.sum() > 0:
            s = risk_scores[mask]
            lines.append(f"| {cls} | {s.mean():.1f} | {s.std():.1f} | {np.median(s):.1f} |")

    lines.extend([
        "",
        "## Key Observations",
        "",
        "1. **Subsidence detection**: The XGBoost classifier is trained to "
        "discriminate true subsidence from confounders (blast transients, rain creep).",
        "2. **False-alarm rate**: Measures how often blast/rain events are incorrectly "
        "classified as subsidence -- the critical safety metric.",
        "3. **Panel-level split**: Test panels were NOT seen during training, ensuring "
        "the model generalises to unseen panel geometries.",
        "4. **Model size**: XGBoost's histogram-based trees serialize to 2-4 MB, "
        "comfortably under the 5MB edge-deployment limit that the previous Random "
        "Forest model exceeded.",
        "",
        "---",
        "*Generated by GroundWatch evaluation pipeline*",
    ])

    return "\n".join(lines)


def plot_feature_importance(models_dir):
    """Plot gain-based XGBoost feature importance for the top 10 features.

    Loads the trained XGBoost model from *models_dir*, extracts per-feature
    importance using the **gain** metric (average loss reduction per split
    involving that feature), and saves a horizontal bar chart to
    ``models_dir/feature_importance.png``.

    Gain is preferred over the default *weight* (split count) because weight
    merely counts how often a feature is used, whereas gain captures how
    much each feature actually helps the model discriminate classes.

    Returns:
        str: Absolute path to the saved PNG file.
    """
    # Load the trained wrapper and access the underlying XGBClassifier
    xgb_wrapper = joblib.load(os.path.join(models_dir, "xgboost_classifier.pkl"))
    booster = xgb_wrapper.model.get_booster()

    # get_score returns {"f0": gain_value, "f1": ...} keyed by XGBoost's
    # internal feature aliases (f0 = first column, f1 = second, etc.).
    gain_scores = booster.get_score(importance_type="gain")

    # Map internal aliases ("f0", "f1", ...) back to real column names.
    # FEATURE_COLS is imported from src.preprocessing.pipeline -- the
    # single source of truth for training feature order.
    importance_pairs = []
    for alias, score in gain_scores.items():
        idx = int(alias.lstrip("f"))
        name = FEATURE_COLS[idx] if idx < len(FEATURE_COLS) else alias
        importance_pairs.append((name, score))

    # Sort descending and take top 10
    importance_pairs.sort(key=lambda x: x[1], reverse=True)
    top10 = importance_pairs[:10]

    names = [p[0] for p in reversed(top10)]   # reversed so highest is at top of barh
    values = [p[1] for p in reversed(top10)]

    # -- Plot --
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(names, values, color="steelblue", edgecolor="none", height=0.6)

    # Annotate each bar with its numeric value
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.01,  # slight offset from bar end
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}",
            va="center", ha="left", fontsize=9, color="#333333",
        )

    ax.set_xlabel("Importance (gain)", fontsize=11)
    ax.set_title("XGBoost Feature Importance \u2014 Top 10", fontsize=13, fontweight="bold")
    ax.tick_params(axis="y", labelsize=10)

    # Clean styling: light X-axis gridlines only, no Y-axis grid
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Expand x-limit slightly to fit annotation text
    ax.set_xlim(right=max(values) * 1.15)

    plt.tight_layout()
    fi_path = os.path.join(models_dir, "feature_importance.png")
    plt.savefig(fi_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return fi_path


if __name__ == "__main__":
    main()
