"""
risk_fusion.py — Risk Score Fusion & Tier Assignment
=====================================================

A PLAIN PYTHON FUNCTION (not a model) that combines the Isolation Forest
anomaly score and Random Forest class probability into a final 0–100
risk score and a Safe/Watch/Warning/Critical tier.

This is NOT a black box — the fusion rule is an explicit, documented
weighted combination with transparent tier thresholds.

Design rationale:
    Two complementary signals are fused:
        1. IF anomaly score (0–1): "how unusual is this window compared
           to normal baseline?" — captures novel/unseen anomaly patterns.
        2. RF subsidence probability (0–1): "how likely is this specific
           window to be actual subsidence vs. a confounder?" — captures
           the learned discriminative features (spectral ratio, cross-node
           correlation, reversal behaviour).

    The weighted combination allows the system to flag subsidence even
    if the RF isn't 100% confident (e.g., early-onset subsidence with
    weak signal) as long as the IF sees it as anomalous.

Tier thresholds (from config.yaml):
    risk_score < 25  → Safe
    25 ≤ score < 50  → Watch
    50 ≤ score < 75  → Warning
    score ≥ 75       → Critical
"""

import numpy as np


def compute_risk_score(if_anomaly_score, rf_subsidence_prob,
                       if_weight=0.3, rf_weight=0.7):
    """Compute fused risk score in [0, 100].

    Formula:
        risk_score = 100 × (if_weight × IF_score + rf_weight × RF_prob)

    Args:
        if_anomaly_score: float or array, [0, 1]. Higher = more anomalous.
        rf_subsidence_prob: float or array, [0, 1]. P(subsidence).
        if_weight: weight for IF component (default 0.3).
        rf_weight: weight for RF component (default 0.7).

    Returns:
        risk_score: float or array in [0, 100].
    """
    if_anomaly_score = np.asarray(if_anomaly_score, dtype=float)
    rf_subsidence_prob = np.asarray(rf_subsidence_prob, dtype=float)

    # Weighted combination, scaled to [0, 100]
    raw = if_weight * if_anomaly_score + rf_weight * rf_subsidence_prob
    risk_score = np.clip(raw * 100.0, 0, 100)
    return risk_score


def assign_risk_tier(risk_score, thresholds=None):
    """Assign a risk tier based on the fused risk score.

    Tier boundaries (configurable):
        risk_score < safe_threshold       → "safe"
        safe ≤ risk_score < watch          → "watch"
        watch ≤ risk_score < warning       → "warning"
        risk_score ≥ warning               → "critical"

    Args:
        risk_score: float or array in [0, 100].
        thresholds: dict with keys "safe", "watch", "warning".

    Returns:
        tier: string or array of tier labels.
    """
    if thresholds is None:
        thresholds = {"safe": 25, "watch": 50, "warning": 75}

    risk_score = np.asarray(risk_score, dtype=float)
    tiers = np.full(risk_score.shape, "safe", dtype=object)

    tiers[risk_score >= thresholds["safe"]] = "watch"
    tiers[risk_score >= thresholds["watch"]] = "warning"
    tiers[risk_score >= thresholds["warning"]] = "critical"

    return tiers


def fuse_predictions(if_anomaly_scores, rf_subsidence_probs, cfg_fusion):
    """Full fusion pipeline: scores → risk_score → tier.

    Args:
        if_anomaly_scores: array of IF anomaly scores, [0, 1].
        rf_subsidence_probs: array of RF subsidence probabilities, [0, 1].
        cfg_fusion: dict with keys "if_weight", "rf_weight", "tier_thresholds".

    Returns:
        (risk_scores, tiers) — arrays of same length as inputs.
    """
    scores = compute_risk_score(
        if_anomaly_scores, rf_subsidence_probs,
        if_weight=cfg_fusion["if_weight"],
        rf_weight=cfg_fusion["rf_weight"],
    )
    tiers = assign_risk_tier(scores, cfg_fusion["tier_thresholds"])
    return scores, tiers
