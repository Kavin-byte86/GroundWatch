"""
feature_engineering.py — Per-Window Feature Computation
========================================================

Computes engineered features from raw sensor readings for each 1-hour
window. These features are what the ML models actually see — not the
raw tilt/vib/strain values directly.

Feature groups:
    1. Tilt features     — current value, rate, rolling statistics
    2. Vibration features — RMS, peak, dominant frequency, spectral ratio
    3. Strain features    — current, rate, cumulative drift
    4. Cross-node features — Pearson correlation with nearest neighbours
    5. Crack proxy        — differential strain rate between adjacent pairs
"""

import numpy as np
import pandas as pd
from scipy import fft as sp_fft


# ──────────────────────────────────────────────────────────────
#  1.  TILT FEATURES
# ──────────────────────────────────────────────────────────────

def compute_tilt_features(df):
    """Compute tilt-derived features for each window.

    Features:
        tilt_current:  current tilt value (degrees)
        tilt_rate_1h:  rate of change over 1 hour (degrees/hr)
        tilt_mean_6h:  rolling mean over 6-hour window
        tilt_std_6h:   rolling std over 6-hour window
        tilt_mean_24h: rolling mean over 24-hour window
        tilt_std_24h:  rolling std over 24-hour window
    """
    out = pd.DataFrame(index=df.index)

    tilt = df["tilt_deg"].astype(float)

    out["tilt_current"] = tilt
    out["tilt_rate_1h"] = tilt.diff().fillna(0)

    # Rolling statistics with min_periods=1 to avoid leading NaNs
    out["tilt_mean_6h"] = tilt.rolling(6, min_periods=1).mean()
    out["tilt_std_6h"] = tilt.rolling(6, min_periods=1).std().fillna(0)
    out["tilt_mean_24h"] = tilt.rolling(24, min_periods=1).mean()
    out["tilt_std_24h"] = tilt.rolling(24, min_periods=1).std().fillna(0)

    return out


# ──────────────────────────────────────────────────────────────
#  2.  VIBRATION FEATURES
# ──────────────────────────────────────────────────────────────

def compute_vibration_features(df):
    """Compute vibration-derived features for each window.

    Features:
        vib_rms:             current RMS value (g)
        vib_peak:            rolling max over 6-hour window
        vib_dom_freq:        dominant frequency (Hz)
        vib_spectral_ratio:  ratio of high-freq energy (>10 Hz) to total
                             This numerically separates blast (high ratio)
                             from creep/subsidence (low ratio).
    """
    out = pd.DataFrame(index=df.index)

    vib = df["vib_rms"].astype(float)
    freq = df["vib_dom_freq_hz"].astype(float)

    out["vib_rms"] = vib
    out["vib_peak"] = vib.rolling(6, min_periods=1).max()
    out["vib_dom_freq"] = freq

    # Spectral energy ratio: blast events have high-freq content (>10 Hz),
    # while subsidence and ambient vibration are low-frequency (<10 Hz).
    # We approximate this from the dominant frequency: high freq → high ratio.
    # A proper FFT-based approach would use raw accelerometer waveforms,
    # but since our data is already aggregated to 1-hour windows, we use
    # the dominant frequency as a proxy for spectral content.
    #
    # Ratio = sigmoid(freq - 10) to map frequency to [0, 1] range
    # where 10 Hz is the empirical boundary between ambient and blast.
    out["vib_spectral_ratio"] = 1.0 / (1.0 + np.exp(-(freq - 10.0) / 3.0))

    return out


# ──────────────────────────────────────────────────────────────
#  3.  STRAIN FEATURES
# ──────────────────────────────────────────────────────────────

def compute_strain_features(df):
    """Compute strain-derived features for each window.

    Features:
        strain_current:   current inter-node strain (mm)
        strain_rate_1h:   rate of change over 1 hour (mm/hr)
        strain_cumulative: cumulative absolute strain drift from start
    """
    out = pd.DataFrame(index=df.index)

    strain = df["strain_mm"].astype(float)

    out["strain_current"] = strain
    out["strain_rate_1h"] = strain.diff().fillna(0)

    # Cumulative drift: how far strain has moved from its initial value.
    # In real subsidence, this monotonically increases; in rain creep it
    # increases then partially reverses.
    out["strain_cumulative"] = (strain - strain.iloc[0]).abs()

    return out


# ──────────────────────────────────────────────────────────────
#  4.  CROSS-NODE CORRELATION
# ──────────────────────────────────────────────────────────────

def compute_cross_node_correlation(all_node_dfs, node_ids, window_size=24):
    """Compute Pearson correlation of tilt trend with nearest neighbours.

    For each node, correlate its 24-hour tilt trend with the tilt trends
    of the 2 nearest neighbours (by node index, which approximates
    spatial proximity since nodes are placed sequentially across the
    trough in generate_dataset.py).

    This feature is a KEY DISCRIMINATOR:
        - TRUE SUBSIDENCE: high correlation (coherent, propagating front)
        - BLAST: low correlation (uncorrelated tilt perturbations)
        - RAIN: low-to-moderate correlation (partially correlated)

    Returns:
        Dict mapping node_id → pd.Series of correlation values.
    """
    correlations = {}

    for i, nid in enumerate(node_ids):
        df_i = all_node_dfs[i]
        tilt_i = df_i["tilt_deg"].astype(float)

        # Find up to 2 nearest neighbours
        neighbours = []
        if i > 0:
            neighbours.append(i - 1)
        if i < len(node_ids) - 1:
            neighbours.append(i + 1)

        if not neighbours:
            correlations[nid] = pd.Series(0.0, index=df_i.index)
            continue

        # Rolling Pearson correlation with each neighbour
        corr_values = []
        for j in neighbours:
            tilt_j = all_node_dfs[j]["tilt_deg"].astype(float)
            # Align lengths
            min_len = min(len(tilt_i), len(tilt_j))
            corr = tilt_i.iloc[:min_len].rolling(window_size, min_periods=6).corr(
                tilt_j.iloc[:min_len]
            ).fillna(0)
            corr_values.append(corr)

        # Mean correlation across neighbours
        avg_corr = pd.concat(corr_values, axis=1).mean(axis=1)
        # Pad to full length if needed
        if len(avg_corr) < len(tilt_i):
            avg_corr = avg_corr.reindex(df_i.index, fill_value=0)
        correlations[nid] = avg_corr

    return correlations


# ──────────────────────────────────────────────────────────────
#  5.  CRACK PROXY
# ──────────────────────────────────────────────────────────────

def compute_crack_proxy_feature(df):
    """Compute crack proxy feature from the raw crack_proxy column.

    crack_proxy in the raw CSV is |d(strain)/dt| — the rate of change of
    inter-node strain. Here we add:
        crack_proxy_cumulative: cumulative sum (monotonic increase during
            active subsidence, stable during blast/rain).
    """
    out = pd.DataFrame(index=df.index)
    cp = df["crack_proxy"].astype(float)
    out["crack_proxy_current"] = cp
    out["crack_proxy_cumulative"] = cp.cumsum()
    return out


# ──────────────────────────────────────────────────────────────
#  6.  TILT REVERSAL INDEX  (NEW — key rain_creep discriminator)
# ──────────────────────────────────────────────────────────────

def compute_tilt_reversal_features(df):
    """Compute tilt reversal index and multi-day rolling features.

    Rain creep's defining characteristic is that tilt REVERSES over days
    as soil dries. True subsidence tilt NEVER reverses — it only
    increases monotonically. This feature captures that difference.

    Features:
        tilt_reversal_72h:   ratio of recent 72h change to total 168h change.
                             Rain → negative (reversal), subsidence → positive.
        tilt_mean_72h:       rolling mean over 3-day window.
        tilt_std_72h:        rolling std over 3-day window.
        tilt_mean_168h:      rolling mean over 7-day window.
        tilt_std_168h:       rolling std over 7-day window.
        tilt_delta_72h:      absolute 3-day tilt change.
        tilt_monotonicity:   fraction of consecutive hours with increasing tilt
                             over a 168h window.  Subsidence → ~0.7–0.9,
                             rain creep → ~0.4–0.5, normal → ~0.5.
    """
    out = pd.DataFrame(index=df.index)
    tilt = df["tilt_deg"].astype(float)

    # --- Multi-day rolling statistics ---
    out["tilt_mean_72h"] = tilt.rolling(72, min_periods=1).mean()
    out["tilt_std_72h"] = tilt.rolling(72, min_periods=1).std().fillna(0)
    out["tilt_mean_168h"] = tilt.rolling(168, min_periods=1).mean()
    out["tilt_std_168h"] = tilt.rolling(168, min_periods=1).std().fillna(0)

    # --- Tilt reversal index ---
    # Compare recent change (last 72h) vs longer-term change (last 168h).
    # If tilt went up and then came back down (rain), the 72h delta will
    # be negative while the 168h delta may still be positive → ratio < 0.
    # Subsidence: both deltas positive, ratio > 0.
    delta_72h = tilt.diff(periods=72).fillna(0)
    delta_168h = tilt.diff(periods=168).fillna(0)
    out["tilt_reversal_72h"] = delta_72h / (delta_168h.abs() + 1e-6)

    # Absolute 3-day tilt change (magnitude of recent movement)
    out["tilt_delta_72h"] = delta_72h.abs()

    # --- Tilt monotonicity ---
    # Fraction of hourly diffs that are positive over a 168h window.
    diffs = tilt.diff().fillna(0)
    out["tilt_monotonicity"] = diffs.rolling(168, min_periods=24).apply(
        lambda x: (x > 0).mean(), raw=True
    ).fillna(0.5)

    return out


# ──────────────────────────────────────────────────────────────
#  7.  VIBRATION PERSISTENCE  (NEW — blast vs sustained events)
# ──────────────────────────────────────────────────────────────

def compute_vibration_persistence(df):
    """Compute features that distinguish transient blasts from sustained vibration.

    Blast vibration is transient (1–2 windows), while other vibration
    sources are more persistent. These features help separate the two.

    Features:
        vib_rms_mean_24h:     24-hour rolling mean of vibration RMS.
        vib_elevated_hours:   rolling count of hours in the last 24 where
                              vib_rms exceeds the baseline by 5×.
    """
    out = pd.DataFrame(index=df.index)
    vib = df["vib_rms"].astype(float)

    out["vib_rms_mean_24h"] = vib.rolling(24, min_periods=1).mean()

    # Count elevated-vibration hours in the last 24 hours.
    # Baseline is ~0.005 g; anything above 0.025 g (5×) is "elevated".
    elevated = (vib > 0.025).astype(float)
    out["vib_elevated_hours"] = elevated.rolling(24, min_periods=1).sum()

    return out


# ──────────────────────────────────────────────────────────────
#  8.  STRAIN REVERSAL  (NEW — mirrors tilt reversal for strain)
# ──────────────────────────────────────────────────────────────

def compute_strain_reversal_features(df):
    """Compute strain reversal and multi-day strain rate features.

    Same logic as tilt reversal: rain-induced strain partially reverses
    as soil dries, while subsidence strain monotonically accumulates.

    Features:
        strain_rate_24h:      24-hour strain rate of change.
        strain_reversal_72h:  ratio of recent 72h strain change to 168h change.
    """
    out = pd.DataFrame(index=df.index)
    strain = df["strain_mm"].astype(float)

    out["strain_rate_24h"] = strain.diff(periods=24).fillna(0)

    delta_72h = strain.diff(periods=72).fillna(0)
    delta_168h = strain.diff(periods=168).fillna(0)
    out["strain_reversal_72h"] = delta_72h / (delta_168h.abs() + 1e-6)

    return out


# ──────────────────────────────────────────────────────────────
#  MASTER FEATURE BUILDER
# ──────────────────────────────────────────────────────────────

def build_features_for_node(df, cross_node_corr=None):
    """Build the full feature vector for a single node's timeseries.

    Args:
        df: raw (filtered) DataFrame for one node.
        cross_node_corr: pd.Series of correlation values (optional).

    Returns:
        DataFrame with all engineered features + metadata columns.
    """
    feat = pd.DataFrame(index=df.index)

    # Metadata columns (carried through for labelling/splitting)
    for col in ["timestamp", "node_id", "lat", "lng"]:
        if col in df.columns:
            feat[col] = df[col].values

    # Original feature groups
    feat = pd.concat([feat, compute_tilt_features(df)], axis=1)
    feat = pd.concat([feat, compute_vibration_features(df)], axis=1)
    feat = pd.concat([feat, compute_strain_features(df)], axis=1)
    feat = pd.concat([feat, compute_crack_proxy_feature(df)], axis=1)

    # NEW feature groups (rain_creep discrimination + blast persistence)
    feat = pd.concat([feat, compute_tilt_reversal_features(df)], axis=1)
    feat = pd.concat([feat, compute_vibration_persistence(df)], axis=1)
    feat = pd.concat([feat, compute_strain_reversal_features(df)], axis=1)

    # Cross-node correlation (if provided)
    if cross_node_corr is not None:
        feat["cross_node_corr"] = cross_node_corr.values[:len(feat)]
    else:
        feat["cross_node_corr"] = 0.0

    # Temperature (as feature — useful for understanding drift)
    if "temp_c" in df.columns:
        feat["temp_c"] = df["temp_c"].astype(float)

    # Gap flag (if present from filtering)
    if "gap_flag" in df.columns:
        feat["gap_flag"] = df["gap_flag"].astype(int)

    return feat
