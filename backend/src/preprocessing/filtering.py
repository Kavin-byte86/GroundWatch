"""
filtering.py — Signal Filtering & Data Cleaning
=================================================

IMPORTANT: This pipeline is designed to work identically on REAL field
data later — it is NOT synthetic-data-specific. Every operation here
mirrors what would be done on actual sensor readings from deployed nodes.

Operations:
    1. Resample to uniform time grid (interpolate small gaps, flag large)
    2. Drop physically impossible values
    3. Temperature-drift compensation
"""

import numpy as np
import pandas as pd


def resample_to_uniform_grid(df, interval_hours=1, max_gap_hours=6):
    """Resample a node DataFrame to a uniform hourly time grid.

    Small gaps (≤ max_gap_hours) are filled via linear interpolation.
    Large gaps are left as NaN and flagged in a 'gap_flag' column.

    This is necessary because LoRa packet dropout creates irregular
    gaps in the raw timeseries. Downstream feature engineering (rolling
    windows, FFT) requires uniform sampling.

    Args:
        df: DataFrame with 'timestamp' column (ISO format or datetime).
        interval_hours: target sampling interval.
        max_gap_hours: gaps longer than this are flagged, not interpolated.

    Returns:
        Resampled DataFrame with 'gap_flag' column added.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()

    # Create uniform time grid
    freq = f"{interval_hours}h"
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df = df.reindex(full_index)
    df.index.name = "timestamp"

    # Identify large gaps before interpolation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    is_nan = df[numeric_cols].isna().all(axis=1)
    gap_flag = np.zeros(len(df), dtype=bool)

    # Find consecutive NaN runs exceeding max_gap_hours
    nan_runs = is_nan.astype(int).groupby((~is_nan).cumsum()).cumsum()
    gap_flag = (nan_runs > max_gap_hours).values

    # Interpolate small gaps (linear)
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit=max_gap_hours)

    # Forward/back fill non-numeric columns (node_id, lat, lng)
    non_numeric = df.select_dtypes(exclude=[np.number]).columns
    df[non_numeric] = df[non_numeric].ffill().bfill()

    df["gap_flag"] = gap_flag
    df = df.reset_index()
    return df


def remove_invalid_values(df):
    """Clamp or drop physically impossible sensor readings.

    Physical constraints:
        - tilt_deg: must be in [−90, +90] (sensor can't read beyond ±90°)
        - vib_rms: must be ≥ 0 (RMS is always non-negative)
        - strain_mm: no hard physical limit, but values beyond ±100 mm
          within a single window are sensor faults
        - temp_c: must be in [−40, +80]°C (sensor operating range)
        - vib_dom_freq_hz: must be in [0, 500] Hz (Nyquist limit for
          typical MEMS accelerometers)
    """
    df = df.copy()

    # Clamp tilt to ±90° — values beyond this indicate sensor fault
    if "tilt_deg" in df.columns:
        df["tilt_deg"] = df["tilt_deg"].clip(-90, 90)

    # Zero-floor vibration RMS
    if "vib_rms" in df.columns:
        df["vib_rms"] = df["vib_rms"].clip(lower=0)

    # Clamp strain
    if "strain_mm" in df.columns:
        df["strain_mm"] = df["strain_mm"].clip(-100, 100)

    # Clamp temperature
    if "temp_c" in df.columns:
        df["temp_c"] = df["temp_c"].clip(-40, 80)

    # Clamp frequency
    if "vib_dom_freq_hz" in df.columns:
        df["vib_dom_freq_hz"] = df["vib_dom_freq_hz"].clip(0, 500)

    return df


def compensate_temperature_drift(df, slope_per_c=0.0029):
    """Subtract temperature-induced systematic drift from tilt readings.

    This is the real-world counterpart to the drift injected in
    noise_injection.py. In a deployed system, each sensor measures its
    own temperature, and this correction is applied in real-time.

    Formula:
        tilt_corrected = tilt_raw − slope × (temp − temp_reference)

    where slope = 0.0029°/°C (Wi-GIM field calibration) and
    temp_reference = median temperature over the dataset.

    Args:
        df: DataFrame with 'tilt_deg' and 'temp_c' columns.
        slope_per_c: drift coefficient (degrees per °C).

    Returns:
        DataFrame with corrected tilt values.
    """
    df = df.copy()
    if "tilt_deg" not in df.columns or "temp_c" not in df.columns:
        return df

    temp_ref = df["temp_c"].median()
    drift = slope_per_c * (df["temp_c"] - temp_ref)
    df["tilt_deg"] = df["tilt_deg"] - drift

    return df
