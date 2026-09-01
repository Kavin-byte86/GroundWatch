"""
generate_dataset.py — Synthetic Dataset Generator (Main Orchestrator)
=====================================================================

Generates 15–20 synthetic mine panels with physics-informed sensor data.

Usage:
    cd D:\\groundwatch-demo\\backend
    python src/data_generation/generate_dataset.py

Outputs:
    data/raw/synthetic/panel_XXX/node_XX.csv   — raw sensor readings
    data/raw/synthetic/panel_XXX/panel_meta.json — panel geometry
    data/labels/window_labels.csv              — ground-truth labels
"""

import os
import sys
import json
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add backend root to path for cross-package imports
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _BACKEND_ROOT)

from src.data_generation.subsidence_profile import PanelGeometry, SubsidenceProfile
from src.data_generation.noise_injection import (
    add_mems_noise, add_vibration_noise, compute_temperature_cycle,
    add_temperature_drift, generate_blast_schedule, apply_blast_transient,
    generate_rain_schedule, apply_rain_creep, apply_packet_dropout,
)


def load_config():
    cfg_path = os.path.join(_BACKEND_ROOT, "configs", "config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def random_panel_geometry(cfg, panel_id, rng):
    """Create a PanelGeometry with parameters randomised within config ranges."""
    geo = cfg["panel_geometry"]
    return PanelGeometry(
        depth_m=rng.uniform(*geo["depth_range_m"]),
        width_m=rng.uniform(*geo["width_range_m"]),
        extraction_thickness_m=rng.uniform(*geo["extraction_thickness_range_m"]),
        angle_of_draw_deg=geo["angle_of_draw_deg"],
        subsidence_factor=rng.uniform(*geo["subsidence_factor_range"]),
        face_advance_rate_m_per_day=rng.uniform(*geo["face_advance_rate_m_per_day"]),
        panel_id=panel_id,
    )


def place_nodes(geom, n_nodes, rng):
    """Position nodes across the subsidence trough.

    Strategy: place nodes at various x-positions (across the trough width)
    and y-positions (along the panel axis). This ensures:
        - 2–3 nodes near trough centre (high displacement)
        - 3–4 nodes near trough edge (moderate displacement, max tilt)
        - 2–3 nodes outside the trough (true "safe zone" controls)
        - 1–2 nodes at the panel boundary (transition zone)

    Returns:
        List of dicts with node_id, x_m, y_m, lat, lng.
    """
    half_w = geom.width / 2.0
    trough_edge = half_w + geom.x_boundary

    # Jharia Coalfield base coordinates (23.76°N, 86.41°E)
    base_lat, base_lng = 23.7644, 86.4131

    nodes = []
    # Distribute x-positions: some inside, some at edge, some outside
    x_positions = []

    # Centre nodes (within ±30% of half-width)
    n_centre = max(2, n_nodes // 4)
    x_positions.extend(rng.uniform(-half_w * 0.3, half_w * 0.3, n_centre))

    # Edge nodes (near ±half_w, the inflection point)
    n_edge = max(2, n_nodes // 3)
    for _ in range(n_edge):
        side = rng.choice([-1, 1])
        x_positions.append(side * rng.uniform(half_w * 0.7, half_w * 1.1))

    # Outside nodes (beyond the angle-of-draw boundary)
    n_outside = max(1, n_nodes - n_centre - n_edge)
    for _ in range(n_outside):
        side = rng.choice([-1, 1])
        x_positions.append(side * rng.uniform(trough_edge * 1.1, trough_edge * 1.5))

    x_positions = x_positions[:n_nodes]

    # Y-positions along the panel axis (for face-advance timing)
    panel_length = geom.width * 1.5  # approximate panel length
    y_positions = rng.uniform(0, panel_length, n_nodes)

    # Node spacing for inter-node calculations (mean spacing)
    node_spacing = np.mean(np.abs(np.diff(sorted(x_positions)))) if n_nodes > 1 else 50.0

    for i in range(n_nodes):
        # Convert x,y offsets to lat/lng (approximate: 1° lat ≈ 111 km)
        lat = base_lat + y_positions[i] / 111000.0
        lng = base_lng + x_positions[i] / (111000.0 * np.cos(np.radians(base_lat)))
        nodes.append({
            "node_id": f"N{i+1:02d}",
            "x_m": float(x_positions[i]),
            "y_m": float(y_positions[i]),
            "lat": round(lat, 6),
            "lng": round(lng, 6),
            "node_spacing_m": float(node_spacing),
        })

    return nodes


def generate_node_timeseries(geom, profile, node_info, n_hours, cfg,
                             has_subsidence, blast_events, rain_events,
                             temp_c, rng):
    """Generate the full sensor timeseries for a single node.

    Returns:
        DataFrame with columns: timestamp, node_id, tilt_deg, vib_rms,
        vib_dom_freq_hz, strain_mm, crack_proxy, temp_c, lat, lng
        Also returns: label_series (str per window), received_mask (bool).
    """
    noise_cfg = cfg["noise"]
    label_cfg = cfg["labels"]
    x = node_info["x_m"]
    y = node_info["y_m"]
    spacing = node_info["node_spacing_m"]

    t_hours = np.arange(n_hours, dtype=float)

    # ── BASELINE SIGNALS (from physics model or zero) ──
    if has_subsidence:
        # Compute onset time for this node's y-position
        tp_cfg = cfg["time_profile"]
        onset_day = rng.uniform(*tp_cfg["onset_range_day"])
        t_onset = float(np.ravel(profile.onset_time_for_position(
            np.array([y]), sim_onset_day=onset_day))[0])

        # --- VECTORIZED computation over all timesteps at once ---
        # Spatial profile at this x-position (scalar, time-independent)
        s_x = float(profile.spatial_profile(np.array([x]))[0])

        # Time profile for all hours (vectorized)
        f_t = profile.time_profile(t_hours, t_onset)

        # Vertical displacement (mm) = S(x) × f(t)
        v_disp = s_x * f_t

        # Tilt: spatial derivative of displacement profile at x
        dx = max(node_info["node_spacing_m"], 1.0)
        s_plus = float(profile.spatial_profile(np.array([x + dx / 2]))[0])
        s_minus = float(profile.spatial_profile(np.array([x - dx / 2]))[0])
        slope_mm_per_m = (s_plus - s_minus) / dx
        tilt_rad = np.arctan(slope_mm_per_m / 1000.0)
        # Tilt scales with time profile (displacement grows over time)
        tilt_base = np.degrees(tilt_rad) * f_t

        # Horizontal displacement for strain calculation
        h_ratio = profile.h_disp_ratio
        # Direction sign from spatial derivative
        s_x_plus = float(profile.spatial_profile(np.array([x + 1.0]))[0])
        s_x_minus = float(profile.spatial_profile(np.array([x - 1.0]))[0])
        ds_dx_sign = -np.sign(s_x_plus - s_x_minus)
        h_disp_here = ds_dx_sign * h_ratio * v_disp

        s_x_n = float(profile.spatial_profile(np.array([x + spacing]))[0])
        s_x_n_plus = float(profile.spatial_profile(np.array([x + spacing + 1.0]))[0])
        s_x_n_minus = float(profile.spatial_profile(np.array([x + spacing - 1.0]))[0])
        ds_dx_sign_n = -np.sign(s_x_n_plus - s_x_n_minus)
        h_disp_neighbor = ds_dx_sign_n * h_ratio * (s_x_n * f_t)

        strain_base = h_disp_here - h_disp_neighbor  # inter-node strain (mm)

        # Subsidence-related vibration: very low frequency, small amplitude
        # proportional to displacement rate
        v_disp_rate = np.gradient(v_disp)  # mm/hr
        vib_subsidence = np.abs(v_disp_rate) * 0.001  # very small contribution

    else:
        v_disp = np.zeros(n_hours)
        tilt_base = np.zeros(n_hours)
        strain_base = np.zeros(n_hours)
        h_disp_here = np.zeros(n_hours)
        vib_subsidence = np.zeros(n_hours)
        t_onset = float("inf")

    # ── VIBRATION BASELINE ──
    vib_rms = np.full(n_hours, noise_cfg["baseline_vib_rms"])
    vib_rms += vib_subsidence
    vib_freq = rng.uniform(
        noise_cfg["baseline_vib_freq_hz"][0],
        noise_cfg["baseline_vib_freq_hz"][1],
        size=n_hours,
    )

    # ── APPLY NOISE LAYERS ──
    tilt_deg = tilt_base.copy()
    strain_mm = strain_base.copy()

    # 1. MEMS noise
    tilt_deg = add_mems_noise(tilt_deg, noise_cfg["mems_noise_sigma_deg"], rng)
    vib_rms = add_vibration_noise(vib_rms, sigma_g=0.002, rng=rng)

    # 2. Temperature drift
    tilt_deg = add_temperature_drift(tilt_deg, temp_c, noise_cfg["temp_drift_slope_per_c"])

    # 3. Blast transients
    vib_rms_copy = vib_rms.copy()
    vib_freq_copy = vib_freq.copy()
    tilt_copy = tilt_deg.copy()
    strain_copy = strain_mm.copy()
    vib_rms, vib_freq, tilt_deg, strain_mm, blast_mask = apply_blast_transient(
        vib_rms_copy, vib_freq_copy, tilt_copy, strain_copy,
        blast_events, noise_cfg["blast"], rng
    )

    # 4. Rain creep (only for affected nodes)
    rain_cfg = noise_cfg["rain"]
    is_affected = rng.random() < rng.uniform(*rain_cfg["affected_node_fraction"])
    tilt_deg, strain_mm, rain_mask = apply_rain_creep(
        tilt_deg, strain_mm, rain_events, rain_cfg, is_affected, rng
    )

    # 5. Small random walk on strain (sensor drift over time)
    strain_drift = np.cumsum(rng.normal(0, 0.0005, n_hours))
    strain_mm += strain_drift

    # ── CRACK PROXY ──
    # Derived from inter-node strain differential (per design decision —
    # we do NOT simulate a dedicated crack sensor).
    # crack_proxy = rate of change of differential strain.
    crack_proxy = np.abs(np.gradient(strain_mm))

    # ── LABELS ──
    labels = np.full(n_hours, "normal", dtype=object)

    # Subsidence labels (highest priority)
    if has_subsidence:
        for h in range(n_hours):
            disp = v_disp[h]
            if disp >= label_cfg["subsidence_critical_mm"]:
                labels[h] = "subsidence_critical"
            elif disp >= label_cfg["subsidence_warning_mm"]:
                labels[h] = "subsidence_warning"
            elif disp >= label_cfg["subsidence_watch_mm"]:
                labels[h] = "subsidence_watch"

    # Confounder labels (only override "normal" — subsidence takes priority)
    for h in range(n_hours):
        if labels[h] == "normal":
            if blast_mask[h]:
                labels[h] = "blast_transient"
            elif rain_mask[h]:
                labels[h] = "rain_creep"

    # ── PACKET DROPOUT ──
    drop_rate = rng.uniform(*noise_cfg["packet_drop_rate"])
    received = apply_packet_dropout(n_hours, drop_rate, rng)

    # ── BUILD OUTPUT DATAFRAME ──
    sim_start = datetime(2026, 3, 1)
    timestamps = [sim_start + timedelta(hours=int(h)) for h in t_hours]

    df = pd.DataFrame({
        "timestamp": timestamps,
        "node_id": node_info["node_id"],
        "tilt_deg": np.round(tilt_deg, 4),
        "vib_rms": np.round(vib_rms, 6),
        "vib_dom_freq_hz": np.round(vib_freq, 2),
        "strain_mm": np.round(strain_mm, 4),
        "crack_proxy": np.round(crack_proxy, 6),
        "temp_c": np.round(temp_c, 1),
        "lat": node_info["lat"],
        "lng": node_info["lng"],
    })

    # Apply packet dropout: set dropped rows to NaN (realistic gaps)
    drop_cols = ["tilt_deg", "vib_rms", "vib_dom_freq_hz", "strain_mm",
                 "crack_proxy", "temp_c"]
    df.loc[~received, drop_cols] = np.nan

    # Labels for dropped windows are still "normal" (no data ≠ anomaly)
    # but we keep them for completeness; the preprocessing pipeline will
    # handle gaps via interpolation or flagging.

    return df, labels, received


def generate_panel(panel_id, panel_idx, cfg, rng, subsidence_flags):
    """Generate all node timeseries for a single panel."""
    ds_cfg = cfg["dataset"]
    n_hours = ds_cfg["simulation_days"] * 24

    # Randomise panel geometry
    geom = random_panel_geometry(cfg, panel_id, rng)

    # Decide if this panel has active subsidence
    has_subsidence = subsidence_flags[panel_idx]

    # Create subsidence profile
    tp = cfg["time_profile"]
    h_ratio = rng.uniform(*cfg["horizontal_displacement"]["vertical_ratio"])
    profile = SubsidenceProfile(
        geom,
        active_phase_fraction=rng.uniform(*tp["active_phase_fraction"]),
        active_phase_days=rng.uniform(*tp["active_phase_days"]),
        residual_tau_days=rng.uniform(*tp["residual_tau_days"]),
        h_disp_ratio=h_ratio,
    )

    # Place nodes
    n_nodes = rng.integers(*ds_cfg["nodes_per_panel"])
    nodes = place_nodes(geom, n_nodes, rng)

    # Panel-level events (shared across all nodes)
    blast_events = generate_blast_schedule(
        n_hours, cfg["noise"]["blast"]["per_day_range"], rng
    )
    rain_events = generate_rain_schedule(
        n_hours,
        cfg["noise"]["rain"]["events_per_month"],
        cfg["noise"]["rain"]["duration_hours"],
        cfg["noise"]["rain"]["decay_days"],
        rng,
    )
    temp_c = compute_temperature_cycle(
        n_hours,
        *cfg["noise"]["temp_range_c"],
        rng,
    )

    # Generate per-node data
    node_dfs = []
    label_rows = []

    for node_info in nodes:
        df, labels, received = generate_node_timeseries(
            geom, profile, node_info, n_hours, cfg,
            has_subsidence, blast_events, rain_events, temp_c, rng,
        )
        node_dfs.append(df)

        # Collect labels
        sim_start = datetime(2026, 3, 1)
        for h in range(n_hours):
            label_rows.append({
                "panel_id": panel_id,
                "node_id": node_info["node_id"],
                "window_timestamp": (sim_start + timedelta(hours=h)).isoformat(),
                "label": labels[h],
            })

    return geom, node_dfs, nodes, label_rows


def main():
    cfg = load_config()
    rng = np.random.default_rng(cfg["dataset"]["random_seed"])

    n_panels = cfg["dataset"]["n_panels"]
    all_label_rows = []

    # Pre-compute which panels have subsidence, interleaving so that
    # subsidence panels are distributed across sorted panel IDs.
    # This ensures val/test splits (which take later panel IDs) also
    # contain subsidence panels for meaningful evaluation.
    n_sub = cfg["dataset"]["panels_with_subsidence"]
    subsidence_flags = [False] * n_panels
    # Distribute subsidence panels evenly: every N/n_sub-th panel
    step = max(1, n_panels / n_sub)
    for j in range(n_sub):
        idx = int(j * step) % n_panels
        subsidence_flags[idx] = True
    # Ensure exactly n_sub panels have subsidence
    current_count = sum(subsidence_flags)
    for j in range(n_panels):
        if current_count >= n_sub:
            break
        if not subsidence_flags[j]:
            subsidence_flags[j] = True
            current_count += 1

    print(f"Generating {n_panels} synthetic panels...")
    print(f"  Panels with active subsidence: {sum(subsidence_flags)}")
    sub_ids = [f"panel_{i+1:03d}" for i, f in enumerate(subsidence_flags) if f]
    print(f"  Subsidence panels: {sub_ids}")
    print(f"  Simulation length: {cfg['dataset']['simulation_days']} days")
    print()

    for i in range(n_panels):
        panel_id = f"panel_{i+1:03d}"
        has_sub = subsidence_flags[i]
        print(f"  [{i+1}/{n_panels}] Generating {panel_id} "
              f"({'SUBSIDENCE' if has_sub else 'stable'})...", end=" ", flush=True)

        geom, node_dfs, nodes, label_rows = generate_panel(
            panel_id, i, cfg, rng, subsidence_flags
        )

        # ── Write output files ──
        panel_dir = os.path.join(_BACKEND_ROOT, "data", "raw", "synthetic", panel_id)
        os.makedirs(panel_dir, exist_ok=True)

        # Node CSVs
        for df, node_info in zip(node_dfs, nodes):
            csv_path = os.path.join(panel_dir, f"{node_info['node_id'].lower()}.csv")
            df.to_csv(csv_path, index=False)

        # Panel metadata
        meta = geom.to_dict()
        meta["nodes"] = [
            {"node_id": n["node_id"], "x_m": round(n["x_m"], 1),
             "y_m": round(n["y_m"], 1), "lat": n["lat"], "lng": n["lng"]}
            for n in nodes
        ]
        meta["has_subsidence"] = subsidence_flags[i]
        meta_path = os.path.join(panel_dir, "panel_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        all_label_rows.extend(label_rows)
        n_nodes = len(node_dfs)
        print(f"{n_nodes} nodes, S_max={geom.s_max_mm:.0f}mm, "
              f"depth={geom.depth:.0f}m, width={geom.width:.0f}m")

    # ── Write labels ──
    labels_dir = os.path.join(_BACKEND_ROOT, "data", "labels")
    os.makedirs(labels_dir, exist_ok=True)
    labels_df = pd.DataFrame(all_label_rows)
    labels_path = os.path.join(labels_dir, "window_labels.csv")
    labels_df.to_csv(labels_path, index=False)

    # ── Print distribution summary ──
    print("\n" + "=" * 60)
    print("LABEL DISTRIBUTION")
    print("=" * 60)
    dist = labels_df["label"].value_counts()
    total = len(labels_df)
    for label, count in dist.items():
        pct = 100 * count / total
        print(f"  {label:25s}  {count:>8d}  ({pct:5.1f}%)")
    print(f"  {'TOTAL':25s}  {total:>8d}")
    print()
    print(f"Dataset written to: {os.path.join(_BACKEND_ROOT, 'data')}")
    print(f"Labels written to:  {labels_path}")


if __name__ == "__main__":
    main()
