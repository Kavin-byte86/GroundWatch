"""
noise_injection.py — Physics-Grounded Noise & Confounder Injection
===================================================================

Every noise term traces to a named, real source — no magic numbers.

Noise sources implemented:
    1. MEMS sensor noise floor     (MPU-6050 datasheet)
    2. Temperature drift           (Wi-GIM field calibration)
    3. Blast-transient events      (DGMS India vibration data)
    4. Rainfall-induced soil creep (geotechnical monitoring literature)
    5. Packet dropout              (LoRa mesh packet loss)

See config.yaml for parameter values and citations.
"""

import numpy as np


# ──────────────────────────────────────────────────────────────
#  1.  MEMS SENSOR NOISE FLOOR
# ──────────────────────────────────────────────────────────────

def add_mems_noise(signal, sigma_deg=0.15, rng=None):
    """Add Gaussian noise matching the MPU-6050 MEMS accelerometer floor.

    Source: InvenSense MPU-6050 Product Specification, Rev 3.4
        - Accelerometer noise density: 400 µg/√Hz
        - At bandwidth 260 Hz: total RMS noise ≈ 6.45 mg ≈ 0.13° tilt equiv.
        - Under field conditions (vibration, thermal), effective noise
          broadens to approximately ±0.1–0.2° (we use 0.15° as typical).

    Args:
        signal: 1-D array of tilt readings (degrees).
        sigma_deg: noise standard deviation (degrees). Default 0.15°.
        rng: numpy random Generator.
    Returns:
        Noisy signal (same shape).
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, sigma_deg, size=signal.shape)
    return signal + noise


def add_vibration_noise(signal, sigma_g=0.002, rng=None):
    """Add baseline vibration noise to vib_rms readings.

    Ambient ground vibration in mining areas: ~0.002–0.01 g RMS.
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = np.abs(rng.normal(0.0, sigma_g, size=signal.shape))
    return signal + noise


# ──────────────────────────────────────────────────────────────
#  2.  TEMPERATURE DRIFT
# ──────────────────────────────────────────────────────────────

def compute_temperature_cycle(n_hours, temp_min=25.0, temp_max=40.0, rng=None):
    """Simulate diurnal temperature cycle for Indian coalfield conditions.

    Models a sinusoidal daily cycle with ±2°C random day-to-day variation.

    Returns:
        temp_c: array of shape (n_hours,) with temperature in °C.
    """
    if rng is None:
        rng = np.random.default_rng()
    t = np.arange(n_hours, dtype=float)
    temp_mid = (temp_min + temp_max) / 2.0
    temp_amp = (temp_max - temp_min) / 2.0

    # Sinusoidal daily cycle: peak at 14:00 (hour 14)
    daily_cycle = temp_mid + temp_amp * np.sin(2 * np.pi * (t - 6) / 24.0)

    # Day-to-day random variation (slow drift)
    n_days = int(np.ceil(n_hours / 24))
    daily_noise = rng.normal(0, 2.0, size=n_days)
    hourly_noise = np.repeat(daily_noise, 24)[:n_hours]

    return daily_cycle + hourly_noise


def add_temperature_drift(tilt_signal, temp_c, slope_per_c=0.0029):
    """Apply temperature-dependent systematic drift to tilt readings.

    Source: Wi-GIM (Wireless Geotechnical Instrumentation & Monitoring)
            field calibration study — measured drift slope of 0.0029°/°C
            on MEMS tilt sensors deployed in open-pit/underground mine
            monitoring installations.

    The drift is SYSTEMATIC (not random) — it correlates with temperature,
    which means it can be partially compensated during preprocessing if
    the temperature is measured. This is the real-world-facing part of
    the pipeline: the same correction is applied in filtering.py.

    Args:
        tilt_signal: 1-D array of tilt values (degrees).
        temp_c: 1-D array of temperature values (°C), same length.
        slope_per_c: drift coefficient (degrees per °C).
    Returns:
        Tilt signal with drift added.
    """
    # Reference temperature: median of range (drift is relative)
    temp_ref = np.median(temp_c)
    drift = slope_per_c * (temp_c - temp_ref)
    return tilt_signal + drift


# ──────────────────────────────────────────────────────────────
#  3.  BLAST-TRANSIENT EVENTS
# ──────────────────────────────────────────────────────────────

def generate_blast_schedule(n_hours, per_day_range=(1, 3), rng=None):
    """Generate a schedule of blast event timestamps.

    Mine blasting is routine and roughly daily-scheduled. Each blast
    affects 1–2 consecutive hour-windows (the blast itself lasts 2–10
    seconds, but the sensor sampling window captures the entire event).

    Returns:
        List of (start_hour, duration_windows) tuples.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_days = int(np.ceil(n_hours / 24))
    events = []
    for day in range(n_days):
        n_blasts = rng.integers(per_day_range[0], per_day_range[1] + 1)
        for _ in range(n_blasts):
            # Blasts typically occur during shift hours (06:00–18:00)
            blast_hour = day * 24 + rng.integers(6, 18)
            if blast_hour < n_hours:
                duration = rng.integers(1, 3)  # 1–2 windows
                events.append((int(blast_hour), int(duration)))
    return events


def apply_blast_transient(vib_rms, vib_freq, tilt_deg, strain_mm,
                          blast_events, cfg_blast, rng=None):
    """Inject blast-transient signatures into sensor readings.

    Physics:
        During a blast, the ground experiences a short, high-amplitude
        vibration dominated by high-frequency content (15–50 Hz blast
        wave, per DGMS India ground vibration standards).

        CRITICALLY: the accompanying tilt/strain values show only a SMALL,
        NON-PERSISTENT perturbation (shock, not sustained movement) that
        DECAYS BACK TO BASELINE within 1 window. This is what makes blast
        events genuine "false positives" — they show high vibration but
        DO NOT indicate structural ground movement.

    Returns:
        Modified (vib_rms, vib_freq, tilt_deg, strain_mm) arrays and
        a boolean mask of blast-affected windows.
    """
    if rng is None:
        rng = np.random.default_rng()

    blast_mask = np.zeros(len(vib_rms), dtype=bool)

    for start_hour, duration in blast_events:
        end_hour = min(start_hour + duration, len(vib_rms))
        if start_hour >= len(vib_rms):
            continue
        blast_mask[start_hour:end_hour] = True

        # Vibration: spike to 0.5–1.0 g RMS (vs ~0.005 g baseline)
        vib_rms[start_hour:end_hour] = rng.uniform(
            cfg_blast["vib_rms_range"][0],
            cfg_blast["vib_rms_range"][1],
            size=end_hour - start_hour
        )

        # Dominant frequency shifts to high-freq blast content
        vib_freq[start_hour:end_hour] = rng.uniform(
            cfg_blast["vib_freq_hz"][0],
            cfg_blast["vib_freq_hz"][1],
            size=end_hour - start_hour
        )

        # Tilt: SMALL, TRANSIENT perturbation — NOT persistent
        # This is the key discriminator from real subsidence tilt
        tilt_perturbation = rng.uniform(
            cfg_blast["tilt_perturbation_deg"][0],
            cfg_blast["tilt_perturbation_deg"][1]
        ) * rng.choice([-1, 1])
        tilt_deg[start_hour:end_hour] += tilt_perturbation
        # Perturbation does NOT persist — no cumulative effect

        # Strain: similarly small and transient
        strain_perturbation = rng.uniform(
            cfg_blast["strain_perturbation_mm"][0],
            cfg_blast["strain_perturbation_mm"][1]
        ) * rng.choice([-1, 1])
        strain_mm[start_hour:end_hour] += strain_perturbation

    return vib_rms, vib_freq, tilt_deg, strain_mm, blast_mask


# ──────────────────────────────────────────────────────────────
#  4.  RAINFALL-INDUCED SOIL CREEP
# ──────────────────────────────────────────────────────────────

def generate_rain_schedule(n_hours, events_per_month=(2, 4),
                           duration_hours_range=(12, 48),
                           decay_days_range=(3, 7), rng=None):
    """Generate a schedule of rainfall events.

    Returns:
        List of dicts: {start_hour, rain_duration_hours, decay_hours}.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_months = max(1, int(np.ceil(n_hours / (30 * 24))))
    events = []
    for month in range(n_months):
        n_events = rng.integers(events_per_month[0], events_per_month[1] + 1)
        for _ in range(n_events):
            start = month * 30 * 24 + rng.integers(0, 30 * 24)
            if start >= n_hours:
                continue
            rain_dur = rng.integers(duration_hours_range[0],
                                     duration_hours_range[1] + 1)
            decay_days = rng.uniform(decay_days_range[0], decay_days_range[1])
            events.append({
                "start_hour": int(start),
                "rain_duration_hours": int(rain_dur),
                "decay_hours": int(decay_days * 24),
            })
    return events


def apply_rain_creep(tilt_deg, strain_mm, rain_events, cfg_rain,
                     is_affected=True, rng=None):
    """Inject rainfall-induced soil creep into tilt and strain signals.

    Physics:
        After rainfall, shallow soil layers absorb water and undergo small
        volumetric expansion / creep. This produces:
            - A small tilt blip (0.01–0.05°) much smaller than real
              subsidence tilt.
            - CRITICALLY: the tilt PARTIALLY REVERSES over the following
              days as the soil dries — UNLIKE true subsidence tilt, which
              NEVER reverses. This reversal behaviour is the actual
              discriminating feature that separates rain creep from
              subsidence in the ML model.

        Implementation: during rain, tilt ramps up following a half-sine.
        During the decay period, tilt decays exponentially back toward
        baseline, reaching ~20–40% of peak (partial reversal — some
        residual permanent deformation from soil compaction remains).

    Returns:
        Modified (tilt_deg, strain_mm) and a boolean mask of rain-affected windows.
    """
    if rng is None:
        rng = np.random.default_rng()

    rain_mask = np.zeros(len(tilt_deg), dtype=bool)

    if not is_affected:
        return tilt_deg, strain_mm, rain_mask

    for event in rain_events:
        start = event["start_hour"]
        rain_dur = event["rain_duration_hours"]
        decay_dur = event["decay_hours"]
        total_dur = rain_dur + decay_dur
        end = min(start + total_dur, len(tilt_deg))

        if start >= len(tilt_deg):
            continue

        # Peak tilt magnitude for this event
        peak_tilt = rng.uniform(cfg_rain["tilt_range_deg"][0],
                                cfg_rain["tilt_range_deg"][1])
        direction = rng.choice([-1, 1])

        # Peak strain magnitude
        peak_strain = rng.uniform(cfg_rain["strain_range_mm"][0],
                                  cfg_rain["strain_range_mm"][1])

        # Build the creep envelope: ramp up, then decay with partial reversal
        for h in range(start, end):
            if h >= len(tilt_deg):
                break
            dt = h - start
            if dt < rain_dur:
                # Ramp-up phase: half-sine rise
                phase = np.sin(np.pi * dt / (2 * rain_dur))
                tilt_deg[h] += direction * peak_tilt * phase
                strain_mm[h] += direction * peak_strain * phase
            else:
                # Decay phase: exponential reversal
                # The tilt decays back toward baseline but retains 20–40%
                # as permanent soil compaction (not full reversal).
                dt_decay = dt - rain_dur
                retention = rng.uniform(0.2, 0.4) if dt_decay == 0 else 0.3
                tau = decay_dur / 3.0  # exponential decay constant
                decay_factor = retention + (1 - retention) * np.exp(-dt_decay / max(tau, 1))
                tilt_deg[h] += direction * peak_tilt * decay_factor
                strain_mm[h] += direction * peak_strain * decay_factor

            rain_mask[h] = True

    return tilt_deg, strain_mm, rain_mask


# ──────────────────────────────────────────────────────────────
#  5.  PACKET DROPOUT (LoRa mesh simulation)
# ──────────────────────────────────────────────────────────────

def apply_packet_dropout(n_hours, drop_rate=0.10, rng=None):
    """Generate a boolean mask of successfully received packets.

    Simulates LoRa mesh packet loss at a configurable rate (5–15%).
    Dropped windows appear as NaN/gaps in the output CSV — NOT as
    perfectly uniform time series.

    Returns:
        received_mask: boolean array where True = packet received.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Burst dropout model: packets tend to drop in short bursts
    # (radio interference, temporary obstruction) rather than i.i.d.
    received = np.ones(n_hours, dtype=bool)
    h = 0
    while h < n_hours:
        if rng.random() < drop_rate:
            # Burst length: 1–5 consecutive dropped packets
            burst_len = rng.integers(1, 6)
            received[h:min(h + burst_len, n_hours)] = False
            h += burst_len
        else:
            h += 1

    return received


# ──────────────────────────────────────────────────────────────
#  6.  CROSS-NODE CORRELATION (handled at generation level)
# ──────────────────────────────────────────────────────────────
# Cross-node correlation is NOT injected as post-hoc noise — it
# arises NATURALLY from the physics model in subsidence_profile.py:
#
#   - TRUE SUBSIDENCE: all nodes within the trough share the same
#     PanelGeometry and SubsidenceProfile. Their tilt/strain values
#     are correlated because they're computed from the same underlying
#     displacement field. The face-advance propagation front creates
#     time-lagged but coherent signals across adjacent nodes.
#
#   - BLAST: vibration spike hits all nodes on the panel simultaneously,
#     but the tilt PERTURBATION is sampled independently per node
#     (random direction/magnitude) — so blast tilt is UNCORRELATED.
#
#   - RAIN CREEP: affects a random SUBSET of nodes with independent
#     peak magnitudes — partially correlated at best, not coherent.
#
# This difference in cross-node correlation structure is a KEY
# discriminating feature exploited by the cross-node Pearson
# correlation feature in feature_engineering.py.
