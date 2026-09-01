"""
server.py -- GroundWatch Live Simulation Server
=================================================

FastAPI + WebSocket server that turns the offline physics engine into a
live, real-time mine subsidence simulator. Loads the trained models
(Isolation Forest + XGBoost) at startup, runs them on each tick, and
streams risk-scored node updates to the React dashboard.

Run from backend/:
    python -m uvicorn src.server:app --host 0.0.0.0 --port 8000

Or with auto-reload during development:
    python -m uvicorn src.server:app --reload --port 8000
"""

import os
import sys
import json
import asyncio
import math
import time
import warnings
import logging
from datetime import datetime, timezone
from collections import deque
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import yaml
import joblib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Path setup — so we can import existing src.* modules
# ---------------------------------------------------------------------------
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from src.data_generation.subsidence_profile import PanelGeometry, SubsidenceProfile
from src.data_generation.noise_injection import (
    add_mems_noise, add_temperature_drift,
)
from src.models.xgboost_classifier import XGBoostModel, collapse_labels
from src.models.risk_fusion import compute_risk_score, assign_risk_tier
from src.preprocessing.pipeline import FEATURE_COLS

# Suppress sklearn feature-names warning (we pass numpy arrays but scaler was
# fit on a DataFrame — the values and order are correct, just no column names)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("groundwatch")


# ===================================================================
#  CONFIG
# ===================================================================
def load_config():
    cfg_path = os.path.join(_BACKEND_ROOT, "configs", "config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


# ===================================================================
#  NODE TOPOLOGY — Jharia Coalfield, Dhanbad (23.74 N, 86.41 E)
# ===================================================================
# Geographic conversion at Jharia's latitude:
#   1 deg latitude  ~ 111,000 m
#   1 deg longitude ~ 111,000 m * cos(23.74 deg) ~ 101,600 m
# So: 100 m north  = 100 / 111000 = 0.000901 deg lat
#     100 m east   = 100 / 101600 = 0.000984 deg lng

JHARIA_CENTER_LAT = 23.7400
JHARIA_CENTER_LNG = 86.4100
M_PER_DEG_LAT = 111_000.0
M_PER_DEG_LNG = 111_000.0 * math.cos(math.radians(JHARIA_CENTER_LAT))  # ~101,600

def _m_to_lat(m):
    return m / M_PER_DEG_LAT

def _m_to_lng(m):
    return m / M_PER_DEG_LNG


# 12 nodes in a cross pattern across the simulated trough.
# x_m = position in metres from trough centre (physics model input).
# Nodes at x_m ~ 0 sit above the panel centre (highest subsidence).
# Nodes at x_m > panel_half_width + x_boundary sit outside the
# angle-of-draw boundary (should be consistently Safe).
NODE_TOPOLOGY = [
    # ---- TROUGH CENTRE (high risk) ----
    {"node_id": "N01", "x_m":    0, "row": 0, "col": 0},
    {"node_id": "N02", "x_m":   25, "row": 0, "col": 1},
    {"node_id": "N03", "x_m":  -30, "row": 1, "col": 0},
    # ---- TROUGH SLOPE (medium risk — steepest tilt gradient) ----
    {"node_id": "N04", "x_m":   80, "row": 0, "col": 2},
    {"node_id": "N05", "x_m":  -85, "row": 1, "col": -1},
    {"node_id": "N06", "x_m":  120, "row": -1, "col": 2},
    # ---- TROUGH EDGE (transitional) ----
    {"node_id": "N07", "x_m":  170, "row": 0, "col": 3},
    {"node_id": "N08", "x_m": -175, "row": 1, "col": -2},
    {"node_id": "N09", "x_m":  200, "row": -1, "col": 3},
    # ---- OUTSIDE TROUGH — control nodes (safe) ----
    {"node_id": "N10", "x_m":  350, "row": 0, "col": 5},
    {"node_id": "N11", "x_m": -360, "row": 1, "col": -4},
    {"node_id": "N12", "x_m":  420, "row": -1, "col": 6},
]

def build_node_coords():
    """Assign lat/lng to each node based on row/col grid offsets."""
    spacing_m = 100  # ~100 m between grid rows/cols
    nodes = {}
    for n in NODE_TOPOLOGY:
        lat = JHARIA_CENTER_LAT + _m_to_lat(n["row"] * spacing_m)
        lng = JHARIA_CENTER_LNG + _m_to_lng(n["col"] * spacing_m)
        nodes[n["node_id"]] = {
            "lat": round(lat, 6),
            "lng": round(lng, 6),
            "x_m": n["x_m"],
        }
    return nodes


# ===================================================================
#  SIMULATION STATE — per-node
# ===================================================================
HISTORY_LEN = 30  # rolling window of "simulated hours" to keep

class NodeState:
    """Mutable state for a single sensor node during simulation."""

    def __init__(self, node_id, lat, lng, x_m):
        self.node_id = node_id
        self.lat = lat
        self.lng = lng
        self.x_m = x_m  # physics-model position
        self.excluded = False
        self.blast_remaining = 0  # ticks remaining in a blast event
        self.tilt_boost = 0.0     # extra simulated-hours offset from "simulate_tilt"
        self.prev_risk_tier = "safe"

        # Rolling reading history for feature engineering
        # Each entry: dict with raw sensor columns
        self.history = deque(maxlen=HISTORY_LEN)

    def reset(self):
        self.excluded = False
        self.blast_remaining = 0
        self.tilt_boost = 0.0
        self.prev_risk_tier = "safe"
        self.history.clear()


# ===================================================================
#  MODEL LOADER
# ===================================================================
class Models:
    """Container for all ML artefacts loaded once at startup."""

    def __init__(self):
        models_dir = os.path.join(_BACKEND_ROOT, "models")
        log.info("Loading models from %s ...", models_dir)

        self.if_model = joblib.load(os.path.join(models_dir, "isolation_forest.pkl"))
        self.xgb_model = joblib.load(os.path.join(models_dir, "xgboost_classifier.pkl"))
        self.scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))

        log.info("  Isolation Forest loaded")
        log.info("  XGBoost classifier loaded")
        log.info("  Scaler loaded")
        self.loaded = True

    def infer(self, feature_vector):
        """Run both models + risk fusion on a single feature vector.

        Args:
            feature_vector: 1-D array of shape (n_features,)
        Returns:
            (risk_score, risk_tier, xgb_class)
        """
        X = np.array([feature_vector], dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale using training scaler
        X_scaled = self.scaler.transform(X)

        # Isolation Forest anomaly score
        if_scores = self.if_model.anomaly_scores(X_scaled)

        # XGBoost class probability for subsidence
        xgb_sub_probs = self.xgb_model.subsidence_probability(X_scaled)
        xgb_pred = self.xgb_model.predict(X_scaled)[0]

        # Risk fusion (using config weights)
        risk_score = float(compute_risk_score(
            if_scores[0], xgb_sub_probs[0],
            if_weight=0.3, rf_weight=0.7,
        ))
        tier_arr = assign_risk_tier(np.array([risk_score]))
        risk_tier = str(tier_arr[0])

        return risk_score, risk_tier, xgb_pred


# ===================================================================
#  SIMULATION ENGINE
# ===================================================================
class SimulationEngine:
    """Drives the live physics simulation and model inference."""

    def __init__(self, models: Models):
        self.models = models
        self.rng = np.random.default_rng(42)

        # Build node states
        coords = build_node_coords()
        self.nodes = {
            nid: NodeState(nid, c["lat"], c["lng"], c["x_m"])
            for nid, c in coords.items()
        }

        # Build a subsidence profile for the simulated panel
        # Use Jharia-typical parameters
        self.geom = PanelGeometry(
            depth_m=300, width_m=250,
            extraction_thickness_m=3.0,
            angle_of_draw_deg=35,
            subsidence_factor=0.75,
            face_advance_rate_m_per_day=3.0,
            panel_id="live_panel",
        )
        self.profile = SubsidenceProfile(
            self.geom,
            active_phase_fraction=0.80,
            active_phase_days=60,
            residual_tau_days=270,
            h_disp_ratio=0.22,
        )

        # Simulation clock: each tick advances by this many "simulated hours"
        self.sim_hours_per_tick = 4.0  # 4 simulated hours per tick
        self.sim_hour = 0.0
        self.t_onset = 0.0  # subsidence starts immediately for demo responsiveness
        self.tick_count = 0
        self.baseline_temp = 32.0  # deg C — Jharia average

    def reset(self):
        """Reset all simulation state."""
        self.sim_hour = 0.0
        self.tick_count = 0
        self.rng = np.random.default_rng(42)
        for ns in self.nodes.values():
            ns.reset()
        log.info("Simulation reset")

    def compute_reading(self, ns: NodeState):
        """Compute a single raw sensor reading for one node at current sim_hour."""
        x = ns.x_m
        t = self.sim_hour + ns.tilt_boost  # tilt_boost from simulate_tilt command

        # --- Subsidence physics ---
        s_x = float(self.profile.spatial_profile(np.array([x]))[0])
        f_t = float(self.profile.time_profile(np.array([t]), self.t_onset)[0])
        v_disp = s_x * f_t  # vertical displacement (mm)

        # Tilt: spatial derivative
        dx = 50.0  # finite difference spacing
        s_plus = float(self.profile.spatial_profile(np.array([x + dx / 2]))[0])
        s_minus = float(self.profile.spatial_profile(np.array([x - dx / 2]))[0])
        slope_mm_per_m = (s_plus - s_minus) / dx
        tilt_base = math.degrees(math.atan(slope_mm_per_m / 1000.0)) * f_t

        # Strain: horizontal displacement differential
        spacing = 100.0
        s_x_n = float(self.profile.spatial_profile(np.array([x + spacing]))[0])
        strain_base = abs(s_x - s_x_n) * f_t * self.profile.h_disp_ratio

        # --- Vibration baseline ---
        vib_rms = max(0.001, self.rng.normal(0.005, 0.002))
        vib_freq = self.rng.uniform(2, 8)

        # --- Blast injection ---
        if ns.blast_remaining > 0:
            vib_rms = self.rng.uniform(0.5, 1.0)
            vib_freq = self.rng.uniform(15, 50)
            tilt_base += self.rng.uniform(0.01, 0.05) * (1 if self.rng.random() > 0.5 else -1)
            strain_base += self.rng.uniform(0.005, 0.02)
            ns.blast_remaining -= 1

        # --- MEMS noise ---
        tilt_noisy = tilt_base + self.rng.normal(0, 0.15)

        # --- Temperature ---
        hour_of_day = self.sim_hour % 24
        temp = self.baseline_temp + 7.5 * math.sin(2 * math.pi * (hour_of_day - 6) / 24)

        # --- Temperature drift on tilt ---
        temp_drift = 0.0029 * (temp - self.baseline_temp)
        tilt_final = tilt_noisy + temp_drift

        # --- Crack proxy (derived from inter-node strain differential) ---
        crack_proxy = abs(strain_base) * self.rng.uniform(0.8, 1.2)

        # Spectral ratio: high for blast, low otherwise
        spectral_ratio = min(1.0, vib_freq / 25.0) if vib_freq > 10 else vib_freq / 50.0

        return {
            "tilt_deg": round(tilt_final, 4),
            "vib_rms": round(abs(vib_rms), 6),
            "vib_dom_freq_hz": round(vib_freq, 2),
            "strain_mm": round(strain_base, 6),
            "crack_proxy": round(crack_proxy, 6),
            "temp_c": round(temp, 1),
            "v_disp_mm": round(v_disp, 4),
            "spectral_ratio": round(spectral_ratio, 4),
        }

    def build_feature_vector(self, ns: NodeState, reading: dict):
        """Build the 17-element feature vector from history + current reading."""
        ns.history.append(reading)
        hist = list(ns.history)
        n = len(hist)

        # Current values
        tilt_current = reading["tilt_deg"]
        vib_rms = reading["vib_rms"]
        vib_dom_freq = reading["vib_dom_freq_hz"]
        strain_current = reading["strain_mm"]
        crack_current = reading["crack_proxy"]
        temp_c = reading["temp_c"]
        spectral_ratio = reading["spectral_ratio"]

        # Rate of change (1-tick ~ 1 simulated hour)
        tilt_rate = (hist[-1]["tilt_deg"] - hist[-2]["tilt_deg"]) if n >= 2 else 0.0
        strain_rate = (hist[-1]["strain_mm"] - hist[-2]["strain_mm"]) if n >= 2 else 0.0

        # Rolling tilt stats
        tilt_vals = [h["tilt_deg"] for h in hist]
        tilt_mean_6h = np.mean(tilt_vals[-6:]) if n >= 1 else 0.0
        tilt_std_6h = np.std(tilt_vals[-6:]) if n >= 2 else 0.0
        tilt_mean_24h = np.mean(tilt_vals[-24:]) if n >= 1 else 0.0
        tilt_std_24h = np.std(tilt_vals[-24:]) if n >= 2 else 0.0

        # Vibration peak (rolling max over last 6)
        vib_peak = max(h["vib_rms"] for h in hist[-6:])

        # Cumulative strain and crack proxy
        strain_cumulative = sum(h["strain_mm"] for h in hist)
        crack_cumulative = sum(h["crack_proxy"] for h in hist)

        # Cross-node correlation placeholder (simplified: 0 for live)
        cross_corr = 0.0

        # Feature vector in EXACT order matching FEATURE_COLS
        return np.array([
            tilt_current,       # tilt_current
            tilt_rate,          # tilt_rate_1h
            tilt_mean_6h,       # tilt_mean_6h
            tilt_std_6h,        # tilt_std_6h
            tilt_mean_24h,      # tilt_mean_24h
            tilt_std_24h,       # tilt_std_24h
            vib_rms,            # vib_rms
            vib_peak,           # vib_peak
            vib_dom_freq,       # vib_dom_freq
            spectral_ratio,     # vib_spectral_ratio
            strain_current,     # strain_current
            strain_rate,        # strain_rate_1h
            strain_cumulative,  # strain_cumulative
            crack_current,      # crack_proxy_current
            crack_cumulative,   # crack_proxy_cumulative
            cross_corr,         # cross_node_corr
            temp_c,             # temp_c
        ], dtype=float)

    def tick(self):
        """Advance simulation by one step. Returns list of update dicts."""
        self.sim_hour += self.sim_hours_per_tick
        self.tick_count += 1
        now_iso = datetime.now(timezone.utc).isoformat()

        updates = []
        active_count = 0
        excluded_count = 0

        # Count excluded for quorum check
        for ns in self.nodes.values():
            if ns.excluded:
                excluded_count += 1
            else:
                active_count += 1

        # Quorum degradation: if >50% of nodes are excluded, cap
        # risk_tier at "watch" to avoid overconfident alerts on
        # badly under-sampled data (simple fault-tolerant guard).
        quorum_degraded = excluded_count > len(self.nodes) / 2

        for ns in self.nodes.values():
            if ns.excluded:
                updates.append({
                    "type": "node_update",
                    "node_id": ns.node_id,
                    "lat": ns.lat,
                    "lng": ns.lng,
                    "tilt_deg": 0.0,
                    "vibration_rms": 0.0,
                    "strain_mm": 0.0,
                    "risk_score": 0.0,
                    "risk_tier": "safe",
                    "status": "excluded",
                    "timestamp": now_iso,
                })
                continue

            # Compute raw reading
            reading = self.compute_reading(ns)

            # Warm-up: need at least 3 readings before inference
            if len(ns.history) < 2:
                ns.history.append(reading)
                updates.append({
                    "type": "node_update",
                    "node_id": ns.node_id,
                    "lat": ns.lat,
                    "lng": ns.lng,
                    "tilt_deg": reading["tilt_deg"],
                    "vibration_rms": reading["vib_rms"],
                    "strain_mm": reading["strain_mm"],
                    "risk_score": 0.0,
                    "risk_tier": "safe",
                    "status": "active",
                    "timestamp": now_iso,
                })
                continue

            # Build feature vector and run inference
            features = self.build_feature_vector(ns, reading)
            risk_score, risk_tier, xgb_class = self.models.infer(features)

            # Quorum guard
            if quorum_degraded and risk_tier in ("warning", "critical"):
                risk_tier = "watch"

            updates.append({
                "type": "node_update",
                "node_id": ns.node_id,
                "lat": ns.lat,
                "lng": ns.lng,
                "tilt_deg": reading["tilt_deg"],
                "vibration_rms": reading["vib_rms"],
                "strain_mm": reading["strain_mm"],
                "risk_score": round(risk_score, 1),
                "risk_tier": risk_tier,
                "status": "active",
                "timestamp": now_iso,
            })

            # Transition alert detection
            if risk_tier in ("warning", "critical") and ns.prev_risk_tier not in ("warning", "critical"):
                alert_msg = self._build_alert_message(ns, reading, risk_tier, xgb_class)
                updates.append({
                    "type": "alert",
                    "node_id": ns.node_id,
                    "severity": risk_tier,
                    "message": alert_msg,
                    "timestamp": now_iso,
                })

            ns.prev_risk_tier = risk_tier

        # System status every 3 ticks
        if self.tick_count % 3 == 0:
            updates.append({
                "type": "system_status",
                "active_nodes": active_count,
                "excluded_nodes": excluded_count,
                "gateway_status": "online",
            })

        return updates

    def _build_alert_message(self, ns, reading, tier, xgb_class):
        """Build a descriptive alert message from actual feature values."""
        tilt = reading["tilt_deg"]
        strain = reading["strain_mm"]
        vib = reading["vib_rms"]

        if xgb_class == "subsidence":
            # Find adjacent nodes with similar elevated tilt
            coherent = []
            for other in self.nodes.values():
                if other.node_id != ns.node_id and not other.excluded and len(other.history) > 0:
                    other_tilt = other.history[-1]["tilt_deg"]
                    if abs(other_tilt) > 0.05 and abs(ns.x_m - other.x_m) < 200:
                        coherent.append(other.node_id)
            if coherent:
                return (f"Sustained tilt {abs(tilt):.2f} deg, strain {strain:.3f} mm, "
                        f"coherent across {ns.node_id}/{'/'.join(coherent[:2])}")
            return f"Subsidence signal: tilt {abs(tilt):.2f} deg, cumulative strain {strain:.3f} mm at {ns.node_id}"
        elif xgb_class == "blast_transient":
            return f"Blast transient: vibration {vib:.3f}g at {ns.node_id}, monitoring for persistence"
        elif xgb_class == "rain_creep":
            return f"Rain-creep detected at {ns.node_id}: tilt {abs(tilt):.2f} deg (monitoring for reversal)"
        else:
            return f"Elevated risk at {ns.node_id}: tilt={abs(tilt):.2f} deg, strain={strain:.3f} mm"

    # --- Command handlers ---
    def handle_simulate_tilt(self, node_id, intensity):
        if node_id in self.nodes:
            # Fast-forward simulated hours by intensity * 500h
            boost = intensity * 500.0
            self.nodes[node_id].tilt_boost += boost
            log.info("simulate_tilt: %s += %.0f sim-hours (intensity=%.2f)", node_id, boost, intensity)

    def handle_inject_blast(self, node_id):
        if node_id in self.nodes:
            self.nodes[node_id].blast_remaining = 3  # 3 ticks of blast
            log.info("inject_blast: %s", node_id)

    def handle_kill_nodes(self, node_ids):
        for nid in node_ids:
            if nid in self.nodes:
                self.nodes[nid].excluded = True
                log.info("kill_node: %s", nid)

    def handle_reset(self):
        self.reset()


# ===================================================================
#  FASTAPI APP + WEBSOCKET
# ===================================================================
connected_clients: set[WebSocket] = set()
engine: SimulationEngine | None = None
models_instance: Models | None = None
sim_task: asyncio.Task | None = None

TICK_INTERVAL_SECONDS = 3.0  # configurable demo speed


async def simulation_loop():
    """Background loop: advance simulation and broadcast to all clients."""
    global engine
    log.info("Simulation loop started (tick every %.1fs)", TICK_INTERVAL_SECONDS)

    while True:
        try:
            updates = engine.tick()

            # Broadcast to all connected clients
            if connected_clients:
                dead = set()
                for ws in connected_clients:
                    try:
                        for msg in updates:
                            await ws.send_json(msg)
                    except Exception:
                        dead.add(ws)
                connected_clients.difference_update(dead)

            await asyncio.sleep(TICK_INTERVAL_SECONDS)

        except asyncio.CancelledError:
            log.info("Simulation loop cancelled")
            break
        except Exception as e:
            log.exception("Simulation tick error: %s", e)
            await asyncio.sleep(TICK_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and start simulation on app startup."""
    global models_instance, engine, sim_task

    models_instance = Models()
    engine = SimulationEngine(models_instance)
    sim_task = asyncio.create_task(simulation_loop())
    log.info("GroundWatch server started — %d nodes, Jharia centre (%.4f, %.4f)",
             len(engine.nodes), JHARIA_CENTER_LAT, JHARIA_CENTER_LNG)

    yield  # app runs

    sim_task.cancel()
    await sim_task
    log.info("Server shutdown")


app = FastAPI(title="GroundWatch Simulation Server", lifespan=lifespan)

# CORS for Vite dev server (default port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": models_instance is not None and models_instance.loaded,
        "nodes": len(engine.nodes) if engine else 0,
        "sim_hour": engine.sim_hour if engine else 0,
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.add(ws)
    log.info("Client connected (%d total)", len(connected_clients))

    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
                cmd = msg.get("command", "")

                if cmd == "simulate_tilt":
                    engine.handle_simulate_tilt(
                        msg.get("node_id", ""),
                        float(msg.get("intensity", 0.5)),
                    )
                elif cmd == "inject_blast":
                    engine.handle_inject_blast(msg.get("node_id", ""))
                elif cmd == "kill_nodes":
                    engine.handle_kill_nodes(msg.get("node_ids", []))
                elif cmd == "reset":
                    engine.handle_reset()
                else:
                    log.warning("Unknown command: %s", cmd)

            except json.JSONDecodeError:
                log.warning("Invalid JSON from client")

    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.debug("WS error: %s", e)
    finally:
        connected_clients.discard(ws)
        log.info("Client disconnected (%d remaining)", len(connected_clients))
