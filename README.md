# GroundWatch

Real-time mine subsidence monitoring and risk classification system for Indian coalfields.

GroundWatch simulates a network of IoT sensors deployed across a longwall coal mine panel, generates physics-consistent sensor readings, trains a two-model ML stack to classify ground movement events, and streams risk scores live to a React dashboard over WebSocket.

---

## Table of Contents

- [What it does](#what-it-does)
- [System Architecture](#system-architecture)
- [Repository Layout](#repository-layout)
- [Quick Start](#quick-start)
- [Backend — Step-by-Step](#backend--step-by-step)
  - [1. Configuration](#1-configuration)
  - [2. Data Generation](#2-data-generation)
  - [3. Preprocessing & Feature Engineering](#3-preprocessing--feature-engineering)
  - [4. Training](#4-training)
  - [5. Evaluation](#5-evaluation)
  - [6. Live Server](#6-live-server)
- [Frontend Dashboard](#frontend-dashboard)
- [ML Models in Detail](#ml-models-in-detail)
- [Model Performance](#model-performance)
- [WebSocket API](#websocket-api)
- [Configuration Reference](#configuration-reference)
- [Dependency Reference](#dependency-reference)

---

## What it does

Underground longwall coal mining causes predictable surface subsidence — the ground above extracted coal settles as goaf forms. GroundWatch classifies each 1-hour sensor window into one of four event types:

| Class | Description |
|-------|-------------|
| `normal` | No anomaly — baseline sensor drift and noise only |
| `blast_transient` | Scheduled mine blasting — sharp high-amplitude vibration, small transient tilt |
| `rain_creep` | Rainfall-induced shallow soil creep — small tilt that partially *reverses* as soil dries |
| `subsidence` | True underground subsidence — monotonically growing tilt + strain, coherent across nodes |

The system is designed for i3-class edge CPUs with a 5 MB serialised model budget. All physics parameters trace to named Indian standards and literature sources (CMRI, IS 14562, DGMS India).

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Data Generation                                                  │
│  subsidence_profile.py  ·  noise_injection.py                    │
│  CMRI influence function, MEMS noise, blast, rain creep          │
└──────────────────────┬───────────────────────────────────────────┘
                       │  raw CSVs + ground-truth labels
┌──────────────────────▼───────────────────────────────────────────┐
│  Preprocessing & Feature Engineering                              │
│  filtering.py  ·  feature_engineering.py  ·  pipeline.py        │
│  28 features: tilt reversal, multi-day rolling, cross-node corr  │
└──────────────────────┬───────────────────────────────────────────┘
                       │  normalised Parquet feature matrices
┌──────────────────────▼───────────────────────────────────────────┐
│  ML Models                                                        │
│  IsolationForest  →  anomaly score [0–1]                         │
│  XGBoost (140 trees)  →  4-class probabilities                   │
│  risk_fusion.py  →  0–100 risk score + Safe/Watch/Warning/Crit.  │
└──────────────────────┬───────────────────────────────────────────┘
                       │  .pkl model files
┌──────────────────────▼───────────────────────────────────────────┐
│  FastAPI + WebSocket Server  (server.py)                          │
│  Physics tick loop → inference → broadcast                        │
└──────────────────────┬───────────────────────────────────────────┘
                       │  ws://localhost:8000/ws
┌──────────────────────▼───────────────────────────────────────────┐
│  React Dashboard  (Vite + Tailwind)                               │
│  MapView  ·  NodeGraph  ·  AlertFeed  ·  ControlPanel            │
└──────────────────────────────────────────────────────────────────┘
```

---

## Repository Layout

```
groundwatch-demo/
├── README.md
├── .gitignore
│
├── backend/
│   ├── requirements.txt
│   ├── configs/
│   │   └── config.yaml              ← all physics + ML hyperparameters
│   │
│   ├── data/                        ← gitignored (generated at runtime)
│   │   ├── raw/synthetic/           ← per-panel sensor CSVs
│   │   ├── processed/               ← normalised Parquet feature matrices
│   │   └── labels/window_labels.csv ← ground-truth labels
│   │
│   ├── models/                      ← gitignored (generated at runtime)
│   │   ├── isolation_forest.pkl
│   │   ├── xgboost_classifier.pkl
│   │   ├── label_encoder.pkl
│   │   ├── scaler.pkl
│   │   └── evaluation_report.md     ← committed; updated after each eval run
│   │
│   └── src/
│       ├── data_generation/
│       │   ├── subsidence_profile.py   ← CMRI physics model
│       │   ├── noise_injection.py      ← MEMS noise, blast, rain injection
│       │   └── generate_dataset.py     ← orchestrator: 18 panels × 180 days
│       │
│       ├── preprocessing/
│       │   ├── filtering.py            ← resample, gap-fill, temp correction
│       │   ├── feature_engineering.py  ← 28-feature extraction per node window
│       │   └── pipeline.py             ← split → normalise → write Parquet
│       │
│       ├── models/
│       │   ├── isolation_forest.py     ← anomaly detector wrapper
│       │   ├── xgboost_classifier.py   ← 4-class classifier wrapper
│       │   ├── risk_fusion.py          ← IF score + XGB prob → risk tier
│       │   └── random_forest.py        ← legacy RF (kept for reference)
│       │
│       ├── training/
│       │   └── train.py               ← end-to-end training pipeline
│       │
│       ├── evaluation/
│       │   └── evaluate.py            ← confusion matrix, feature importance, report
│       │
│       └── server.py                  ← FastAPI + WebSocket live simulation
│
└── frontend/
    ├── index.html
    ├── package.json
    └── src/
        ├── main.jsx
        ├── App.jsx                    ← root component, WebSocket state
        ├── socket.js                  ← WS client with auto-reconnect
        ├── index.css
        └── components/
            ├── MapView.jsx            ← Leaflet map with risk-coloured nodes
            ├── NodeGraph.jsx          ← Recharts time-series per node
            ├── AlertFeed.jsx          ← scrolling real-time alert strip
            └── ControlPanel.jsx       ← node list + manual command panel
```

---

## Quick Start

```powershell
# ── Backend ──────────────────────────────────────────────
cd backend
pip install -r requirements.txt

python src/data_generation/generate_dataset.py   # ~2 min
python src/preprocessing/pipeline.py             # ~3 min
python src/training/train.py                     # ~2 min
python -m uvicorn src.server:app --port 8000

# ── Frontend (new terminal) ───────────────────────────────
cd frontend
npm install
npm run dev                                       # → http://localhost:5173
```

---

## Backend — Step-by-Step

### 1. Configuration

**File:** `backend/configs/config.yaml`

Single source of truth for every physics parameter and ML hyperparameter. Key sections:

```yaml
dataset:
  n_panels: 18
  simulation_days: 180
  sampling_interval_hours: 1

noise:
  mems:
    noise_std_deg: 0.15          # ±0.15° RMS (InvenSense MPU-6050 spec)
  blast:
    events_per_day: [1, 3]
    vib_rms_range_g: [0.5, 1.0]
  rain:
    events_per_month: [3, 5]     # increased 3× for balanced training data
    affected_node_fraction: [0.4, 0.7]
    decay_days: [3, 7]           # tilt reversal window

models:
  xgboost:
    n_estimators: 140
    max_depth: 8
    learning_rate: 0.05
    subsample: 0.8
    colsample_bytree: 0.8
    min_child_weight: 5
  risk_fusion:
    if_weight: 0.3
    rf_weight: 0.7
    tier_thresholds: {safe: 25, watch: 50, warning: 75}
  max_model_size_mb: 5           # hard edge-deployment constraint
```

---

### 2. Data Generation

**Run:**
```powershell
cd backend
python src/data_generation/generate_dataset.py
```

**What it produces:**
- `data/raw/synthetic/panel_XXX/nNN.csv` — hourly sensor readings per node
- `data/raw/synthetic/panel_XXX/panel_meta.json` — panel geometry params
- `data/labels/window_labels.csv` — ground-truth label per (panel, node, hour)

**Physics model** (`subsidence_profile.py`):

Implements the CMRI/NCB influence-function approach (IS 14562:1998). Displacement at surface point `x`, time `t`:

```
S(x, t) = S_max × gaussian_profile(x) × time_profile(t)
```

- **Spatial profile**: Gaussian bell, half-width = depth × tan(angle_of_draw). Angle of draw fixed at 35° for Indian coal measures.
- **Time profile**: two-phase — rapid active phase (70–90% of S_max over 30–90 days) + exponential residual tail (τ = 180–365 days).
- **Panel parameters** sampled per panel: depth 100–600 m, width 100–400 m, seam thickness 2–4 m, subsidence factor 0.6–0.9.

**Noise & confounder injection** (`noise_injection.py`):

| Source | Parameters | Reference |
|--------|-----------|-----------|
| MEMS noise | ±0.15° RMS | InvenSense MPU-6050 datasheet |
| Temperature drift | 0.0029°/°C | Wi-GIM calibration study |
| Blast vibration | 0.5–1.0 g RMS, 15–50 Hz | DGMS India ground vibration standards |
| Rain creep tilt | 0.01–0.08°, 3–7 day reversal | Geotechnical monitoring literature |
| Packet dropout | 5–15% burst loss | LoRa mesh characterisation |

Key physics properties:
- **Blast**: high vibration (×100 baseline), small non-persistent tilt — decays within 1–2 windows, zero cumulative displacement.
- **Rain creep**: tilt ramps up during rainfall, then *partially reverses* (20–40% permanent residual) as soil dries. This reversal pattern is the primary discriminator from true subsidence.

---

### 3. Preprocessing & Feature Engineering

**Run:**
```powershell
python src/preprocessing/pipeline.py
```

**What it produces:**
- `data/processed/features_train.parquet`
- `data/processed/features_val.parquet`
- `data/processed/features_test.parquet`
- `models/scaler.pkl` — StandardScaler fitted on train split only

**Panel-level train/val/test split** (in `pipeline.py`):
- 13 panels → train | 3 panels → val | 2 panels → test
- Split is by panel ID, **not row**, to prevent geometry leakage across splits.

**Filtering** (`filtering.py`):
1. Resample to uniform 1-hour grid (forward-fill gaps ≤ 6 hours)
2. Remove out-of-range sensor readings
3. Compensate temperature-induced tilt drift: `tilt_corrected = tilt - 0.0029 × (temp - temp_ref)`

**Feature engineering** (`feature_engineering.py`) — 28 features total:

| Group | Features | Purpose |
|-------|----------|---------|
| **Tilt (short-term)** | `tilt_current`, `tilt_rate_1h`, `tilt_mean/std_6h`, `tilt_mean/std_24h` | Instantaneous + intraday dynamics |
| **Vibration** | `vib_rms`, `vib_peak`, `vib_dom_freq`, `vib_spectral_ratio` | Frequency content: blasts peak >10 Hz; creep < 2 Hz |
| **Strain** | `strain_current`, `strain_rate_1h`, `strain_cumulative` | Cumulative drift monotonically increases only in subsidence |
| **Crack proxy** | `crack_proxy_current`, `crack_proxy_cumulative` | `\|d(strain)/dt\|` — cumulative value tracks subsidence severity |
| **Tilt reversal** *(added v2)* | `tilt_reversal_72h`, `tilt_delta_72h` | `Δtilt(72h) / Δtilt(168h)` — negative = reversal (rain), positive = continuation (subsidence) |
| **Multi-day rolling** *(added v2)* | `tilt_mean/std_72h`, `tilt_mean/std_168h`, `tilt_monotonicity` | Rain events span 12–48 h + 3–7 day decay; 24 h window missed the full lifecycle |
| **Vibration persistence** *(added v2)* | `vib_rms_mean_24h`, `vib_elevated_hours` | Blasts are 1–2 window spikes; sustained elevation → different source |
| **Strain reversal** *(added v2)* | `strain_rate_24h`, `strain_reversal_72h` | Mirrors tilt reversal for strain sensor |
| **Cross-node** | `cross_node_corr` | 72 h rolling Pearson correlation with neighbouring nodes. Subsidence propagates coherently (high); rain is patchy (low–moderate) |
| **Environment** | `temp_c` | Temperature for thermal drift context |

> **Why not SMOTE?** These features are physically coupled through the subsidence physics — `vib_rms`, `cross_node_corr`, and `strain_cumulative` co-vary in ways SMOTE cannot preserve. Instead, `events_per_month` in the rain config was tripled to generate real physics-consistent events.

---

### 4. Training

**Run:**
```powershell
python src/training/train.py
```

Saves to `models/`: `isolation_forest.pkl`, `xgboost_classifier.pkl`, `label_encoder.pkl`, `scaler.pkl`

Asserts model size ≤ 5 MB (hard edge-deployment constraint) before exiting.

**Isolation Forest** (`models/isolation_forest.py`):
- Trained on normal-class windows only
- Contamination parameter auto-tuned to the observed anomaly rate on the validation set
- Outputs a normalised anomaly score ∈ [0, 1]
- 150 trees, `max_features=0.8` — serialised size: 1.18 MB

**XGBoost Classifier** (`models/xgboost_classifier.py`):
- Subsidence tiers collapsed: `subsidence_watch / warning / critical → subsidence`
- 4 classes: `normal`, `blast_transient`, `rain_creep`, `subsidence`
- Per-class sample weights applied for imbalanced training data
- Early stopping on validation `mlogloss` (patience = 25 rounds)

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| `n_estimators` | 140 | Uses model size headroom (4.82 MB < 5 MB limit) |
| `max_depth` | 8 | Deeper trees needed for rain_creep subtleties |
| `learning_rate` | 0.05 | Lower rate + more trees = better generalisation |
| `tree_method` | `hist` | O(bins × features) — mandatory for i3-CPU training speed |
| `subsample` | 0.8 | Row sampling for ensemble diversity |
| `colsample_bytree` | 0.8 | Feature sampling per tree |
| `min_child_weight` | 5 | Prevents overfitting on rare event boundaries |

---

### 5. Evaluation

**Run:**
```powershell
python src/evaluation/evaluate.py
```

**Outputs:**
- Console: overall accuracy, false-alarm rate, classification report, risk score distribution
- `models/confusion_matrix.png`
- `models/feature_importance.png` (top-10 XGBoost gain)
- `models/evaluation_report.md`

**False-alarm rate** = fraction of blast/rain windows incorrectly classified as subsidence. This is the primary safety metric — each false alert triggers an unnecessary evacuation or shutdown.

---

### 6. Live Server

**Run:**
```powershell
# Standard
python -m uvicorn src.server:app --host 0.0.0.0 --port 8000

# With hot-reload during development
python -m uvicorn src.server:app --reload --port 8000
```

`server.py` loads all trained models at startup, then runs a physics tick loop. Each tick:

1. Advances the subsidence simulation by 4 simulated hours
2. Injects MEMS noise + temperature drift
3. Extracts the same 28 features used during training
4. Runs Isolation Forest → anomaly score; XGBoost → class probabilities
5. Computes fused risk score and tier via `risk_fusion.py`
6. Broadcasts `node_update` messages to all connected WebSocket clients

**Simulation geography:** Jharia Coalfield, Dhanbad (23.74°N, 86.41°E). Nodes are placed on a configurable grid at ~100 m spacing.

---

## Frontend Dashboard

**Stack:** React 18, Vite, Tailwind CSS, Leaflet, Recharts, Lucide icons

**Run:**
```powershell
cd frontend
npm install
npm run dev      # → http://localhost:5173
```

Connects automatically to `ws://localhost:8000/ws`. Reconnects every 3 seconds on disconnect (exponential backoff disabled — fixed 3 s for fast recovery in a demo setting).

**Components:**

| Component | File | Description |
|-----------|------|-------------|
| **MapView** | `components/MapView.jsx` | Leaflet map. Each node rendered as a coloured circle — green (Safe), amber (Watch), orange (Warning), red (Critical). Click to select a node. |
| **NodeGraph** | `components/NodeGraph.jsx` | Recharts time-series for the selected node: tilt (°), vibration RMS (g), strain (mm) — last 60 data points. |
| **AlertFeed** | `components/AlertFeed.jsx` | Scrolling strip of real-time alerts, newest first, colour-coded by tier. Capped at 200 entries. |
| **ControlPanel** | `components/ControlPanel.jsx` | Tabular node list with current risk tier badges. Manual command interface (send JSON commands to server). |
| **socket.js** | `src/socket.js` | WebSocket client singleton. `subscribe(fn)`, `onConnectionChange(fn)`, `send(obj)` exports. Auto-reconnects on close. |

**Risk tier colours:**

| Tier | Meaning | Colour |
|------|---------|--------|
| Safe | Risk score < 25 | Green |
| Watch | 25–49 | Amber |
| Warning | 50–74 | Orange |
| Critical | ≥ 75 | Red |

---

## ML Models in Detail

### Risk Fusion (`models/risk_fusion.py`)

Combines both model outputs into a single transparent 0–100 risk score:

```
risk_score = 100 × (0.3 × IF_anomaly_score + 0.7 × XGB_subsidence_prob)
```

The 0.7 weight on XGBoost reflects that the classifier has been explicitly trained to discriminate subsidence from confounders. The 0.3 IF component acts as a safety net for novel anomaly patterns not represented in training data.

Tier assignment is a simple threshold lookup — fully auditable, no secondary model.

### Why two models?

| Model | Strength | Limitation |
|-------|----------|-----------|
| Isolation Forest | Detects *any* anomaly, including novel events not in training | Cannot distinguish between subsidence and blast/rain |
| XGBoost | High precision discrimination across 4 specific classes | Blind to event types not in training data |
| **Fusion** | Flags if *either* signal is elevated | — |

---

## Model Performance

Test set: 2 held-out panels (panel_017, panel_018), **never seen during training.**

### Improvement history

| Version | Changes | Accuracy | rain_creep F1 | False-alarm rate | Size |
|---------|---------|:--------:|:-------------:|:----------------:|:----:|
| v1 (baseline) | 17 features, 80 trees, depth 6 | 68.6% | 0.24 | 1.72% | 1.15 MB |
| v2 | +11 features, hyperparameter tuning | 76.9% | 0.36 | 0.68% | 4.38 MB |
| **v3 (current)** | **+ 3× rain_creep data enrichment** | **75.1%** | **0.62** | **0.79%** | **4.82 MB** |

> v3 raw accuracy is slightly below v2 because the test set itself changed — rain_creep grew from 10.2% → 22.8% of test windows (harder problem). Per-class metrics show the true improvement: rain_creep F1 nearly tripled from baseline.

### Final per-class metrics (v3)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| subsidence | 0.84 | **1.00** | **0.91** | 12,041 |
| normal | 0.84 | 0.71 | 0.77 | 51,578 |
| blast_transient | 0.69 | 0.86 | 0.77 | 9,708 |
| rain_creep | 0.57 | 0.67 | 0.62 | 21,713 |

### Risk score separation

| True class | Mean score | Median |
|-----------|:----------:|:------:|
| subsidence | 88.3 | 88.6 |
| blast_transient | 14.0 | 13.4 |
| normal | 8.3 | 4.5 |
| rain_creep | 5.9 | 4.7 |

Both confounders sit deep in the Safe zone regardless of classification — the risk fusion layer is robust to normal↔rain_creep confusion at the decision boundary.

---

## WebSocket API

All messages are JSON. The server pushes; the client can optionally send commands.

### Server → Client

**`node_update`** — emitted every tick for each active node:
```json
{
  "type": "node_update",
  "node_id": "n03",
  "lat": 23.7418,
  "lng": 86.4132,
  "tilt_deg": 0.042,
  "vibration_rms": 0.006,
  "strain_mm": 1.24,
  "risk_score": 81.3,
  "risk_tier": "critical",
  "label": "subsidence",
  "timestamp": "2024-03-15T14:23:00Z"
}
```

**`alert`** — emitted when a node's risk tier rises:
```json
{
  "type": "alert",
  "node_id": "n03",
  "tier": "critical",
  "message": "Node n03 entered CRITICAL state — risk score 81.3",
  "timestamp": "2024-03-15T14:23:00Z"
}
```

**`system_status`** — emitted every 10 ticks:
```json
{
  "type": "system_status",
  "active_nodes": 9,
  "excluded_nodes": 1,
  "gateway_status": "online"
}
```

### Client → Server

```json
{ "command": "reset_simulation" }
{ "command": "exclude_node", "node_id": "n07" }
{ "command": "include_node", "node_id": "n07" }
```

---

## Configuration Reference

`backend/configs/config.yaml` — all values with inline comments. Key tuneable parameters:

| Key | Default | Effect |
|-----|---------|--------|
| `dataset.n_panels` | 18 | Number of synthetic mine panels to generate |
| `dataset.simulation_days` | 180 | Length of each panel simulation in days |
| `noise.rain.events_per_month` | [3, 5] | Rain events per month per panel |
| `noise.rain.affected_node_fraction` | [0.4, 0.7] | Fraction of nodes affected per rain event |
| `noise.blast.events_per_day` | [1, 3] | Daily blast schedule |
| `models.xgboost.n_estimators` | 140 | Number of boosted trees |
| `models.xgboost.max_depth` | 8 | Maximum tree depth |
| `models.xgboost.learning_rate` | 0.05 | Boosting learning rate |
| `models.risk_fusion.if_weight` | 0.3 | Weight of IF anomaly score in fusion |
| `models.risk_fusion.rf_weight` | 0.7 | Weight of XGBoost subsidence probability |
| `models.risk_fusion.tier_thresholds.safe` | 25 | Risk score threshold for Watch tier |
| `models.risk_fusion.tier_thresholds.watch` | 50 | Risk score threshold for Warning tier |
| `models.risk_fusion.tier_thresholds.warning` | 75 | Risk score threshold for Critical tier |
| `models.max_model_size_mb` | 5 | Hard edge-deployment size constraint |

---

## Dependency Reference

### Backend (`backend/requirements.txt`)

| Package | Version | Use |
|---------|---------|-----|
| `numpy` | ≥1.24, <2.0 | Array ops throughout |
| `scipy` | ≥1.11 | Gaussian subsidence profile, signal processing |
| `pandas` | ≥2.0 | Timeseries + Parquet I/O |
| `scikit-learn` | ≥1.3 | IsolationForest, StandardScaler, LabelEncoder |
| `xgboost` | ≥2.0, <3.0 | Multi-class gradient boosted classifier |
| `joblib` | ≥1.3 | Model serialisation (.pkl) |
| `pyarrow` | ≥14.0 | Parquet feature matrix storage |
| `pyyaml` | ≥6.0 | Config loading |
| `matplotlib` | ≥3.7 | Confusion matrix + feature importance plots |
| `fastapi` | ≥0.110 | REST + WebSocket server |
| `uvicorn[standard]` | ≥0.29 | ASGI server |

### Frontend (`frontend/package.json`)

| Package | Use |
|---------|-----|
| `react` / `react-dom` | UI framework |
| `vite` | Build tool + dev server |
| `tailwindcss` | Utility CSS |
| `leaflet` / `react-leaflet` | Interactive map |
| `recharts` | Time-series charts |
| `lucide-react` | Icon set |

---

## References

- CMRI Technical Manual on Mine Subsidence (1997)
- IS 14562:1998 — Indian Standard for Subsidence Prediction in Coal Mines
- Chatterjee et al. (2015), *J. Earth System Science* — GPS deformation, Jharia Coalfield
- Dey et al. (2018) — InSAR subsidence mapping, Raniganj Coalfield
- Djamaluddin et al. (2011), *Int. J. Mining Sci. & Technology*
- DGMS India Circular on Ground Vibration Limits from Blasting
- InvenSense MPU-6050 Product Specification Rev 3.4
- Wi-GIM field calibration study — MEMS tilt sensor temperature drift coefficients
