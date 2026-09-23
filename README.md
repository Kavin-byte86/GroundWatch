# GroundWatch

**Real-time mine subsidence monitoring and risk classification system for Indian coalfields.**

GroundWatch is a physics-informed synthetic dataset generator + ML classification pipeline + live WebSocket dashboard, built to detect underground mine subsidence from virtual IoT sensor readings. The system discriminates true ground subsidence from confounders (blast transients, rainfall-induced soil creep) and streams risk scores to a React dashboard in real time.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [ML Pipeline](#ml-pipeline)
  - [1. Data Generation](#1-data-generation)
  - [2. Preprocessing & Feature Engineering](#2-preprocessing--feature-engineering)
  - [3. Models](#3-models)
  - [4. Training](#4-training)
  - [5. Evaluation](#5-evaluation)
- [Model Performance](#model-performance)
- [Live Simulation Server](#live-simulation-server)
- [Frontend Dashboard](#frontend-dashboard)
- [Configuration](#configuration)
- [Setup & Running](#setup--running)

---

## Overview

Underground longwall coal mining causes predictable surface subsidence — the ground settles as extracted coal pillars are replaced by goaf. GroundWatch simulates a network of IoT sensors (MEMS tilt sensors, vibration transducers, strain gauges) placed across a mine panel surface, generates physics-consistent sensor readings, and trains a two-model ML stack to classify each 1-hour time window as:

| Class | Description |
|-------|-------------|
| `normal` | Baseline sensor readings, no anomaly |
| `blast_transient` | Scheduled mine blasting — high vibration, transient tilt perturbation |
| `rain_creep` | Rainfall-induced shallow soil creep — small tilt that partially reverses |
| `subsidence` | True underground subsidence — monotonically growing tilt + strain |

The subsidence severity tiers (`watch`, `warning`, `critical`) are preserved in the raw labels and used by the risk fusion layer; the classifier collapses them to a single `subsidence` class for discrimination.

**Domain grounding:** All physics parameters trace to named literature sources (CMRI, IS 14562, Chatterjee et al. 2015, DGMS India, InvenSense MPU-6050 datasheet). See inline comments in `config.yaml` and `noise_injection.py`.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  Data Generation Layer                    │
│  subsidence_profile.py   noise_injection.py              │
│  (CMRI influence function, MEMS noise, blast, rain creep)│
└────────────────────────┬─────────────────────────────────┘
                         │  raw CSVs + labels
┌────────────────────────▼─────────────────────────────────┐
│               Preprocessing & Feature Engineering         │
│  filtering.py   feature_engineering.py   pipeline.py     │
│  (28 features: tilt reversal, multi-day rolling,         │
│   vibration persistence, strain reversal, cross-node corr)│
└────────────────────────┬─────────────────────────────────┘
                         │  Parquet feature matrices
┌────────────────────────▼─────────────────────────────────┐
│                     ML Models                             │
│  IsolationForest  (anomaly score 0→1)                    │
│  XGBoostClassifier  (4-class softmax, 140 trees)         │
│  RiskFusion  (weighted combination → 0–100 risk score)   │
└────────────────────────┬─────────────────────────────────┘
                         │  .pkl models
┌────────────────────────▼─────────────────────────────────┐
│             FastAPI + WebSocket Server (server.py)        │
│  Physics tick loop → inference → risk tier → broadcast   │
└────────────────────────┬─────────────────────────────────┘
                         │  WebSocket
┌────────────────────────▼─────────────────────────────────┐
│               React Dashboard (Vite + Tailwind)           │
│  MapView   NodeGraph   AlertFeed   ControlPanel          │
└──────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
groundwatch-demo/
├── backend/
│   ├── configs/
│   │   └── config.yaml              # All physics + model hyperparameters
│   ├── data/
│   │   ├── raw/synthetic/           # Generated panel CSVs (gitignored)
│   │   ├── processed/               # Parquet feature matrices (gitignored)
│   │   └── labels/window_labels.csv # Ground-truth labels (gitignored)
│   ├── models/                      # Trained .pkl files (gitignored)
│   │   ├── isolation_forest.pkl
│   │   ├── xgboost_classifier.pkl
│   │   ├── label_encoder.pkl
│   │   ├── scaler.pkl
│   │   └── evaluation_report.md
│   ├── requirements.txt
│   └── src/
│       ├── data_generation/
│       │   ├── subsidence_profile.py  # CMRI influence-function physics model
│       │   ├── noise_injection.py     # MEMS noise, blast, rain creep injection
│       │   └── generate_dataset.py    # Orchestrator: 18 panels × 180 days
│       ├── preprocessing/
│       │   ├── filtering.py           # Resampling, gap handling, temp correction
│       │   ├── feature_engineering.py # 28-feature extraction per node window
│       │   └── pipeline.py            # Panel-split → normalize → Parquet output
│       ├── models/
│       │   ├── isolation_forest.py    # Unsupervised anomaly scorer
│       │   ├── xgboost_classifier.py  # 4-class gradient boosted classifier
│       │   ├── random_forest.py       # Legacy RF (kept for reference)
│       │   └── risk_fusion.py         # IF score + XGB prob → 0–100 risk tier
│       ├── training/
│       │   └── train.py               # End-to-end training pipeline
│       ├── evaluation/
│       │   └── evaluate.py            # Confusion matrix, report, feature importance
│       └── server.py                  # FastAPI + WebSocket live simulation server
└── frontend/
    ├── index.html
    ├── package.json
    └── src/
        ├── main.jsx
        ├── App.jsx                    # Root: WebSocket state, layout
        ├── socket.js                  # WebSocket client with auto-reconnect
        └── components/
            ├── MapView.jsx            # Leaflet map — colour-coded risk nodes
            ├── NodeGraph.jsx          # Recharts tilt/vib/strain time-series
            ├── ControlPanel.jsx       # Node list + manual commands
            └── AlertFeed.jsx          # Scrolling real-time alert strip
```

---

## ML Pipeline

### 1. Data Generation

**Files:** `src/data_generation/`

#### `subsidence_profile.py` — Physics Model
Implements the CMRI/NCB influence-function approach (IS 14562:1998) to compute vertical and horizontal displacement fields over a mine panel:

- **Spatial profile** `S(x)`: Gaussian bell with half-width determined by depth and angle-of-draw (35° for Indian coal measures).
- **Time profile** `f(t)`: Two-phase — rapid active phase (70–90% of S_max over 30–90 days) + exponential residual tail (τ = 180–365 days).
- **Parameters** sampled per panel: depth 100–600 m, width 100–400 m, seam thickness 2–4 m, subsidence factor 0.6–0.9.

#### `noise_injection.py` — Sensor Noise & Confounders
Every noise term traces to a named source:

| Noise source | Parameters | Source |
|---|---|---|
| MEMS sensor noise | ±0.15° RMS | InvenSense MPU-6050 datasheet (400 µg/√Hz @ 260 Hz BW) |
| Temperature drift | 0.0029°/°C | Wi-GIM field calibration study |
| Blast vibration | 0.5–1.0 g RMS, 15–50 Hz, 1–3/day | DGMS India ground vibration standards |
| Rain creep tilt | 0.01–0.08°, 3–7 day reversal | Geotechnical monitoring literature |
| Packet dropout | 5–15% burst loss | LoRa mesh characterization |

**Key confounder physics:**
- **Blast transients**: High vibration (100× baseline), small *non-persistent* tilt perturbation that decays within 1 window. Zero cumulative effect.
- **Rain creep**: Tilt ramps up during rainfall (half-sine), then *partially reverses* exponentially as soil dries (20–40% permanent residual). The reversal is what makes it distinguishable from subsidence.

#### `generate_dataset.py` — Orchestrator
Generates 18 synthetic mine panels (12 with subsidence, 6 stable) × 180 simulated days × 8–11 sensor nodes each. Writes:
- `data/raw/synthetic/panel_XXX/nodeNN.csv` — raw sensor timeseries
- `data/raw/synthetic/panel_XXX/panel_meta.json` — panel geometry
- `data/labels/window_labels.csv` — per-window ground-truth labels

**Rain event rate** (after accuracy improvement): `[3, 5]` events/month per panel, affecting `[0.4, 0.7]` of nodes. This produces ~23% rain_creep windows in training (was 6.6% originally). Physics-grounded generation was chosen over SMOTE because the features are physically coupled — interpolating `vib_rms`, `cross_node_corr`, and `strain_cumulative` independently can produce implausible feature combinations.

---

### 2. Preprocessing & Feature Engineering

**Files:** `src/preprocessing/`

#### `filtering.py`
- Resamples each node's CSV to a uniform 1-hour grid (forward-filling gaps ≤ 6 hours)
- Removes invalid sensor readings (NaN, out-of-range)
- Compensates temperature-induced tilt drift (slope: 0.0029°/°C, reference = median temperature)

#### `feature_engineering.py` — 28 Features
Features are grouped into 8 categories:

| # | Group | Features | Rationale |
|---|-------|----------|-----------|
| 1 | **Tilt (short)** | `tilt_current`, `tilt_rate_1h`, `tilt_mean/std_6h`, `tilt_mean/std_24h` | Instantaneous + short-term tilt dynamics |
| 2 | **Vibration** | `vib_rms`, `vib_peak`, `vib_dom_freq`, `vib_spectral_ratio` | Frequency content separates blast (>10 Hz) from creep (<10 Hz) |
| 3 | **Strain** | `strain_current`, `strain_rate_1h`, `strain_cumulative` | Cumulative drift is monotonic in subsidence |
| 4 | **Crack proxy** | `crack_proxy_current`, `crack_proxy_cumulative` | \|d(strain)/dt\| — cumulative value is a subsidence severity indicator |
| 5 | **Tilt reversal** *(new)* | `tilt_reversal_72h`, `tilt_delta_72h` | Rain creep reverses sign; subsidence never does. Ratio of 72h delta to 168h delta captures this. |
| 6 | **Multi-day rolling** *(new)* | `tilt_mean/std_72h`, `tilt_mean/std_168h`, `tilt_monotonicity` | Rain events span days — 24h window misses the full lifecycle. Monotonicity fraction: subsidence ≈ 0.7–0.9, rain ≈ 0.5. |
| 7 | **Vibration persistence** *(new)* | `vib_rms_mean_24h`, `vib_elevated_hours` | Blasts are 1–2 window spikes; sustained elevated vibration implies a different source |
| 8 | **Strain reversal** *(new)* | `strain_rate_24h`, `strain_reversal_72h` | Mirrors tilt reversal logic for strain sensor |
| — | **Cross-node** | `cross_node_corr` | 72h Pearson correlation with neighbouring nodes. Subsidence = coherent propagating front (high corr); blast = independent tilt perturbations (low corr); rain = patchy (moderate) |
| — | **Environment** | `temp_c` | Temperature for drift context |

The cross-node correlation window was increased from 24h to **72h** to better capture sustained spatial coherence patterns.

#### `pipeline.py` — Full Orchestrator
1. Processes all 18 panels: filter → engineer features → merge labels
2. Splits **by panel ID** (not by row) to prevent geometry leakage: 13 train / 3 val / 2 test panels
3. Fits a `StandardScaler` on training data only; applies to val/test
4. Writes `features_train.parquet`, `features_val.parquet`, `features_test.parquet`
5. Saves fitted scaler to `models/scaler.pkl` for production inference

---

### 3. Models

**Files:** `src/models/`

#### `isolation_forest.py` — Anomaly Detector
- Trained **only on normal-class windows** from the training split
- Contamination parameter tuned to match the actual anomaly rate on the validation set
- Outputs a normalised anomaly score ∈ [0, 1] (1 = maximally anomalous)
- 150 trees, max_features=0.8 — model size: **1.18 MB**

#### `xgboost_classifier.py` — 4-Class Classifier
- Multi-class gradient boosted trees (`multi:softprob` objective)
- Subsidence tiers collapsed: `subsidence_watch/warning/critical → subsidence`
- Sample weights computed per-class (`n_samples / (n_classes × n_class_samples)`) for balanced training
- Uses early stopping on validation `mlogloss` (patience = 25 rounds)

**Hyperparameters (tuned for i3-class edge CPU + 5 MB size limit):**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 140 | Headroom vs 5 MB limit allows 140 at depth 8 |
| `max_depth` | 8 | Deeper than baseline to capture rain_creep subtleties |
| `learning_rate` | 0.05 | Lower rate + more trees for better generalization |
| `tree_method` | `hist` | O(bins × features) — mandatory for fast i3-CPU training |
| `subsample` | 0.8 | Row sampling for ensemble diversity |
| `colsample_bytree` | 0.8 | Feature sampling per tree |
| `min_child_weight` | 5 | Prevents splits on tiny leaf groups |

Model size: **4.82 MB** (under 5 MB edge-deployment limit).

#### `risk_fusion.py` — Risk Score Fusion
Combines the two model outputs into a transparent 0–100 risk score:

```
risk_score = 100 × (0.3 × IF_anomaly_score + 0.7 × XGB_subsidence_prob)
```

| Risk Score | Tier |
|-----------|------|
| < 25 | Safe |
| 25–49 | Watch |
| 50–74 | Warning |
| ≥ 75 | Critical |

Thresholds are configurable in `config.yaml`. The 0.7 XGB weight reflects that the classifier has been explicitly trained to discriminate subsidence from confounders, while the IF provides an unsupervised signal for novel anomaly patterns not seen in training.

#### `random_forest.py` — Legacy
The original classifier. Replaced by XGBoost because the RF serialized to 6–12 MB, exceeding the 5 MB edge limit. Kept for reference.

---

### 4. Training

**File:** `src/training/train.py`

```bash
cd backend
python src/training/train.py
```

Pipeline:
1. Loads train/val Parquet splits
2. Trains Isolation Forest on normal-only windows; validates anomaly score separation
3. Trains XGBoost with early stopping on validation mlogloss
4. Logs per-class recall on validation set
5. Saves models to `backend/models/`
6. Asserts model size ≤ 5 MB (hard edge-deployment constraint)

**Prerequisites:** Run `generate_dataset.py` then `pipeline.py` first.

---

### 5. Evaluation

**File:** `src/evaluation/evaluate.py`

```bash
cd backend
python src/evaluation/evaluate.py
```

Outputs:
- Console: accuracy, false-alarm rate, full classification report, risk score distribution
- `models/confusion_matrix.png` — labelled confusion matrix
- `models/feature_importance.png` — top-10 XGBoost gain-based feature importance
- `models/evaluation_report.md` — markdown report

**False-alarm rate** is the primary safety metric: fraction of blast/rain windows misclassified as subsidence. At 0.79%, fewer than 1 in 125 confounder windows triggers a false alert.

---

## Model Performance

Results on 2 held-out test panels (panel_017, panel_018) — **never seen during training**.

### Improvement Progression

| Round | Changes | Test Accuracy | rain_creep F1 | False-Alarm Rate | Model Size |
|-------|---------|:---:|:---:|:---:|:---:|
| Baseline | 17 features, 80 trees, depth 6 | 68.6% | 0.24 | 1.72% | 1.15 MB |
| Round 1 | +11 features + hyperparameter tuning | 76.9% | 0.36 | 0.68% | 4.38 MB |
| **Round 2** | **+ physics-grounded rain_creep enrichment** | **75.1%** | **0.62** | **0.79%** | **4.82 MB** |

> Round 2 raw accuracy is lower than Round 1 because the test set itself changed: rain_creep grew from 10.2% → 22.8% of test windows, making the problem harder. Per-class metrics show the real gain — rain_creep F1 nearly tripled.

### Final Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| subsidence | 0.84 | **1.00** | **0.91** | 12,041 |
| normal | 0.84 | 0.71 | 0.77 | 51,578 |
| blast_transient | 0.69 | 0.86 | 0.77 | 9,708 |
| rain_creep | 0.57 | 0.67 | **0.62** | 21,713 |

### Risk Score Separation

| True Class | Mean | Median |
|-----------|------|--------|
| subsidence | 88.3 | 88.6 |
| blast_transient | 14.0 | 13.4 |
| normal | 8.3 | 4.5 |
| rain_creep | 5.9 | 4.7 |

The risk fusion layer maintains clean separation between subsidence (Critical zone) and all confounders (Safe zone), even when the classifier confuses normal vs rain_creep at the classification boundary.

---

## Live Simulation Server

**File:** `src/server.py`

FastAPI + WebSocket server that runs the trained models live. Loads all three models at startup, then ticks every 3 simulated seconds (advancing 4 simulated hours per tick).

```bash
cd backend
python -m uvicorn src.server:app --host 0.0.0.0 --port 8000
# or with auto-reload during development:
python -m uvicorn src.server:app --reload --port 8000
```

**Per-tick pipeline:**
1. Advances physics simulation (subsidence displacement, tilt, strain)
2. Injects MEMS noise + temperature drift
3. Extracts the same 28 features as the offline pipeline
4. Runs Isolation Forest + XGBoost inference
5. Computes risk score and tier via `risk_fusion.py`
6. Broadcasts `node_update` WebSocket messages to all connected clients

**WebSocket message types:**

| Type | Payload |
|------|---------|
| `node_update` | `{node_id, lat, lng, tilt_deg, vibration_rms, strain_mm, risk_score, risk_tier, label}` |
| `alert` | `{node_id, tier, message, timestamp}` |
| `system_status` | `{active_nodes, excluded_nodes, gateway_status}` |

**Simulation geometry:** Jharia Coalfield, Dhanbad (23.74°N, 86.41°E). Nodes placed on a configurable grid (default 100 m spacing).

---

## Frontend Dashboard

**Stack:** React 18 + Vite + Tailwind CSS + Leaflet + Recharts

```bash
cd frontend
npm install
npm run dev   # starts on http://localhost:5173
```

**Components:**

| Component | Description |
|-----------|-------------|
| `MapView.jsx` | Leaflet map with colour-coded risk tier markers. Clicking a node selects it. |
| `NodeGraph.jsx` | Recharts time-series showing tilt, vibration RMS, and strain for the selected node (last 60 data points) |
| `ControlPanel.jsx` | Node list with current risk tier badges; manual command interface |
| `AlertFeed.jsx` | Scrolling strip of real-time alerts, colour-coded by tier (Safe → Critical) |
| `socket.js` | WebSocket client with exponential backoff reconnection |

**Risk tier colour scheme:**

| Tier | Colour |
|------|--------|
| Safe | Green (`risk-safe`) |
| Watch | Amber (`risk-watch`) |
| Warning | Orange (`risk-warning`) |
| Critical | Red (`risk-critical`) |

---

## Configuration

All parameters live in `backend/configs/config.yaml`. Key sections:

```yaml
dataset:
  n_panels: 18
  simulation_days: 180
  sampling_interval_hours: 1

noise:
  rain:
    events_per_month: [3, 5]        # physics-grounded rain_creep enrichment
    affected_node_fraction: [0.4, 0.7]

models:
  xgboost:
    n_estimators: 140
    max_depth: 8
    learning_rate: 0.05
  risk_fusion:
    if_weight: 0.3
    rf_weight: 0.7                   # key name kept for compatibility
    tier_thresholds:
      safe: 25
      watch: 50
      warning: 75
  max_model_size_mb: 5               # hard edge-deployment constraint
```

---

## Setup & Running

### Prerequisites
- Python ≥ 3.10
- Node.js ≥ 18

### Backend

```bash
cd backend
pip install -r requirements.txt

# 1. Generate synthetic dataset (~2 min)
python src/data_generation/generate_dataset.py

# 2. Preprocess and extract features (~3 min)
python src/preprocessing/pipeline.py

# 3. Train models (~2 min)
python src/training/train.py

# 4. Evaluate (optional — view confusion matrix + report)
python src/evaluation/evaluate.py

# 5. Start live server
python -m uvicorn src.server:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The dashboard connects automatically to `ws://localhost:8000/ws`.

### Full pipeline in one shot

```powershell
cd backend
python src/data_generation/generate_dataset.py
python src/preprocessing/pipeline.py
python src/training/train.py
python -m uvicorn src.server:app --port 8000
```

---

## References

- CMRI Technical Manual on Mine Subsidence (1997)
- IS 14562:1998 — Indian Standard for Subsidence Prediction
- Chatterjee et al. (2015), *J. Earth System Science* — GPS deformation, Jharia Coalfield
- Dey et al. (2018) — InSAR subsidence mapping, Raniganj Coalfield
- Djamaluddin et al. (2011), *Int. J. Mining Sci. & Technology*
- DGMS India Circular on Ground Vibration Limits
- InvenSense MPU-6050 Product Specification Rev 3.4
- Wi-GIM field calibration study — MEMS tilt sensor temperature drift
