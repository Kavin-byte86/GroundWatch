
# GroundWatch
AI-enabled low-cost wireless surface mesh for real-time subsidence monitoring, prediction &amp; early warning in underground coal mines. ESP32 + LoRa sensor nodes detect tilt, vibration &amp; strain; Isolation Forest anomaly detection + GIS dashboard alerts mine operators.

## About GroundWatch

Surface subsidence caused by underground coal mining threatens nearby communities, infrastructure, and agricultural land across Indian coalfields. Current monitoring in India relies mainly on periodic manual surveys and satellite InSAR studies — both useful, but neither provides real-time, continuous warning before ground failure occurs. InSAR revisits a site only once every 6–12 days and cannot trigger an instant alert; manual surveys happen weeks or months apart.

**GroundWatch fills that gap** with a low-cost, real-time early-warning layer deployed directly on the surface above underground mine panels — not a replacement for existing methods, but the continuous ground-truth layer between them.

### What it does

A wireless mesh of solar-powered sensor nodes is deployed across the surface land above an underground coal panel. Each node continuously measures ground tilt, vibration, and inter-node strain, and transmits readings over a long-range LoRa mesh to a gateway. Machine learning models running at the edge classify incoming data in real time — separating genuine subsidence signatures from false positives like blasting vibration or rain-driven soil creep — and raise a graded Safe/Watch/Warning/Critical alert the moment ground movement crosses a validated threshold. Alerts and live deformation status are shown on a GIS dashboard for mine operators, safety officers, and regulators, with automated SMS/email notification on higher-severity events.

The core differentiator: commercial systems like Worldsensing or Senceive cost $500–3,000 per node, making dense coverage across India's underground mines financially unworkable at scale. GroundWatch is built on off-the-shelf, widely available hardware to bring per-node cost down to roughly ₹2,500–6,000 — enabling the kind of dense, affordable deployment that makes real-time monitoring practical across many mines, not just a handful of high-value sites.

### Tech Stack

**Hardware:** ESP32 (node controller) · SX1276/RFM95 LoRa module · MPU6050 (tilt) · ADXL345 (vibration) · HX711 + strain gauge (displacement) · NEO-6M GNSS (sparse reference nodes) · 5W solar + Li-ion/LiFePO4 + MPPT

**Firmware/Comms:** C/C++ (Arduino/ESP-IDF) · LoRa mesh (865–867 MHz India ISM band) · packed binary payloads for low-power transmission

**Backend:** Python · FastAPI · WebSocket streaming · SQLite (edge buffer)

**ML/AI:** scikit-learn (Isolation Forest — unsupervised anomaly detection) · XGBoost (multi-class severity classification, `tree_method="hist"` for CPU/edge inference) · physics-informed synthetic dataset generated from the CMRI influence-function subsidence model

**Frontend:** React · Vite · Tailwind CSS · Recharts (live sensor graphs) · Leaflet.js (GIS map view)

**Alerts:** SMS/email/push notification pipeline for tiered escalation to mine safety personnel


<img width="2752" height="1536" alt="Gemini_Generated_Image_f9v7shf9v7shf9v7" src="https://github.com/user-attachments/assets/2d40b978-2c3c-420c-b40e-173de8e6421e" />
