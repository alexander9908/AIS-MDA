Here’s your complete README.md — you can copy it directly or find it at
# AIS-MDA — Increasing Maritime Domain Awareness with Spatio-Temporal Models

This repository provides a **ready-to-run** pipeline for working with real, irregular, and noisy **AIS** (Automatic Identification System) data to solve three key tasks:
- **Trajectory prediction** — forecast future vessel positions (sequence-to-sequence).
- **Anomaly detection** — detect abnormal vessel behavior using self-supervised learning.
- **Port call & ETA prediction** — classify the next port and estimate time of arrival.

It includes:
- Modular data loading, cleaning, and segmentation.
- Feature engineering (Δt, Δx/Δy, rate of turn, acceleration).
- Label generation for trajectory, anomaly, and ETA tasks.
- Baseline models (Constant Velocity, GRU, LSTM) and **TPTrans** (CNN + Transformer hybrid).
- Evaluation metrics (ADE, FDE, MAE, MAPE, AUROC, etc.).
- Ready-made configs, environment setup, and Docker support.

---

## 📁 Project Layout

ais-mda/
├── README.md
├── env/
│   ├── environment.yml
│   └── Dockerfile
├── configs/
├── scripts/
└── src/

---


## 🚢 Background

AIS messages contain:
- Vessel ID (MMSI)
- Timestamp
- Latitude, Longitude
- Speed over ground (SOG)
- Course over ground (COG)
- Heading
- Navigational status
- Vessel type, draught, destination (optional)

These form **spatio-temporal trajectories** representing vessel movement patterns.

Real AIS is **irregular**, **noisy**, and **error-prone** — making it ideal for testing robust sequential models.

---

## 🎯 Objectives

1. Build a robust preprocessing pipeline for noisy, irregular AIS data.  
2. Develop baseline and advanced sequence models for:
   - Vessel trajectory forecasting.
   - Vessel anomaly detection (self-supervised).
   - Port arrival and ETA classification/regression.
3. Benchmark models using relevant spatio-temporal metrics.
4. Evaluate performance on real data and demonstrate maritime use cases.

---

## 📚 Key References

These research papers guide our methodology:

1. **Artificial Intelligence in Ship Trajectory Prediction** (2024)  
   → Survey of ML and DL models for vessel trajectory forecasting, evaluation methods, and data preprocessing best practices.  
   *(Provides overall taxonomy and baseline models.)*

2. **TPTrans: Vessel Trajectory Prediction Model Based on CNN and Transformer** (2023)  
   → Introduces *TPTrans*, combining convolutional (local) and transformer (global) layers for superior trajectory accuracy, especially in turning segments.  
   *(Inspires our hybrid model implementation.)*

3. **Vessel Trajectory Prediction with Deep Learning Techniques** (JMSE, 2025)  
   → Evaluates Bi-LSTM, GRU, and Transformer models with real AIS data, showing preprocessing, segmentation, and horizon-dependent accuracy trends.  
   *(Used as baseline validation and feature engineering guide.)*

4. **Prediction of Vessel Arrival Time to Port — A Review of Current Studies** (2025)  
   → Reviews ETA prediction literature; defines key feature groups (vessel, route, environment, external) and performance metrics (MAE, MAPE, P95).  
   *(Defines the design for port-call and ETA subtask.)*

---

## 🧩 Project Structure

ais-mda/
├── README.md
├── env/
│   ├── environment.yml
│   └── Dockerfile
├── data/
│   ├── raw/          # Raw AIS data (CSV or Parquet)
│   ├── interim/      # Cleaned, gap-split trajectories
│   └── processed/    # Windowed sequences for model training
├── notebooks/
│   ├── 00_explore_ais.ipynb
│   ├── 10_clean_segment.ipynb
│   └── 20_train_baselines.ipynb
├── src/
│   ├── dataio/       # loaders and cleaning
│   ├── features/     # feature engineering (Δt, Δx, ROT, accel)
│   ├── labeling/     # trajectory, ETA, anomaly label creation
│   ├── models/       # GRU/LSTM, TPTrans, etc.
│   ├── train/        # task-specific training scripts
│   ├── eval/         # metric computation (ADE, FDE, MAE, etc.)
│   └── utils/        # geospatial and batching helpers
├── configs/          # YAML configs for experiments
└── scripts/          # CLI automation for preprocessing/training

### Expanded Project Structure
ais-mda/
├── README.md
├── env/                         # environment & docker
│   ├── environment.yml
│   └── Dockerfile
├── data/
│   ├── raw/                     # original AIS dumps (parquet/csv)
│   ├── interim/                 # cleaned segments
│   └── processed/               # windowed tensors / features
├── notebooks/
│   ├── 00_explore_ais.ipynb
│   ├── 10_build_segments.ipynb
│   └── 20_train_baselines.ipynb
├── src/
│   ├── config.py
│   ├── dataio/
│   │   ├── load_ais.py          # robust loader (csv/parquet)
│   │   ├── clean.py             # QC, outlier rules, denoise
│   │   └── segment.py           # trajectory splits, resampling (optional)
│   ├── features/
│   │   ├── kinematics.py        # Δt, Δx/Δy, ROT, accel, etc.
│   │   └── context.py           # cells, port proximity, route features
│   ├── labeling/
│   │   ├── traj_labels.py       # next-K deltas
│   │   ├── eta_labels.py        # next port + true ETA
│   │   └── anomalies.py         # synthetic anomalies for eval
│   ├── models/
│   │   ├── kinematic.py         # CV/CTRV baselines (and EKF wrapper)
│   │   ├── rnn_seq2seq.py       # LSTM/BiLSTM/GRU baselines
│   │   └── tptrans.py           # CNN+Transformer (TPTrans-style)
│   ├── train/
│   │   ├── train_traj.py
│   │   ├── train_eta.py
│   │   └── train_anom.py
│   ├── eval/
│   │   ├── metrics_traj.py      # ADE, FDE, DFD/Hausdorff
│   │   ├── metrics_eta.py       # MAE, MAPE, P95
│   │   └── metrics_anom.py      # AUROC, AUPRC, TTD
│   └── utils/
│       ├── geo.py               # proj, haversine, UTM helpers
│       └── batching.py          # masking, padding
├── configs/
│   ├── traj_gru_small.yaml
│   ├── traj_tptrans_base.yaml
│   ├── eta_gru.yaml
│   └── anom_masked.yaml
└── scripts/
    ├── make_interim.sh
    ├── make_processed.sh
    └── train.sh

---

## 🧪 Workflow Overview

### 1. Data Preparation
- Download AIS data (e.g., NOAA, MarineCadastre, Global Fishing Watch, or regional sources).  
- Clean and preprocess:
  - Remove invalid or duplicate points.
  - Split trajectories on large time gaps.
  - Compute derived kinematic features (Δlat, Δlon, Δt, ROT, acceleration, etc.).
  - Retain irregular sampling or interpolate moderately (≤60 s).

### 2. Feature Engineering
- Encode course (COG) as sin/cos.
- Compute H3 cell index or UTM grid for spatial context.
- Optional: Add distance & bearing to nearest port (for ETA).

### 3. Task Labeling
- **Trajectory** → predict next *K* points (Δx, Δy).
- **Anomaly** → self-supervised, using reconstruction or forecast error.
- **Port/ETA** → classify next port and regress ETA using port polygons (e.g., NGA WPI).

### 4. Modeling
- **Baselines:** Constant velocity, Kalman filter, Bi-LSTM, GRU.
- **Advanced:** CNN+Transformer hybrid (TPTrans).
- **Optional:** Self-supervised pretraining for anomalies.

### 5. Evaluation
| Task | Key Metrics | Notes |
|------|--------------|-------|
| Trajectory | ADE, FDE, Hausdorff | Compare across horizons |
| ETA | MAE, MAPE, P95 | Compare against naive baseline |
| Anomaly | AUROC, AUPRC, TTD | Test with planted anomalies |

### 6. Visualization
- Plot true vs predicted trajectories.
- Show horizon-based accuracy decay.
- Plot ETA error distributions.

## 🚀 Quickstart

### 0) Create environment

Using Conda or Mamba:
```bash
conda env create -f env/environment.yml
conda activate ais

With Docker (GPU optional):

docker build -t ais-mda . -f env/Dockerfile
docker run --rm -it -v "$(pwd)":/workspace -w /workspace ais-mda bash
```

⸻

1) Prepare interim dataset

(Clean → Segment → Feature Engineering)
```bash
bash scripts/make_interim.sh \
  --raw data/raw/*.parquet \
  --out data/interim \
  --gap_hours 6 --max_sog 40
```
This script:
	•	Loads AIS CSV/Parquet files.
	•	Cleans invalid coordinates, speeds, and duplicates.
	•	Splits trajectories on large time gaps.
	•	Adds kinematic features (Δt, Δx, Δy, accel, COG sin/cos).
	•	Adds grid-cell context (cell_id).

Output:
data/interim/interim.parquet

⸻

2) Build processed tensors for model training

Trajectory task
```bash
bash scripts/make_processed.sh \
  --interim data/interim/interim.parquet \
  --task trajectory --window 64 --horizon 12 \
  --out /mnt/data/processed/traj_w64_h12
```

ETA task
```bash
bash scripts/make_processed.sh \
  --interim data/interim/interim.parquet \
  --task eta --window 64 \
  --out /mnt/data/processed/eta_w64
```

Anomaly task
```bash
bash scripts/make_processed.sh \
  --interim data/interim/interim.parquet \
  --task anomaly --window 64 --horizon 12 \
  --out /mnt/data/processed/anom_w64_h12
```

⸻

3) Train models

GRU baseline (trajectory)

```bash
python -m src.train.train_traj --config configs/traj_gru_small.yaml

TPTrans hybrid (CNN + Transformer)

python -m src.train.train_traj --config configs/traj_tptrans_base.yaml

ETA prediction (GRU)

python -m src.train.train_eta --config configs/eta_gru.yaml
```

⸻

🧠 Features
	•	Handles irregular sampling (keeps Δt as input).
	•	Computes local kinematics (Δx, Δy, speed change, turn rate).
	•	Produces windowed sequences for supervised or self-supervised training.
	•	Flexible model config (via YAML) and plug-and-play architecture.
	•	Metrics: ADE, FDE (trajectory), MAE/MAPE/P95 (ETA), AUROC/AUPRC (anomaly).

⸻

🧩 Model Zoo

Model	Type	Use Case	Description
Constant Velocity	Baseline	Trajectory	Predicts next points by last velocity
GRUSeq2Seq	RNN	Trajectory / ETA	Encoder-decoder GRU
TPTrans	CNN + Transformer	Trajectory	Local + global spatio-temporal model
GRU Forecaster	RNN	Anomaly	Self-supervised reconstruction/forecasting

⸻

⚙️ Data Expectations

Minimum columns:

mmsi, timestamp, lat, lon

Recommended:

sog, cog, heading, nav_status, shiptype, draught, destination

⸻

📈 Metrics

Task	Metrics	Notes
Trajectory	ADE, FDE, Hausdorff	Position accuracy per horizon
ETA	MAE, MAPE, P95	Time-of-arrival accuracy
Anomaly	AUROC, AUPRC, TTD	Detection accuracy and latency

Metric	Description
ADE	Average Displacement Error — mean L2 distance between predicted and true positions.
FDE	Final Displacement Error — distance at final predicted step.
MAE / MAPE	Mean (Absolute) Error / Mean Absolute Percentage Error for ETA.
P95	95th percentile ETA error.
AUROC / AUPRC	Anomaly detection quality.
TTD	Time-to-detection (anomaly detection latency).

⸻

🧠 Insights from the Literature

Insight	Source
RNNs (Bi-LSTM/GRU) perform best for short- to mid-term predictions.	JMSE 2025
CNN + Transformer improves turning and long-horizon accuracy.	TPTrans 2023
ETA accuracy depends on vessel dynamics, route features, and environment.	ETA Review 2025
Data cleaning, segmentation, and feature engineering strongly affect model quality.	AI in Ship Trajectory Prediction 2024

⸻

🏁 Deliverables
	1.	Clean AIS dataset with feature and label sets.
	2.	Baseline + TPTrans model checkpoints.
	3.	Evaluation report (tables, figures, ablation results).
	4.	Final presentation: “Deep Learning for Maritime Domain Awareness”.

⸻

📚 References
	1.	Artificial Intelligence in Ship Trajectory Prediction (2024) — Overview of ML and DL techniques for AIS trajectory modeling.
	2.	TPTrans: Vessel Trajectory Prediction Model Based on CNN and Transformer (2023) — CNN + Transformer hybrid for maritime motion forecasting.
	3.	Vessel Trajectory Prediction with Deep Learning Techniques (JMSE, 2025) — Empirical benchmark of RNNs and Transformers on AIS data.
	4.	Prediction of Vessel Arrival Time to Port — A Review of Current Studies (2025) — ETA modeling review and factor taxonomy.

⸻

🛠️ Troubleshooting
	•	CUDA not found → comment out pytorch-cuda in env/environment.yml for CPU use.
	•	Column mismatch → adjust src/dataio/load_ais.py or rename columns in raw data.
	•	Insufficient data → reduce window or horizon in YAML configs.
	•	Permissions → make sure scripts are executable:

chmod +x scripts/*.sh

⸻

🧩 Next Steps
	•	Add weather or sea condition data for ETA enhancement.
	•	Implement masked-step pretraining for anomaly detection.
	•	Integrate Graph Attention for vessel proximity interactions.
	•	Deploy inference API using FastAPI or Flask.

⸻

📖 Citation

If you build on this work, cite the corresponding research papers above.

⸻

Author: Group 11 — 2025
Location: Copenhagen, Denmark

---

/Users/alexanderschiotz/Desktop/DTU/Master/Deep Learning/Projects/ais-mda