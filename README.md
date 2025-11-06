
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
│   ├── requirements.txt
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
│   ├── requirements.txt
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

#### Trajectory Evaluation Plots

Use the dedicated evaluation helper to score checkpoints and generate Europe-context plots:

- auto-zoom on actual (recommended):
```bash
python -m src.eval.eval_traj_newnewnew \
  --split_dir data/map_reduced/val \
  --ckpt data/checkpoints/traj_tptrans.pt \
  --model tptrans \
  --lat_idx 0 --lon_idx 1 \
  --y_order latlon \
  --past_len 64 --max_plots 8 \
  --out_dir data/figures \
  --auto_extent --extent_source actual --extent_outlier_sigma 3.0 \
  --denorm --lat_min 54 --lat_max 58 --lon_min 6 --lon_max 16
```
- full Europe view:
```bash
python -m src.eval.eval_traj_newnewnew \
  --split_dir data/map_reduced/val \
  --ckpt data/checkpoints/traj_tptrans.pt \
  --model tptrans \
  --lat_idx 0 --lon_idx 1 \
  --y_order latlon \
  --past_len 64 --max_plots 8 \
  --out_dir data/figures \
  --denorm --lat_min 54 --lat_max 58 --lon_min 6 --lon_max 16
```

displays a full plot of the true past path, true future path and the predicted future path.


---

## 🚀 Quickstart

### 0) Create Environment

Using **Python venv + pip** (recommended):

Using VS Code:
Open command palette:
- Ctrl+Shift+P
- Command+Shift+P

Python Create environment → Select venv → Choose requirements.txt

Or via terminal:
```bash
# Create virtual environment
python3 -m venv .venv

# Activate it (macOS/Linux)
source .venv/bin/activate

# or on Windows
.venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r env/requirements.txt

With Docker (GPU optional):

docker build -t ais-mda -f env/Dockerfile .
docker run --rm -it -v "$(pwd)":/workspace -w /workspace ais-mda bash
```

⸻

1) Prepare Interim Dataset

(Clean → Segment → Feature Engineering)

```bash
bash scripts/make_interim.sh \
  --raw data/raw/*.csv \
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
```bash
data/interim/interim.parquet
```
⸻

2) Build Processed Tensors for Model Training

Trajectory task
```bash
bash scripts/make_processed.sh \
  --interim data/interim/interim.parquet \
  --task trajectory --window 64 --horizon 12 \
  --out data/processed/traj_w64_h12
```
ETA task
```bash
bash scripts/make_processed.sh \
  --interim data/interim/interim.parquet \
  --task eta --window 64 \
  --out data/processed/eta_w64
```
Anomaly task
```bash
bash scripts/make_processed.sh \
  --interim data/interim/interim.parquet \
  --task anomaly --window 64 --horizon 12 \
  --out data/processed/anom_w64_h12
```

⸻

3) Train Models

GRU baseline (trajectory)
```bash
python -m src.train.train_traj --config configs/traj_gru_small.yaml
```
TPTrans hybrid (CNN + Transformer)
```bash
python -m src.train.train_traj --config configs/traj_tptrans_base.yaml
```
ETA prediction (GRU)
```bash
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

Metric descriptions
	•	ADE: Average Displacement Error — mean L2 distance between predicted and true positions.
	•	FDE: Final Displacement Error — distance at final predicted step.
	•	MAE / MAPE: Mean (Absolute) Error / Mean Absolute Percentage Error for ETA.
	•	P95: 95th percentile ETA error.
	•	AUROC / AUPRC: Anomaly detection quality.
	•	TTD: Time-to-detection (anomaly detection latency).

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
	4.	Final presentation: “Deep Learning for Maritime Domain Awareness”

⸻

📚 References
	1.	Artificial Intelligence in Ship Trajectory Prediction (2024) — Overview of ML and DL techniques for AIS trajectory modeling.
	2.	TPTrans: Vessel Trajectory Prediction Model Based on CNN and Transformer (2023) — CNN + Transformer hybrid for maritime motion forecasting.
	3.	Vessel Trajectory Prediction with Deep Learning Techniques (JMSE, 2025) — Empirical benchmark of RNNs and Transformers on AIS data.
	4.	Prediction of Vessel Arrival Time to Port — A Review of Current Studies (2025) — ETA modeling review and factor taxonomy.

⸻

🛠️ Troubleshooting
	•	CUDA not found → Install CPU-only PyTorch or remove CUDA wheels from requirements.txt.
	•	Column mismatch → Adjust src/dataio/load_ais.py or rename columns in raw data.
	•	Insufficient data → Reduce window or horizon in YAML configs.
	•	Permissions → Make sure scripts are executable:
```bash
chmod +x scripts/*.sh
```


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
