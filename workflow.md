# WORKFLOW.md — AIS-MDA End-to-End Guide

This document explains how to run the project step-by-step on a fresh machine using a Python virtual environment (no conda required). It also covers how data flows through the pipeline, what each script does, where files are written, and how to evaluate results.

⸻

14) Nested Cross-Validation (SGD + Hyperparameter Tuning)

This project includes a **2‑level (nested) cross‑validation** script that performs:
- **Outer folds** (GroupKFold by **MMSI**) to estimate generalization (prevents vessel leakage).
- **Inner folds** (GroupKFold by MMSI on the outer‑train) to tune **SGD** hyperparameters.

### 14.1 Install extra dependency
Make sure scikit‑learn is available:
```bash
pip install scikit-learn>=1.3
```

If it’s not already in your `env/requirements.txt`, add it and reinstall:
```bash
echo "scikit-learn>=1.3" >> env/requirements.txt
pip install -r env/requirements.txt
```

### 14.2 Ensure processed data contains split metadata
Rebuild trajectory tensors so `window_mmsi.npy` and `target_scaler.npz` are present:
```bash
bash scripts/make_processed.sh \
  --interim data/interim/interim.parquet \
  --task trajectory --window 64 --horizon 12 \
  --out data/processed/traj_w64_h12
```

### 14.3 Run nested CV (TPTrans example)
```bash
python -m src.train.nested_cv_traj \
  --processed_dir data/processed/traj_w64_h12 \
  --model tptrans \
  --outer_folds 5 \
  --inner_folds 3 \
  --max_trials 8
```

**Parameters**
- `--model {tptrans,gru}`: choose the architecture to evaluate.
- `--outer_folds`: number of outer folds (test estimation).
- `--inner_folds`: number of inner folds (HP tuning on outer‑train).
- `--max_trials`: number of random hyperparameter settings to try.

**What it does**
- Uses **GroupKFold by MMSI** for both outer and inner splits.
- Trains with **SGD (momentum + Nesterov)** for a small number of epochs per trial.
- Standardizes **targets (Y)** if `target_scaler.npz` exists; predictions are unscaled before ADE/FDE.

**Output**
- Console shows per‑fold **ADE/FDE** and the **best hyperparameters** chosen on the inner loop.
- Final summary: `ADE mean±std` and `FDE mean±std` across outer folds.

### 14.4 Suggested next step: retrain with best HPs
After nested CV identifies promising hyperparameters, update your config (e.g., `configs/traj_tptrans_base.yaml`) with the winners (e.g., `d_model`, `nhead`, `enc_layers`, `batch_size`, `lr`, `momentum` if switching to SGD) and run a longer training (e.g., `epochs: 30–50`).

Example TPTrans retrain:
```bash
python -m src.train.train_traj --config configs/traj_tptrans_base.yaml
```

Then evaluate as usual (matching model ↔ checkpoint):
```bash
python -m src.eval.evaluate_traj \
  --processed_dir data/processed/traj_w64_h12 \
  --ckpt data/checkpoints/traj_tptrans.pt \
  --model tptrans \
  --plot
```

*Notes*
- Nested CV is compute‑intensive; start with fewer folds/trials for speed, then scale up.
- Keep `outer` folds **≥ 5** for a stable generalization estimate when dataset size allows.
- Always use **MMSI‑grouped** splits to avoid train/test leakage across windows of the same vessel.

0) Prerequisites
	•	Python 3.11+ (3.13 works; you’re using it already)
	•	pip (latest)
	•	(Optional) Docker if you prefer containerized runs
	•	(Optional) GPU + CUDA-enabled PyTorch (pip wheels)

⸻

1) Clone & Environment

# from the directory where you keep repos
```bash
git clone <your-repo-url> ais-mda
cd ais-mda
```

# create & activate virtualenv
```bash
python3 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
```

# install dependencies
```bash
pip install --upgrade pip
pip install -r env/requirements.txt
```

If you prefer Docker:
```bash
docker build -t ais-mda -f env/Dockerfile .
docker run --rm -it -v "$(pwd)":/workspace -w /workspace ais-mda bash
```


⸻

2) Project Layout (what’s where)

ais-mda/
├── README.md
├── WORKFLOW.md                     # this file
├── env/
│   ├── requirements.txt            # pip dependencies
│   └── Dockerfile                  # optional container
├── data/
│   ├── raw/                        # your raw AIS data (CSV or Parquet)
│   ├── interim/                    # cleaned + segmented parquet
│   ├── processed/                  # windowed tensors per task
│   └── figures/                    # plots produced during eval
├── configs/                        # YAML configs per task/model
│   ├── traj_gru_small.yaml
│   ├── traj_tptrans_base.yaml
│   ├── eta_gru.yaml
│   └── anom_masked.yaml
├── scripts/
│   ├── make_interim.sh/.py         # clean + segment + feature-engineer
│   ├── make_processed.sh/.py       # build X.npy/Y.npy (+ scaler, MMSIs)
│   ├── train.sh                    # example training invocations
│   └── eval.sh                     # example evaluation invocations
└── src/
    ├── config.py
    ├── dataio/                     # loading, cleaning, segmenting
    ├── features/                   # kinematics, grid/cell encodings
    ├── labeling/                   # window/label makers
    ├── models/                     # GRU, TPTrans, baselines
    ├── train/                      # training scripts (traj/eta/anom)
    ├── eval/                       # evaluation metrics + scripts
    └── utils/                      # geodesy, batching helpers


⸻

3) Data Preparation

3.1 Raw data

Place AIS files in data/raw/ as CSV or Parquet (minimum columns: mmsi, timestamp, lat, lon; recommended: sog, cog, heading, nav_status, shiptype, draught, destination).

You can also use the synthetic sample you generated:
```bash
data/raw/ais_sample.csv
```


3.2 Clean, segment, feature-engineer → interim.parquet
```bash
bash scripts/make_interim.sh \
  --raw data/raw/*.csv \
  --out data/interim \
  --gap_hours 6 \
  --max_sog 40
```

What this does
	•	Loads CSV/Parquet robustly (column normalization).
	•	Drops invalid points (lat/lon out of range, duplicate timestamps, unrealistic speed).
	•	Computes per-MMSI dt (seconds since previous point).
	•	Segments trajectories when time gap > gap_hours.
	•	Adds kinematic features: dx, dy, accel, cog_sin, cog_cos, etc.
	•	Adds grid/cell context (e.g., cell_id).

Output
	•	data/interim/interim.parquet

⸻

4) Build Processed Tensors (by task)

This converts interim.parquet into model-ready tensors (X.npy, Y.npy or y_eta.npy) and saves normalization and per-window MMSI mapping where relevant.

The processing script also saves:
	•	scaler.npz — feature mean/std (for normalization during train/eval)
	•	window_mmsi.npy — MMSI per window (enables vessel-wise splits)

4.1 Trajectory prediction (seq2seq on deltas)
```bash
bash scripts/make_processed.sh \
  --interim data/interim/interim.parquet \
  --task trajectory \
  --window 64 \
  --horizon 12 \
  --out data/processed/traj_w64_h12
```

Output
	•	data/processed/traj_w64_h12/X.npy — shape [N, 64, F]
	•	data/processed/traj_w64_h12/Y.npy — shape [N, 12, 2] (future dx,dy)
	•	data/processed/traj_w64_h12/scaler.npz — mean/std for X
	•	data/processed/traj_w64_h12/window_mmsi.npy — MMSI per window
	•	data/processed/traj_w64_h12/meta.json

4.2 ETA (time-to-port) baseline
```bash
bash scripts/make_processed.sh \
  --interim data/interim/interim.parquet \
  --task eta \
  --window 64 \
  --out data/processed/eta_w64
```

Output
	•	data/processed/eta_w64/X.npy     — shape [N, 64, F]
	•	data/processed/eta_w64/y_eta.npy — shape [N] (seconds to arrival; placeholder if true labels absent)
	•	data/processed/eta_w64/scaler.npz
	•	data/processed/eta_w64/meta.json

Notes:
	•	If time_to_port_sec exists in interim.parquet, the script uses it.
	•	Otherwise it fabricates a pseudo-label from future dt sum (for demo only).

4.3 Anomaly (self-supervised forecast)
```bash
bash scripts/make_processed.sh \
  --interim data/interim/interim.parquet \
  --task anomaly \
  --window 64 \
  --horizon 12 \
  --out data/processed/anom_w64_h12
```

Output
	•	data/processed/anom_w64_h12/X.npy, Y.npy
	•	scaler.npz, window_mmsi.npy, meta.json

⸻

5) Configure Training

Open the YAML in configs/ and confirm processed_dir matches your chosen output. Example:

configs/traj_tptrans_base.yaml
```yaml
task: trajectory
processed_dir: data/processed/traj_w64_h12   # <- make sure this matches
out_dir: data/checkpoints
window: 64
horizon: 12
features: [dx, dy, sog, cog_sin, cog_cos, accel, dt, cell_id]
model:
  name: tptrans
  d_model: 192
  nhead: 4
  enc_layers: 4
  dec_layers: 2
loss: huber
optimizer: adamw
lr: 0.0003
batch_size: 96
epochs: 5            # increase for better results, e.g., 30
val_frac: 0.2        # optional (row-wise split); for vessel-wise, see §7
```

configs/eta_gru.yaml
```yaml
task: eta
processed_dir: data/processed/eta_w64
out_dir: data/checkpoints
window: 64
features: [sog, cog_sin, cog_cos, accel, dt, dx, dy, dist_to_port, bearing_to_port]
model:
  name: gru
  d_model: 128
  layers: 2
loss: mae
optimizer: adam
lr: 0.001
batch_size: 256
epochs: 5            # increase to 20–40 once labels are real
```

⸻

6) Train

6.1 Trajectory (TPTrans or GRU)

# TPTrans
```bash
python -m src.train.train_traj --config configs/traj_tptrans_base.yaml
```

# GRU baseline
```bash
python -m src.train.train_traj --config configs/traj_gru_small.yaml
```

What happens
	•	Loads X.npy/Y.npy
	•	Applies normalization if scaler.npz exists
	•	Trains the model & saves checkpoint:
	•	data/checkpoints/traj_model.pt (best per epoch if val, else last)

6.2 ETA
```bash
python -m src.train.train_eta --config configs/eta_gru.yaml
```

What happens
	•	Loads X.npy & y_eta.npy
	•	Applies normalization if scaler.npz exists
	•	Trains a GRU + linear head and saves:
	•	data/checkpoints/eta_model.pt

Note: For meaningful ETA results, you’ll want true time_to_port_sec labels (see §10).

⸻

7) Evaluate & Visualize

Create the figures folder (if not created yet):

```bash
mkdir -p data/figures
```

7.1 Trajectory metrics & plots

# prints ADE/FDE and saves a few trajectory plots to data/figures/
```bash
python -m src.eval.evaluate_traj \
  --processed_dir data/processed/traj_w64_h12 \
  --ckpt data/checkpoints/traj_model.pt \
  --model tptrans \
  --plot
```

### IMPORTANT: Match checkpoint to model (GRU vs TPTrans)

Use a TPTrans checkpoint with `--model tptrans`, and a GRU checkpoint with `--model gru`. Loading a GRU checkpoint into TPTrans (or vice‑versa) will raise a state_dict key/size mismatch.

**TPTrans evaluation (generic TPTrans filename):**
```bash
python -m src.eval.evaluate_traj \
  --processed_dir data/processed/traj_w64_h12 \
  --ckpt data/checkpoints/traj_model.pt \
  --model tptrans \
  --plot
```

**TPTrans evaluation (explicit TPTrans filename):**
```bash
python -m src.eval.evaluate_traj \
  --processed_dir data/processed/traj_w64_h12 \
  --ckpt data/checkpoints/traj_tptrans.pt \
  --model tptrans \
  --plot
```

If your checkpoint was produced by the GRU trainer (e.g., `traj_gru_model.pt`), evaluate as GRU instead:
```bash
python -m src.eval.evaluate_traj \
  --processed_dir data/processed/traj_w64_h12 \
  --ckpt data/checkpoints/traj_model.pt \
  --model gru \
  --plot
```

*Tip:* If you’re unsure which model produced a checkpoint, open it and inspect the keys:
```python
import torch
sd = torch.load("data/checkpoints/<file>.pt", map_location="cpu")
print([k for k in sd.keys() if isinstance(k, str)][:10])
# GRU-style keys often include: 'enc.weight_ih_l0', 'enc.weight_hh_l0', ...
# TPTrans-style keys include: 'conv.net.0.weight', 'encoder.layers.0.self_attn.in_proj_weight', ...
```

Outputs
	•	VAL: ADE=..., FDE=...
	•	PNGs under data/figures/ (pred vs true trajectories)

Vessel-wise split (recommended):
If your processed_dir contains window_mmsi.npy, the evaluator will split by MMSI instead of rows (prevents data leakage).

7.2 ETA metrics
```bash
python -m src.eval.evaluate_eta \
  --processed_dir data/processed/eta_w64 \
  --ckpt data/checkpoints/eta_model.pt
```

Outputs
	•	VAL: MAE=..., MAPE=...%, P95=... (meaningful once real labels exist)

⸻

8) What the pieces do (quick map)
	•	scripts/make_interim.py:
Loads raw AIS → cleans → segments (by gap_hours) → adds kinematics & context → saves interim.parquet.
	•	scripts/make_processed.py:
Creates sliding windows:
	•	Trajectory/Anomaly: X features, Y future deltas; saves scaler.npz, window_mmsi.npy.
	•	ETA: X features, y_eta label; saves scaler.npz.
	•	src/models/:
	•	GRUSeq2Seq: GRU encoder-decoder baseline.
	•	TPTrans: CNN + Transformer encoder, GRU decoder hybrid.
	•	kinematic.py: classic baselines (e.g., Constant Velocity).
	•	src/train/train_traj.py:
Trains GRU/TPTrans for trajectory prediction (uses Huber loss).
Supports AMP + grad clipping + optional val split.
	•	src/train/train_eta.py:
Trains GRU + linear head for ETA.
	•	src/eval/:
	•	metrics_traj.py: ADE/FDE.
	•	metrics_eta.py: MAE/MAPE/P95.
	•	evaluate_traj.py: loads checkpoint, prints metrics, saves trajectory plots.
	•	evaluate_eta.py: loads ETA checkpoint, prints metrics.

⸻

9) Tips & Tuning
	•	Train longer: bump epochs (e.g., 30–50) and tune lr (3e-4 for TPTrans with AdamW is a good start).
	•	Batch size: increase until you hit memory limits.
	•	Normalization: make sure scaler.npz is created in processed_dir and training/eval both apply it.
	•	MMSI split: prefer vessel-wise splits to avoid leakage (use window_mmsi.npy as done in the evaluator).
	•	Baselines: compare against Constant Velocity; add EKF/KF for a classic reference.

⸻

10) Making ETA Realistic (Recommended)

The current ETA labels may be placeholders. For real ETA:
	1.	Load port polygons (e.g., WPI/harbor GeoJSON or shapefile).
	2.	Spatially join AIS points to determine when a vessel enters a destination port polygon.
	3.	For each window, label time_to_port_sec = time difference from window end to first port-entry timestamp.
	4.	Save time_to_port_sec into interim.parquet, then rebuild ETA processed data.

⸻

11) Troubleshooting
	•	FileNotFoundError: .../X.npy
Your config points to the wrong processed_dir. Update the YAML to a project-relative path like data/processed/traj_w64_h12.
	•	incompatible index ... during segmentation
Make sure segment_trajectories uses groupby(...).transform(...) (not apply), and that your DataFrame is sorted by ["mmsi","timestamp"].
	•	zsh: no matches found: (GRU)
That’s your shell globbing parentheses. Ignore it, or run setopt nonomatch.
	•	Matplotlib save error (FileNotFoundError)
Ensure data/figures/ exists: mkdir -p data/figures.
	•	CUDA not found
Install a CPU wheel or the correct CUDA PyTorch wheel for your system (see PyTorch’s install page).

⸻

12) One-shot run (sanity check)

# 1) Interim
```bash
bash scripts/make_interim.sh \
  --raw data/raw/*.csv \
  --out data/interim
```

# 2) Process (traj)
```bash
bash scripts/make_processed.sh \
  --interim data/interim/interim.parquet \
  --task trajectory --window 64 --horizon 12 \
  --out data/processed/traj_w64_h12
```

# 3) Train
```bash
python -m src.train.train_traj --config configs/traj_tptrans_base.yaml
```

# 4) Evaluate + plots
```bash
mkdir -p data/figures
python -m src.eval.evaluate_traj \
  --processed_dir data/processed/traj_w64_h12 \
  --ckpt data/checkpoints/traj_model.pt \
  --model tptrans \
  --plot
```

⸻

13) Ready for a Report
	•	Include ADE/FDE and (if available) per-horizon curves.
	•	Show sample trajectory plots (best/middle/worst).
	•	Compare GRU vs TPTrans vs CV baseline.
	•	For ETA: MAE/MAPE/P95 histograms; discuss feature importance if you add more signals.
	•	Discuss data leakage defenses (MMSI split), normalization, and realistic labeling choices.

⸻
