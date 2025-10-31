# AIS-MDA Agent Brief

## Mission Synopsis
- **Objective:** Increase maritime domain awareness by modeling spatio-temporal AIS trajectories for trajectory forecasting, anomaly detection, and port call / ETA prediction.
- **Data:** Real-world AIS messages (irregular, noisy). Core fields: `mmsi`, `timestamp`, `lat`, `lon`, `sog`, `cog`, `heading`, `nav_status`, vessel/voyage attributes.
- **Deliverables:** Cleaned datasets, processed sliding windows, trained checkpoints (Constant Velocity, GRUSeq2Seq, TPTrans, ETA GRU), evaluation reports, figures, and usage documentation.
- **Key references:** AI in Ship Trajectory Prediction (2024), TPTrans (2023), JMSE Vessel Trajectory Benchmark (2025), ETA Review (2025) — see root `README.md` for links.

## Project Goals & Tasks
- **Trajectory prediction:** Sequence-to-sequence forecasting of Δlat/Δlon (or absolute positions) across short- and mid-horizon windows.
- **Anomaly detection:** Self-supervised reconstruction/forecasting to spot atypical vessel motion (synthetic injection currently available; real anomaly labeling optional).
- **Port call & ETA prediction:** Classify next port / regress arrival times using engineered kinematic + contextual features.
- **Data robustness:** Handle irregular sampling, sensor noise, missing values, and large-scale AIS ingestion without vessel leakage across splits.

## Data & Feature Summary
- **Raw schema:** `mmsi`, `timestamp`, `lat`, `lon`, `sog`, `cog`, `heading`, `nav_status`, optional `shiptype`, `draught`, `destination`, `rot`, dimensions, etc.
- **Canonical ingestion:** `src/dataio/load_ais.py` normalizes headers (CSV/Parquet), coerces numerics, parses timestamps, clips invalid coordinates, sorts by `mmsi`/`timestamp`.
- **Segmentation metadata:** `segment_id` assigned when time gaps exceed configurable threshold (default 6 h) per MMSI.
- **Derived kinematics:** Δx/Δy (projected meters), Δt, velocities (`dx_dt`, `dy_dt`), speed change (`accel`), trigonometric course (`cog_sin`, `cog_cos`), optional grid cells (`cell_id`).
- **ETA-specific context:** Distance/bearing to destination ports when available (`dist_to_port`, `bearing_to_port`); fallback pseudo labels computed via cumulative Δt.

## End-to-End Workflow Overview
1. **Raw ingestion (`src/dataio/load_ais.py`):**
   - Handles CSV/Parquet, normalizes headers, coerces datatypes, sorts by `mmsi` and `timestamp`.
   - Clips geographic outliers; drops invalid rows.
2. **Segmentation (`src/dataio/segment.py`):**
   - Splits trajectories at long temporal gaps (default 6 h) per MMSI.
   - Filters short segments; produces `segment_id`.
3. **Feature engineering (`src/features`):**
   - `kinematics.py` computes `dt`, projected meters (`x`, `y`), deltas (`dx`, `dy`), turn encodings (`cog_sin`, `cog_cos`), and acceleration.
   - `context.py` adds grid cells or other contextual encodings.
4. **Label generation (`src/labeling`):**
   - `traj_labels.make_traj_windows` → sliding windows for trajectory deltas (Δlat, Δlon) with configurable window/horizon.
   - `eta_labels.make_eta_windows` → windows plus scalar ETA (real or pseudo).
   - `anomalies.inject_synthetic_anomalies` → optional self-supervised anomaly signals.
5. **Processing scripts (`scripts/`):**
   - `make_interim.sh` → orchestrates load ➝ clean ➝ segment ➝ feature step.
   - `make_processed.sh` → builds task-specific NumPy tensors (`X.npy`, `Y.npy`, `y_eta.npy`, `scaler.npz`, `window_mmsi.npy`).
   - `run_all.sh` → chained run (preprocess ➝ train ➝ evaluate).
6. **Training (`src/train`):**
   - `train_traj.py` trains GRUSeq2Seq or TPTrans with Huber loss, AMP, gradient clipping.
   - `train_eta.py` fine-tunes GRU encoder-decoder with linear head for ETA regression.
   - `nested_cv_traj.py` runs MMSI-grouped nested CV with SGD random search.
7. **Evaluation (`src/eval`):**
   - Metrics: ADE/FDE (`metrics_traj.py`), MAE/MAPE/P95 (`metrics_eta.py`), AUROC/AUPRC/TTD for anomalies.
   - `evaluate_traj.py` supports MMSI-wise validation splits, plotting, JSON metric dumps.

## MapReduce Preprocessing (Large AIS Pipeline)
- **Step 0** — Stage raw daily CSVs in a directory (e.g., NOAA, MarineCadastre exports).
- **Step 1** (`src/preprocessing/csv2pkl.py`):
  - Stream each CSV, normalize coordinates/speeds, enforce ROI bounds, deduplicate, and pickle per-day dictionaries `{mmsi: np.ndarray([lat, lon, sog, cog, heading, rot, nav, timestamp, mmsi])}`.
- **Step 2** (`src/preprocessing/map_reduce.py`):
  - **Map/Shuffling:** Regroup pickled segments by MMSI into `TEMP_DIR/mmsi/*.pkl`.
  - **Reduce:** Concatenate each vessel’s segments, run `process_single_mmsi_track` to clean, split voyages, interpolate to 5 min cadence, normalize, and drop low-quality tracks.
  - **Output:** Final per-sample pickles `FINAL_DIR/{mmsi}_{voyage_id}_processed.pkl` with schema `{"mmsi": id, "traj": ndarray}` (aligned with TrAISformer/GeoTrackNet formats).
- **Utilities:** `src/preprocessing/utils.py` supplies Koyak-based outlier detection, interpolation, and geodesic helpers.
- **Notes:** Process MMSIs sequentially to limit memory; MapReduce scaffolding enables distributed preprocessing of very large AIS collections.

## Model Portfolio
- **Baselines (`src/models/kinematic.py`):** Constant velocity extrapolation for ADE/FDE reference.
- **GRUSeq2Seq (`src/models/rnn_seq2seq.py`):**
  - Encoder-decoder GRUs on engineered features; horizon-sized zero-input decoder.
  - Used for trajectory and ETA tasks (ETA adds linear head in `train_eta.py`).
- **TPTrans (`src/models/tptrans.py`):**
  - 1D CNN encoder for local patterns ➝ Transformer encoder for global context ➝ GRU decoder projecting to Δlat/Δlon.
  - Configurable `d_model`, `nhead`, encoder/decoder layers, dropout.
- **Training regime:**
  - Default optimizer AdamW; optional SGD in nested CV.
  - Mixed precision, gradient clipping, optional scalers (`scaler.npz`, `target_scaler.npz`).
- **Evaluation metrics:**
  - Trajectory: ADE, FDE, Hausdorff (optional).
  - ETA: MAE, MAPE, 95th percentile error.
  - Anomaly: AUROC, AUPRC, time-to-detection.

## Technology Stack
- **Python 3.11+**; dependencies listed in `env/requirements.txt`.
- **Data libraries:** pandas, polars, pyarrow, geopandas, shapely, h3.
- **Parallelism:** pandarallel (optional) for data prep.
- **Deep learning:** PyTorch (torchvision/torchaudio for completeness).
- **Notebooks:** `notebooks/` for exploration (`00_explore_ais.ipynb`, etc.).
- **Containerization:** `env/Dockerfile` for reproducible builds.

## Configuration & Experiment Management
- **YAML configs (`configs/`):** Define `processed_dir`, model hyperparameters, optimizer settings, train/val splits (`val_frac`), logging paths.
- **Example (`configs/traj_tptrans_base.yaml`):** Sets window=64, horizon=12, `model: {name: tptrans, d_model: 192, nhead: 4, enc_layers: 4, dec_layers: 2}`, `epochs`, `lr`, `batch_size`.
- **CLI overrides:** Most scripts accept flags to override config entries (e.g., `--config`, `--processed_dir`, `--outer_folds`).
- **Reproducibility:** Nested CV enforces MMSI-grouped splits, random seeds fixed in data loaders, metrics written to `metrics/`.

## Operational Workflow (CLI)
```bash
# 0) Environment
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r env/requirements.txt

# 1) Intermediate parquet (clean + segment + features)
bash scripts/make_interim.sh \
  --raw data/raw/*.csv \
  --out data/interim

# 2) Task-specific tensors
bash scripts/make_processed.sh \
  --interim data/interim/interim.parquet \
  --task trajectory --window 64 --horizon 12 \
  --out data/processed/traj_w64_h12

# 3) Train models
python -m src.train.train_traj --config configs/traj_tptrans_base.yaml
python -m src.train.train_eta --config configs/eta_gru.yaml

# 4) Evaluate
python -m src.eval.evaluate_traj \
  --processed_dir data/processed/traj_w64_h12 \
  --ckpt data/checkpoints/traj_tptrans.pt \
  --model tptrans \
  --plot
```

## Key Directories & Artifacts
- `data/raw/`: source AIS files (CSV/Parquet).
- `data/interim/`: cleaned, segmented parquet.
- `data/processed/<task>/`: NumPy tensors (`X.npy`, `Y.npy`, `y_eta.npy`), scalers, `window_mmsi.npy`.
- `configs/`: YAML experiment definitions (window, horizon, model hyperparameters).
- `scripts/`: automation entry points (preprocess, run_all, summarize_results).
- `src/`: modular Python package for data IO, features, labeling, models, training, evaluation, utilities.
- `metrics/`: JSON evaluation reports generated by evaluation scripts.

## Validation & Reporting
- **Trajectory:** Report ADE/FDE mean ± std, per-horizon curves when available, qualitative plots (best/median/worst trajectories).
- **ETA:** Publish MAE/MAPE/P95, histograms, optional calibration analysis; explicitly state label source (real port polygons vs pseudo).
- **Anomaly:** Provide AUROC/AUPRC, time-to-detection, and qualitative case studies.
- **Logging:** Persist metrics in `metrics/` JSON, store plots in `data/figures/`, document experiment context in README or report.

## Guidance for AI Contributors
- **Coding principles:**
  - Favor modular, well-documented functions; keep preprocessing memory-aware.
  - Maintain compatibility with large-scale AIS data (streaming, MapReduce-ready).
  - Respect MMSI-group splits to avoid vessel leakage.
  - Align new features with existing configs/CLI scripts.
- **Testing & validation:**
  - When feasible, add unit-style smoke tests or notebook snippets for new components.
  - Use ADE/FDE and ETA metrics for regression tasks; report baselines for comparison.
- **Documentation:**
  - Update or create READMEs for new modules or scripts.
  - Record data assumptions, config changes, and CLI usage.
  - Store metrics via existing JSON/figure conventions.

## AI Contribution Checklist
- Confirm preprocessing compatibility with MapReduce pipeline and `make_*` scripts.
- Keep data splits MMSI-aware; avoid leakage by using provided `window_mmsi.npy`.
- Extend configs and scripts when adding models/features; update documentation accordingly.
- Validate core metrics locally before shipping; highlight limitations or TODOs in README.
- Ensure code runs within existing environment (no silent dependency drift).

## System Prompt for Downstream Agents
```
You are an AIS-MDA coding assistant. Always:
1. Write clean, maintainable, and well-factored code with meaningful names.
2. Add focused code comments when logic is non-obvious or domain-specific.
3. Provide or update a README (or README section) explaining how to use any new script, module, or workflow you implement.
4. Favor incremental, well-tested changes; validate with available scripts or metrics.
5. Respect existing data-processing contracts (MapReduce pipeline, MMSI-based splits) and keep scalability in mind.
```
