# Kalman Filter Baseline for AIS Trajectory Prediction

This folder contains a complete Kalman Filter implementation serving as a classical baseline to compare against neural network models.

---

## 🔄 How It Works: Step-by-Step Execution Flow

### 1. **Load Preprocessed Data**
```python
# Function: load_trajectories()
# Location: kalman_filter/baselines/train_kalman.py
```
- **Input:** Directory with `*_processed.pkl` files (e.g., `/dtu/blackhole/10/178320/preprocessed_test/`)
- **Process:** 
  - Scans directory for all `.pkl` files ending in `_processed.pkl`
  - Loads each file: `{'mmsi': int, 'traj': np.ndarray}`
  - Validates trajectory shape: `(T, 9)` where T ≥ 20 timesteps
  - Filters out short/invalid trajectories
- **Output:** List of trajectory arrays, each shape `(T, 9)`
- **Example:** 993 valid trajectories loaded from 1000 files

---

### 2. **Split Data by Vessel (MMSI-Aware)**
```python
# Function: split_trajectories()
# Location: kalman_filter/baselines/train_kalman.py
```
- **Input:** List of trajectories, `val_frac=0.15`, `test_frac=0.15`
- **Process:**
  - Shuffles trajectories randomly (seed=42 for reproducibility)
  - Splits into three sets:
    - Test: 15% of trajectories (148 trajectories)
    - Validation: 15% of trajectories (148 trajectories)
    - Train: 70% of trajectories (697 trajectories)
  - **Critical:** Each vessel (MMSI) stays in only ONE split (no data leakage)
- **Output:** `train_trajs`, `val_trajs`, `test_trajs`

---

### 3. **Create Sliding Windows**
```python
# Function: create_windows()
# Location: kalman_filter/baselines/train_kalman.py
```
- **Input:** Trajectories, `window=64`, `horizon=12`, `max_windows=999999`
- **Process:**
  - For each trajectory with T timesteps:
    - Creates overlapping windows starting at each position
    - Window format: `[start : start+64]` → predict `[start+64 : start+76]`
    - Continues until not enough data for full window+horizon
  - Extracts only lat/lon columns for targets (columns 0,1)
  - Stops when reaching `max_windows` total across all trajectories
- **Output:** 
  - `X`: Input windows `(N, 64, 9)` - N windows, 64 timesteps, 9 features
  - `Y`: Target positions `(N, 12, 2)` - N windows, 12 future steps, lat/lon
- **Example:** From 148 test trajectories → 73,482 windows

---

### 4. **Initialize Kalman Filter**
```python
# Class: TrajectoryKalmanFilter
# Location: kalman_filter/kalman_filter.py
```
- **Input:** `KalmanFilterParams` with noise values
- **Process:**
  - Sets up state-space model:
    - State vector: `[lat, lon, v_lat, v_lon]` (position + velocity)
    - Measurement: `[lat, lon]` (only positions observed)
    - Dynamics: Constant velocity model `x_{k+1} = F·x_k + noise`
  - Initializes covariance matrices:
    - `Q`: Process noise (uncertainty in motion model)
    - `R`: Measurement noise (GPS/AIS sensor uncertainty)
    - `P`: State covariance (uncertainty in estimates)
- **Default Parameters:**
  - `process_noise_pos = 1e-5`
  - `process_noise_vel = 1e-4`
  - `measurement_noise = 1e-4`
  - `dt = 300.0` seconds (5-minute intervals)

---

### 5. **Predict (For Each Window)**
```python
# Method: kf.predict()
# Location: kalman_filter/kalman_filter.py
```
- **Input:** One window `(64, 9)` of historical positions
- **Process:**
  - **Filter Phase:** Process 64 historical observations
    - For each timestep t in [0, 63]:
      - **Prediction step:** `x̂ = F·x`, `P = F·P·F^T + Q`
      - **Update step:** Incorporate measurement, update state estimate
      - Uses Kalman gain to balance prediction vs measurement
  - **Forecast Phase:** Project 12 steps into future
    - Starting from final filtered state at t=63
    - For each step h in [0, 11]:
      - **Pure prediction:** `x̂ = F·x̂` (no new measurements)
      - Extract position: `[lat, lon]`
- **Output:** Predicted trajectory `(12, 2)` - 12 future positions

---

### 6. **Batch Prediction (All Test Windows)**
```python
# Method: kf.predict_batch()
# Location: kalman_filter/kalman_filter.py
```
- **Input:** All test windows `X_test (73482, 64, 9)`
- **Process:**
  - Processes windows in batches of 100 (for memory efficiency)
  - For each batch:
    - Calls `kf.predict()` on each window
    - Collects predictions
  - Shows progress bar (tqdm)
- **Output:** All predictions `(73482, 12, 2)`

---

### 7. **Compute Metrics**
```python
# Function: evaluate_kalman()
# Location: kalman_filter/baselines/train_kalman.py
```
- **Input:** Predictions `(N, 12, 2)`, Ground truth `Y (N, 12, 2)`
- **Process:**
  - **ADE (Average Displacement Error):**
    - For each prediction: mean Euclidean distance across all 12 steps
    - Average across all N windows
    - Formula: `mean(||pred - truth||)` over all timesteps and windows
  - **FDE (Final Displacement Error):**
    - Euclidean distance at final step (t=12)
    - Average across all N windows
    - Formula: `mean(||pred[:, 11] - truth[:, 11]||)`
  - **Per-horizon ADE:**
    - Compute ADE separately for each timestep [1, 2, ..., 12]
    - Shows how error grows with prediction horizon
- **Output:** 
  ```python
  {
    'ade': 0.003265,           # Mean error across all steps
    'fde': 0.007339,           # Error at final step
    'per_horizon_ade': [...],  # Error at each step
    'n_samples': 73482         # Number of windows evaluated
  }
  ```

---

### 8. **Save Results**
```python
# Function: main() - end section
# Location: kalman_filter/baselines/train_kalman.py
```
- **Process:**
  - **JSON file** (`metrics/kalman_filter.json`):
    - Saves all metrics, parameters, dataset info
    - Compatible with neural network comparison scripts
  - **Text summary** (`data/checkpoints/kalman_filter_summary.txt`):
    - Human-readable format
    - Quick reference for results
- **Output Files:**
  ```
  metrics/kalman_filter.json
  data/checkpoints/kalman_filter_summary.txt
  ```

---

## 📊 Complete Execution Example

```bash
# Command on HPC
bsub < kalman_filter/eval_kalman_full.sh
```

**What happens:**
1. ✅ Loads 993 trajectories from `/dtu/blackhole/10/178320/preprocessed_test/`
2. ✅ Splits: 697 train, 148 validation, 148 test
3. ✅ Creates ~73,000 sliding windows from test set
4. ✅ Initializes Kalman Filter with default parameters
5. ✅ Processes each window:
   - Filters 64 historical observations
   - Predicts 12 future positions
6. ✅ Computes ADE/FDE metrics
7. ✅ Saves results to JSON + text files
8. ✅ **Runtime:** ~5-10 minutes on 4 CPU cores

**Results:**
- Test ADE: 0.003265 (≈363 meters real-world)
- Test FDE: 0.007339 (≈815 meters at 1-hour horizon)
- Per-step error growth: 22m @ 5min → 815m @ 60min

---

## 📁 Folder Structure

```
kalman_filter/
├── README.md                      # This file
├── kalman_filter.py              # Core Kalman Filter implementation
├── baselines/                     # Training, evaluation, and analysis scripts
│   ├── train_kalman.py           # Main training/evaluation script
│   ├── test_kalman.py            # Unit tests
│   ├── compare_models.py         # Compare with neural networks
│   ├── visualize_kalman.py       # Generate plots
│   └── README_KALMAN.md          # Technical documentation
├── eval_kalman.sh                # Quick evaluation script
├── eval_kalman_full.sh           # HPC job: Full evaluation
├── preprocess_mapreduce.sh       # HPC job: Data preprocessing
└── tune_kalman.sh                # HPC job: Hyperparameter tuning
```

---

## 🚀 Quick Start

### Local Evaluation (Small Dataset)
```bash
cd /path/to/AIS-MDA

# Run on local data
bash kalman_filter/eval_kalman.sh data/map_reduce_final 64 12
```

### HPC Evaluation (Full Dataset)

**Step 1: SSH to HPC and setup**
```bash
ssh [username]@hpc.dtu.dk
cd ~/AIS-MDA
python3 -m venv .venv && source .venv/bin/activate
pip install -r env/requirements.txt
```

**Step 2: Update data path in job script**
```bash
# Edit kalman_filter/eval_kalman_full.sh
nano kalman_filter/eval_kalman_full.sh

# Change this line:
FINAL_DIR="/dtu/blackhole/10/178320/preprocessed_test"  # Your path
```

**Step 3: Submit job**
```bash
bsub < kalman_filter/eval_kalman_full.sh
```

**Step 4: Monitor progress**
```bash
bstat
tail -f hpc_jobs/logs/kalman_eval_*.out
```

---

## 📊 What You Get

After running evaluation, you'll have:

1. **Metrics JSON** (`metrics/kalman_filter.json`)
   - ADE/FDE scores
   - Per-horizon error breakdown
   - Configuration used

2. **Summary Text** (`data/checkpoints/kalman_filter_summary.txt`)
   - Human-readable results
   - Quick reference

3. **Visualizations** (optional, via `visualize_kalman.py`)
   - Trajectory prediction plots
   - Error distributions

---

## 📈 Expected Performance

For Denmark AIS data (5-minute sampling):

| Metric | Value | Real-World |
|--------|-------|------------|
| ADE | 0.004-0.010 | 445-1110m |
| FDE | 0.009-0.020 | 1000-2220m |

**Strengths:**
- Excellent for straight-line cruising
- Fast inference (100-1000× faster than NNs)
- Interpretable parameters

**Limitations:**
- Poor during turns/maneuvers
- Linear model only
- No learning from data

---

## 🔧 Available Scripts

### Local Scripts

**`eval_kalman.sh`** - Quick evaluation
```bash
bash kalman_filter/eval_kalman.sh <data_dir> <window> <horizon>
```

### HPC Job Scripts

**`eval_kalman_full.sh`** - Full evaluation (1-2 hours)
- 8 CPU cores, 8GB RAM each
- Evaluates all trajectories
- Saves to `metrics/kalman_filter.json`

**`preprocess_mapreduce.sh`** - Data preprocessing (8-16 hours)
- 16 CPU cores, 16GB RAM each
- Converts daily pickles → processed pickles
- Only needed if you have raw daily data

**`tune_kalman.sh`** - Hyperparameter tuning (8 hours)
- 16 CPU cores, 8GB RAM each
- Grid search over noise parameters
- Saves best config to `data/checkpoints/kalman_filter_best_params.json`

---

## 📚 Documentation

- **`baselines/README_KALMAN.md`** - Complete technical documentation
  - Mathematical foundation
  - Implementation details
  - Parameter tuning guide
  - Comparison with neural networks

---

## 🔗 Integration with Main Repo

### Import in Python Scripts
```python
# From anywhere in the repo
from kalman_filter.kalman_filter import TrajectoryKalmanFilter, KalmanFilterParams

# Or use the baselines module
from kalman_filter.baselines.train_kalman import load_trajectories, evaluate_kalman
```

### Use in Notebooks
```python
import sys
sys.path.append('..')  # If in notebooks/
from kalman_filter.kalman_filter import TrajectoryKalmanFilter

kf = TrajectoryKalmanFilter()
predictions = kf.predict(window, horizon=12)
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'kalman_filter'"
Add repo root to Python path:
```python
import sys
sys.path.insert(0, '/path/to/AIS-MDA')
```

### "No pickle files found"
Check that data path points to directory with `*_processed.pkl` files.

### Job scripts don't run on HPC
Make sure paths are updated in the job scripts to match your HPC directory structure.

---

## 📞 Quick Reference

```bash
# Local evaluation
bash kalman_filter/eval_kalman.sh data/map_reduce_final 64 12

# HPC submission
bsub < kalman_filter/eval_kalman_full.sh

# Check HPC job status
bstat

# View results
cat metrics/kalman_filter.json
cat data/checkpoints/kalman_filter_summary.txt
```

---

**For detailed technical documentation, see `baselines/README_KALMAN.md`**
