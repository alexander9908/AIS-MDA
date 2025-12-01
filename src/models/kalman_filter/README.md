# Kalman Filter Baseline for AIS Trajectory Prediction

This folder contains a complete Kalman Filter implementation serving as a classical baseline for trajectory forecasting. It is designed to be evaluated on a pre-split dataset and includes functionality for advanced, map-based visualizations.

---

## 🚀 Core Functionality

- **Model**: A constant velocity Kalman Filter with a 4D state `[latitude, longitude, velocity_lat, velocity_lon]`.
- **Evaluation**: Calculates Haversine-based Average Displacement Error (ADE) and Final Displacement Error (FDE) in meters. It is specifically configured to report FDE at 1-hour, 2-hour, and 3-hour horizons.
- **Data Handling**: Works with pre-split `train/`, `val/`, and `test/` directories containing processed `.pkl` files. This ensures a fair, MMSI-aware comparison against other models.
- **Visualization**: Generates high-quality plots of predicted vs. actual trajectories on a real map of Denmark. It uses `contextily` for live map tiles and falls back to a pre-generated land/sea mask created with `roaring-landmask`.

---

## 🛠️ Workflow & Scripts

The workflow is managed by shell scripts designed for a DTU HPC environment using the LSF scheduler (`bsub`).

### 1. Full Evaluation (`eval_kalman_full.sh`)

This is the main script for running the final evaluation on the test set.

- **What it does**:
    1.  Sets up the Python environment.
    2.  Runs `src/models/kalman_filter/baselines/train_kalman.py` on the full test dataset located at `/dtu/blackhole/10/178320/preprocessed_1/final/test`.
    3.  Uses a prediction horizon of 3 hours (`--horizon 36`).
    4.  The Python script evaluates the predictions and calculates Haversine ADE/FDE, saving the results to a JSON file in the `src/models/kalman_filter/` directory.

- **How to run**:
  ```bash
  bsub < src/models/kalman_filter/eval_kalman_full.sh
  ```

### 2. Visualization (`visualize_kalman.sh`)

This script generates the trajectory plots.

- **What it does**:
    1.  Sets up the Python environment.
    2.  **Builds Water Mask**: Runs `src/models/kalman_filter/build_water_mask.py` to create a `water_mask.png` file using the `roaring-landmask` library. This provides a fallback map background and requires no manual downloads.
    3.  **Generates Plots**: Runs `src/models/kalman_filter/baselines/visualize_kalman.py` to predict trajectories from the test set and plot them over a map background.
    4.  Saves the output `.png` files to `src/models/kalman_filter/visualizations/`.

- **How to run**:
  ```bash
  # First, ensure roaring-landmask is installed
  # pip install roaring-landmask

  # Then, submit the job
  bsub < src/models/kalman_filter/visualize_kalman.sh
  ```

---

## ⚙️ Noise Parameters

The filter uses fixed, untuned noise parameters defined in `src/models/kalman_filter/kalman_filter.py`:

- **`R` (Measurement Noise)**: `1e-5` for latitude and longitude, representing sensor inaccuracy.
- **`Q` (Process Noise)**: `1e-4` for position and velocity components, accounting for unmodeled dynamics like turns and speed changes.

While tuning these parameters could improve performance, the current implementation serves as a consistent and reproducible baseline.
      - Extract position: `[lat, lon]`
- **Output:** Predicted trajectory `(12, 2)` - 12 future positions

---

### 6. **Batch Prediction (All Test Windows)**
```python
# Method: kf.predict_batch()
# Location: src/models/kalman_filter/kalman_filter.py
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
# Location: src/models/kalman_filter/baselines/train_kalman.py
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
# Location: src/models/kalman_filter/baselines/train_kalman.py
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
bsub < src/models/kalman_filter/eval_kalman_full.sh
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
src/models/kalman_filter/
├── README.md                      # This file
├── kalman_filter.py               # Core Kalman Filter implementation
├── baselines/                     # Training, evaluation, and analysis scripts
│   ├── train_kalman.py            # Main training/evaluation script
│   ├── test_kalman.py             # Unit tests
│   ├── compare_models.py          # Compare with neural networks
│   ├── visualize_kalman.py        # Generate plots
│   └── README_KALMAN.md           # Technical documentation
├── eval_kalman.sh                 # Quick evaluation script
├── eval_kalman_full.sh            # HPC job: Full evaluation
├── preprocess_mapreduce.sh        # HPC job: Data preprocessing
└── tune_kalman.sh                 # HPC job: Hyperparameter tuning
```

---

## 🚀 Quick Start

### Local Evaluation (Small Dataset)
```bash
cd /path/to/AIS-MDA

# Run on local data
bash src/models/kalman_filter/eval_kalman.sh data/map_reduce_final 64 12
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
# Edit src/models/kalman_filter/eval_kalman_full.sh
nano src/models/kalman_filter/eval_kalman_full.sh

# Change this line:
FINAL_DIR="/dtu/blackhole/10/178320/preprocessed_test"  # Your path
```

**Step 3: Submit job**
```bash
bsub < src/models/kalman_filter/eval_kalman_full.sh
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
bash src/models/kalman_filter/eval_kalman.sh <data_dir> <window> <horizon>
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
from src.models.kalman_filter.kalman_filter import TrajectoryKalmanFilter, KalmanFilterParams

# Or use the baselines module
from src.models.kalman_filter.baselines.train_kalman import load_trajectories, evaluate_kalman
```

### Use in Notebooks
```python
import sys
sys.path.append('..')  # If in notebooks/
from src.models.kalman_filter.kalman_filter import TrajectoryKalmanFilter

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
bash src/models/kalman_filter/eval_kalman.sh data/map_reduce_final 64 12

# HPC submission
bsub < src/models/kalman_filter/eval_kalman_full.sh

# Check HPC job status
bstat

# View results
cat metrics/kalman_filter.json
cat data/checkpoints/kalman_filter_summary.txt
```

---

**For detailed technical documentation, see `baselines/README_KALMAN.md`**
