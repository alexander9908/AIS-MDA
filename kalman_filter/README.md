# Kalman Filter Baseline for AIS Trajectory Prediction

This folder contains a complete Kalman Filter implementation serving as a classical baseline to compare against neural network models.

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
