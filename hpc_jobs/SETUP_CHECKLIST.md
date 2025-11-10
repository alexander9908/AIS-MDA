# HPC Setup Checklist for Kalman Filter Evaluation

## Quick Start Guide

You have **3 months of daily pickle files** and need to run the full pipeline.

---

## Step-by-Step Instructions

### ✅ 1. One-time HPC Setup
```bash
# SSH to HPC
ssh [username]@hpc.dtu.dk

# Navigate to repo
cd ~/AIS-MDA

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r env/requirements.txt

# Login to WandB
wandb login
```

---

### ✅ 2. Configure MapReduce Preprocessing Job

Edit `hpc_jobs/preprocess_mapreduce.sh` and update these 3 paths:

```bash
INPUT_DIR="/dtu/blackhole/10/178320/preprocessed_1/pickles"  # Your daily pickles
TEMP_DIR="/work3/[YOUR_USERNAME]/AIS-data/map_reduce_temp"   # ⚠️ UPDATE THIS
FINAL_DIR="/work3/[YOUR_USERNAME]/AIS-data/map_reduce_final" # ⚠️ UPDATE THIS
```

**Example:** If your username is `s204572`:
```bash
TEMP_DIR="/work3/s204572/AIS-data/map_reduce_temp"
FINAL_DIR="/work3/s204572/AIS-data/map_reduce_final"
```

---

### ✅ 3. Submit MapReduce Preprocessing (FIRST JOB)

```bash
cd ~/AIS-MDA
bsub < hpc_jobs/preprocess_mapreduce.sh
```

**What this does:**
- Processes 77 days of daily pickle files
- Groups by MMSI, cleans, interpolates, normalizes
- Creates ~700-1500 processed trajectory files
- Runtime: 8-16 hours
- Memory: 256GB (16 cores × 16GB each)

**Monitor progress:**
```bash
bstat                                          # Check job status
tail -f hpc_jobs/logs/mapreduce_<JOBID>.out   # Watch live output
```

---

### ✅ 4. Verify Preprocessing Output

After job completes:
```bash
# Check output files
ls /work3/[YOUR_USERNAME]/AIS-data/map_reduce_final/ | head

# Expected: 209056000_0_processed.pkl, 209056000_1_processed.pkl, etc.

# Count files
ls -1 /work3/[YOUR_USERNAME]/AIS-data/map_reduce_final/*.pkl | wc -l
# Should see: 700-1500 files
```

---

### ✅ 5. Configure Kalman Filter Evaluation Job

Edit `hpc_jobs/eval_kalman_full.sh` and update this path:

```bash
FINAL_DIR="/work3/[YOUR_USERNAME]/AIS-data/map_reduce_final"  # ⚠️ UPDATE THIS
```

---

### ✅ 6. Submit Kalman Filter Evaluation (SECOND JOB)

```bash
bsub < hpc_jobs/eval_kalman_full.sh
```

**What this does:**
- Loads all processed trajectories
- Creates sliding windows (window=64, horizon=12)
- Evaluates Kalman Filter baseline
- Runtime: 1-2 hours
- Memory: 64GB (8 cores × 8GB each)

**Monitor progress:**
```bash
bstat
tail -f hpc_jobs/logs/kalman_eval_<JOBID>.out
```

---

### ✅ 7. Check Results

After evaluation completes:
```bash
# View metrics
cat ~/AIS-MDA/metrics/kalman_filter.json

# View summary
cat ~/AIS-MDA/data/checkpoints/kalman_filter_summary.txt

# Copy results to your local machine
scp [username]@hpc.dtu.dk:~/AIS-MDA/metrics/kalman_filter.json .
```

**Expected metrics:**
- Test ADE: ~0.004 (≈445 meters)
- Test FDE: ~0.009 (≈1000 meters at 1-hour horizon)

---

## Timeline

| Task | Duration | Description |
|------|----------|-------------|
| **Setup** | 10 min | Install requirements, login to WandB |
| **Configure jobs** | 5 min | Update paths in shell scripts |
| **MapReduce preprocessing** | 8-16 hours | Process 3 months of data |
| **Kalman evaluation** | 1-2 hours | Train and test baseline |
| **Total** | ~10-18 hours | (Mostly unattended HPC time) |

---

## Troubleshooting

### "Directory does not exist" error
Create output directories manually:
```bash
mkdir -p /work3/[YOUR_USERNAME]/AIS-data/map_reduce_temp
mkdir -p /work3/[YOUR_USERNAME]/AIS-data/map_reduce_final
```

### "Out of memory" during MapReduce
Increase memory allocation in `preprocess_mapreduce.sh`:
```bash
#BSUB -R "rusage[mem=24GB]"  # Increase from 16GB to 24GB
```

### "Job exceeds runtime limit"
Increase time limit:
```bash
#BSUB -W 48:00  # Increase from 24:00 to 48 hours
```

### "No module named src.preprocessing"
Make sure you're in the repo directory and venv is activated:
```bash
cd ~/AIS-MDA
source .venv/bin/activate
```

---

## Quick Reference

**Submit preprocessing:**
```bash
bsub < hpc_jobs/preprocess_mapreduce.sh
```

**Submit evaluation:**
```bash
bsub < hpc_jobs/eval_kalman_full.sh
```

**Check status:**
```bash
bstat
```

**Kill job:**
```bash
bkill <JOBID>
```

**View logs:**
```bash
tail -f hpc_jobs/logs/mapreduce_<JOBID>.out
tail -f hpc_jobs/logs/kalman_eval_<JOBID>.out
```
