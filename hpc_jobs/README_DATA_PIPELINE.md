# HPC Data Pipeline for Kalman Filter Evaluation

## Data Flow Overview

Your AIS data goes through this pipeline on the HPC:

```
Raw CSVs (per day)
    ↓
[csv2pkl.py] → Daily pickle files (intermediate)
    ↓
[map_reduce.py] → Final processed pickles
    ↓
[train_kalman.py] → Metrics & evaluation
```

## File Formats

### 1. Daily Pickle Files (Intermediate)
- **Location on HPC**: Your "3 months folder with data per day"
- **Format**: `{date}.pkl` containing `{mmsi: np.ndarray([LAT, LON, SOG, ...]), ...}`
- **Purpose**: Intermediate format after CSV parsing
- **Created by**: `src/preprocessing/csv2pkl.py`

### 2. Final Processed Pickles (What Kalman Filter needs)
- **Location on HPC**: Should be in something like `/work3/[username]/AIS-data/map_reduce_final/`
- **Format**: `{mmsi}_{voyage_id}_processed.pkl`
- **Content**: 
  ```python
  {
      'mmsi': 209056000,
      'traj': np.ndarray  # shape (T, 9)
  }
  ```
- **Columns**: `[LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI]`
- **Normalized**: Lat/Lon normalized to [0, 1] for Denmark region (54-59°N, 5-17°E)
- **Sampling**: 5-minute intervals (300s)
- **Created by**: `src/preprocessing/map_reduce.py`

## Before Running Kalman Filter on HPC

### YOUR SITUATION: You have daily pickles like this
```
/dtu/blackhole/10/178320/preprocessed_1/pickles:
aisdk-2025-02-27.pkl  aisdk-2025-08-11.pkl  ... (3 months of daily files)
```

You need to run MapReduce preprocessing FIRST to create final processed files.

### Step 1: Update paths in preprocessing job script
Edit `hpc_jobs/preprocess_mapreduce.sh` and update these paths:
```bash
INPUT_DIR="/dtu/blackhole/10/178320/preprocessed_1/pickles"  # Your daily pickles
TEMP_DIR="/work3/[your_username]/AIS-data/map_reduce_temp"  # UPDATE username
FINAL_DIR="/work3/[your_username]/AIS-data/map_reduce_final" # UPDATE username
```

### Step 2: Submit MapReduce preprocessing job
```bash
cd ~/AIS-MDA
bsub < hpc_jobs/preprocess_mapreduce.sh
```

This will:
- Process all your daily pickle files (~77 days based on your listing)
- Group all data by MMSI (Stage 1: Map & Shuffle)
- Clean, interpolate, and normalize each vessel's trajectory (Stage 2: Reduce)
- Split into individual voyages
- Save final processed pickles: `{mmsi}_{voyage_id}_processed.pkl`

**Resources allocated:**
- 16 CPU cores
- 16GB RAM per core (256GB total)
- 24 hour time limit
- Logs: `hpc_jobs/logs/mapreduce_<JOBID>.out`

**Expected runtime:** 8-16 hours for 3 months of Denmark AIS data

**Expected output:** 700-1500 processed trajectory files in `FINAL_DIR`

### Step 3: Wait for preprocessing to complete
Monitor with:
```bash
bstat  # Check job status
tail -f hpc_jobs/logs/mapreduce_<JOBID>.out  # Watch progress
```

### Step 4: After preprocessing completes, check output
```bash
ls /work3/[username]/AIS-data/map_reduce_final/ | head
# Should see: 209056000_0_processed.pkl, 209056000_1_processed.pkl, etc.

# Count total files
ls -1 /work3/[username]/AIS-data/map_reduce_final/*.pkl | wc -l
```

### Step 5: Run Kalman Filter evaluation
Once preprocessing is complete, update `hpc_jobs/eval_kalman_full.sh`:
```bash
FINAL_DIR="/work3/[your_username]/AIS-data/map_reduce_final"
```

Then submit:
```bash
bsub < hpc_jobs/eval_kalman_full.sh
```

This will:
- Load all processed trajectories
- Split into train/val/test (70%/15%/15%)
- Create sliding windows (window=64, horizon=12)
- Evaluate Kalman Filter baseline
- Save metrics to `metrics/kalman_filter.json`

**Expected runtime:** 1-2 hours on 8 CPU cores

## Expected Results

With 3 months of Denmark AIS data (~700-1000 processed trajectories), you should get:
- **Test ADE**: ~0.004 (≈445 meters in real-world distance)
- **Test FDE**: ~0.009 (≈1000 meters at 1-hour horizon)
- **Runtime**: 1-2 hours on 8 CPU cores

Results will be saved to:
- `metrics/kalman_filter.json` (for comparison with TPTrans/GRU)
- `data/checkpoints/kalman_filter_summary.txt` (human-readable)

## Troubleshooting

### "No pickle files found"
- Check that `FINAL_DIR` path is correct
- Verify files end with `_processed.pkl`
- Make sure map_reduce.py has been run

### "Not enough trajectories"
- map_reduce.py filters out short/low-quality tracks
- Check the output of map_reduce.py to see how many valid voyages were created

### "Out of memory"
- Reduce `--window_size` or process fewer files
- The job script allocates 8GB per core, should be sufficient

## Quick Reference

**If you have daily pickles but NOT final processed pickles:**
1. Run `map_reduce.py` first (submit as HPC job, takes hours)
2. Then run Kalman Filter evaluation

**If you already have final processed pickles:**
1. Update `FINAL_DIR` in `hpc_jobs/eval_kalman_full.sh`
2. Submit with `bsub < hpc_jobs/eval_kalman_full.sh`
