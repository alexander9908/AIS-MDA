#!/bin/sh

### Job Name:
#BSUB -J kalman_test

### Queue Name:
#BSUB -q hpc

### Resource Request:
# 4 cores is sufficient for the parallel joblib execution
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 4:00

### Output files
#BSUB -o logs/kalman_%J.out
#BSUB -e logs/kalman_%J.err

# --- Setup Environment ---
cd ~/AIS-MDA
module swap python3/3.13.2
source .venv/bin/activate

# --- Define Paths ---
# Using the TEST directory specifically
TEST_DIR="/dtu/blackhole/10/178320/preprocessed_1/final/test"
OUT_DIR="data/checkpoints"

echo "=================================================="
echo "Starting Kalman Filter Baseline Evaluation"
echo "Test Data: $TEST_DIR"
echo "Output Dir: $OUT_DIR"
echo "Cores: 4"
echo "=================================================="

# --- Run Evaluation ---
# Note: --n_jobs 4 matches the BSUB -n 4 request
python -m kalman_filter.baselines.train_kalman \
    --final_dir "${TEST_DIR}" \
    --out_dir "${OUT_DIR}" \
    --window 64 \
    --horizon 12 \
    --n_jobs 8

echo "Job completed at $(date)"