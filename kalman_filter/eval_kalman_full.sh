#!/bin/sh

### Job Name:
#BSUB -J kalman_eval_test

### Queue Name:
#BSUB -q hpc

### Requesting 8 CPU cores, 8GB memory per core (Kalman Filter is CPU-bound)
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"

### Setting a runtime limit of 4 hours (conservative for full dataset)
#BSUB -W 2:00

### Email notification when job begins and ends
#BSUB -B
#BSUB -N

### Output and error files
#BSUB -o Output_%J.out
#BSUB -e Output_%J.err


### cd to repo dir
cd ~/AIS-MDA

### load python module FIRST (before venv)
module swap python3/3.13.2

### activate environment (after module load)
source .venv/bin/activate

### Run Kalman Filter evaluation on pre-split data
DATA_ROOT="/dtu/blackhole/10/178320/preprocessed_1/final"
TRAIN_DIR="${DATA_ROOT}/train"
VAL_DIR="${DATA_ROOT}/val"
TEST_DIR="${DATA_ROOT}/test"

echo "Starting Kalman Filter evaluation on pre-split dataset..."
echo "Train directory: $TRAIN_DIR"
echo "Val directory: $VAL_DIR"
echo "Test directory: $TEST_DIR"
echo "Window size: 64, Horizon: 12"
echo "Start time: $(date)"

python -m kalman_filter.baselines.train_kalman \
    --train_dir "${TRAIN_DIR}" \
    --val_dir "${VAL_DIR}" \
    --test_dir "${TEST_DIR}" \
    --window 64 \
    --horizon 36 \
    --max_windows 999999

echo "End time: $(date)"
echo "Results saved to metrics/kalman_filter.json"
echo "Summary saved to data/checkpoints/kalman_filter_summary.txt"
