#!/bin/sh

### Job Name:
#BSUB -J kalman_eval_full

### Queue Name:
#BSUB -q hpc

### Requesting 8 CPU cores, 8GB memory per core (Kalman Filter is CPU-bound)
#BSUB -n 8
#BSUB -R "rusage[mem=8GB]"

### Setting a runtime limit of 4 hours (conservative for full dataset)
#BSUB -W 4:00

### Email notification when job begins and ends
#BSUB -B
#BSUB -N

### Output and error files
#BSUB -o hpc_jobs/logs/kalman_eval_%J.out
#BSUB -e hpc_jobs/logs/kalman_eval_%J.err


### cd to repo dir
cd ~/AIS-MDA

### activate environment
. .venv/bin/activate

### load python module
module swap python3/3.13.2

### Run Kalman Filter evaluation on ALL data
### This will process all trajectories with window=64, horizon=12
### IMPORTANT: Update FINAL_DIR to point to your processed pickle files on HPC
FINAL_DIR="/dtu/blackhole/10/178320/preprocessed_test"  # Your *_processed.pkl files

echo "Starting Kalman Filter evaluation on full dataset..."
echo "Data directory: $FINAL_DIR"
echo "Window size: 64, Horizon: 12"
echo "Start time: $(date)"

python -m kalman_filter.baselines.train_kalman \
    --final_dir $FINAL_DIR \
    --window_size 64 \
    --horizon 12 \
    --val_frac 0.15 \
    --test_frac 0.15

echo "End time: $(date)"
echo "Results saved to metrics/kalman_filter.json"
echo "Summary saved to data/checkpoints/kalman_filter_summary.txt"
