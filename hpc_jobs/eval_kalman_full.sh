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

### Run Kalman Filter evaluation on ALL data in map_reduce_final/
### This will process ~700 trajectories with window=64, horizon=12
echo "Starting Kalman Filter evaluation on full dataset..."
echo "Data directory: data/map_reduce_final"
echo "Window size: 64, Horizon: 12"
echo "Start time: $(date)"

python -m src.baselines.train_kalman \
    --final_dir data/map_reduce_final \
    --window_size 64 \
    --horizon 12 \
    --val_frac 0.15 \
    --test_frac 0.15

echo "End time: $(date)"
echo "Results saved to metrics/kalman_filter.json"
echo "Summary saved to data/checkpoints/kalman_filter_summary.txt"
