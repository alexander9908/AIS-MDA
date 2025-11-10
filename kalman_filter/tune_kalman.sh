#!/bin/sh

### Job Name:
#BSUB -J kalman_tune

### Queue Name:
#BSUB -q hpc

### Requesting 16 CPU cores, 8GB memory per core (grid search is parallelizable)
#BSUB -n 16
#BSUB -R "rusage[mem=8GB]"

### Setting a runtime limit of 8 hours (grid search takes longer)
#BSUB -W 8:00

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

### Run Kalman Filter hyperparameter tuning
### This will perform grid search over process/measurement noise parameters
### IMPORTANT: Update FINAL_DIR to point to your processed pickle files on HPC
FINAL_DIR="/work3/s204572/AIS-data/map_reduce_final"  # UPDATE THIS PATH for HPC

echo "Starting Kalman Filter hyperparameter tuning..."
echo "Data directory: $FINAL_DIR"
echo "Window size: 64, Horizon: 12"
echo "Start time: $(date)"

python -m kalman_filter.baselines.train_kalman \
    --final_dir $FINAL_DIR \
    --window_size 64 \
    --horizon 12 \
    --val_frac 0.15 \
    --test_frac 0.15 \
    --tune

echo "End time: $(date)"
echo "Best parameters saved to data/checkpoints/kalman_filter_best_params.json"
echo "Results saved to metrics/kalman_filter.json"
