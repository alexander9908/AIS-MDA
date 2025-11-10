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
#BSUB -o hpc_jobs/logs/kalman_tune_%J.out
#BSUB -e hpc_jobs/logs/kalman_tune_%J.err


### cd to repo dir
cd ~/AIS-MDA

### activate environment
. .venv/bin/activate

### Run Kalman Filter hyperparameter tuning
### This will perform grid search over process/measurement noise parameters
echo "Starting Kalman Filter hyperparameter tuning..."
echo "Data directory: data/map_reduce_final"
echo "Window size: 64, Horizon: 12"
echo "Start time: $(date)"

python -m src.baselines.train_kalman \
    --final_dir data/map_reduce_final \
    --window_size 64 \
    --horizon 12 \
    --val_frac 0.15 \
    --test_frac 0.15 \
    --tune

echo "End time: $(date)"
echo "Best parameters saved to data/checkpoints/kalman_filter_best_params.json"
echo "Results saved to metrics/kalman_filter.json"
