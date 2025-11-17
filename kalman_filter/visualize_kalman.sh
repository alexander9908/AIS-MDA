#!/bin/sh

### Job Name:
#BSUB -J kalman_viz

### Queue Name:
#BSUB -q hpc

### Requesting 4 CPU cores, 4GB memory per core
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"

### Setting a runtime limit of 30 minutes
#BSUB -W 0:30

### Email notification when job begins and ends
#BSUB -B
#BSUB -N

### Output and error files
#BSUB -o logs/kalman_viz_%J.out
#BSUB -e logs/kalman_viz_%J.err


### cd to repo dir
cd ~/AIS-MDA

### load python module FIRST (before venv)
module swap python3/3.13.2

### activate environment (after module load)
source .venv/bin/activate

### Create log directory if it doesn't exist
mkdir -p logs

### --- Paths ---
# Using the test set for visualization
DATA_DIR="/dtu/blackhole/10/178320/preprocessed_1/final/test"
OUTPUT_DIR="kalman_filter/visualizations"

echo "Starting Kalman Filter visualization..."
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Start time: $(date)"

python -m kalman_filter.baselines.visualize_kalman \
    --final_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --window 64 \
    --horizon 12

echo "End time: $(date)"
echo "Visualizations saved to $OUTPUT_DIR"
