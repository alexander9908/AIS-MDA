#!/bin/bash
#BSUB -J visualize_kalman
#BSUB -o logs/kalman_visualization_%J.out
#BSUB -e logs/kalman_visualization_%J.err
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=4G]"
#BSUB -W 00:30

### cd to repo dir
cd ~/AIS-MDA

### load python module FIRST (before venv)
module swap python3/3.13.2

### activate environment (after module load)
source .venv/bin/activate

### Create log directory if it doesn't exist
mkdir -p logs

# --- Install/Verify roaring-landmask ---
# The PBF file will be downloaded to the cache on first run.
echo "Verifying/installing roaring-landmask..."
pip install roaring-landmask

# --- Run Visualization ---
# This script generates the high-quality, zoomed-in plots for the final 3-hour evaluation.
echo "Running visualization script for final 3-hour horizon plots..."
python -m kalman_filter.baselines.visualize_kalman \
    --final_dir /dtu/blackhole/10/178320/preprocessed_1/final/test \
    --output_dir data/figures/kalman_final_3h \
    --horizon 36 \
    --n_examples 6

echo "Visualization script finished."
echo "Plots saved in data/figures/kalman_final_3h"
