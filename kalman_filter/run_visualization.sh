#!/bin/bash
#BSUB -J visualize_kalman
#BSUB -o logs/kalman_visualization_%J.out
#BSUB -e logs/kalman_visualization_%J.err
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=4G]"
#BSUB -W 00:30

# --- Setup environment ---
echo "Setting up environment..."
module load python3/3.13.2
source .venv/bin/activate
echo "Environment setup complete."
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"

# --- Run Visualization ---
# This script generates the high-quality, zoomed-in plots.
echo "Running visualization script..."
python -m kalman_filter.baselines.visualize_kalman \
    --final_dir /dtu/blackhole/10/178320/preprocessed_1/final \
    --output_dir data/figures/kalman_final \
    --horizon 36

echo "Visualization script finished."
