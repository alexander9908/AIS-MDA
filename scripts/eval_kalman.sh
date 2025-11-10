#!/usr/bin/env bash
# Evaluate Kalman Filter baseline on AIS trajectory prediction
# Usage: bash scripts/eval_kalman.sh [--tune]

set -euo pipefail

FINAL_DIR="${1:-data/map_reduce_final}"
WINDOW="${2:-64}"
HORIZON="${3:-12}"

echo "=== Kalman Filter Baseline Evaluation ==="
echo "Data directory: $FINAL_DIR"
echo "Window size: $WINDOW"
echo "Horizon: $HORIZON"
echo ""

# Check if tuning is requested
if [[ "${4:-}" == "--tune" ]]; then
    echo "Running with hyperparameter tuning..."
    python -m src.baselines.train_kalman \
        --final_dir "$FINAL_DIR" \
        --window "$WINDOW" \
        --horizon "$HORIZON" \
        --max_files 500 \
        --max_windows 10000 \
        --tune
else
    echo "Running with default parameters..."
    python -m src.baselines.train_kalman \
        --final_dir "$FINAL_DIR" \
        --window "$WINDOW" \
        --horizon "$HORIZON" \
        --max_files 500 \
        --max_windows 10000 \
        --process_noise_pos 1e-5 \
        --process_noise_vel 1e-4 \
        --measurement_noise 1e-4
fi

echo ""
echo "=== Evaluation Complete ==="
echo "Results saved to metrics/kalman_filter.json"
