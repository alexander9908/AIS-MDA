#!/bin/bash
# Evaluate MTM pretrained model reconstruction quality

set -e

echo "============================================"
echo "Evaluating MTM Pretrained Model"
echo "============================================"

# Default arguments
FINAL_DIR="${1:-data/map_reduce_final}"
CKPT="${2:-data/checkpoints/traj_mtm.pt}"
NUM_SAMPLES="${3:-1000}"

python -m src.eval.evaluate_mtm \
    --final_dir "$FINAL_DIR" \
    --ckpt "$CKPT" \
    --window 64 \
    --batch_size 128 \
    --d_model 192 \
    --nhead 4 \
    --enc_layers 4 \
    --mask_ratio 0.12 \
    --num_samples "$NUM_SAMPLES" \
    --vis_samples 5 \
    --out_dir data/figures \
    --metrics_dir metrics

echo ""
echo "✓ Evaluation complete!"
echo "  - Metrics saved to: metrics/mtm_reconstruction.json"
echo "  - Visualizations saved to: data/figures/mtm_reconstruction_samples.png"
