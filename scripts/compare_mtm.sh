#!/bin/bash
# Compare MTM-pretrained vs baseline trajectory prediction

set -e

echo "============================================"
echo "Comparing MTM-Pretrained vs Baseline Models"
echo "============================================"

PROCESSED_DIR="${1:-data/processed/traj_w64_h12/}"
CKPT_BASELINE="${2:-data/checkpoints/traj_tptrans_baseline.pt}"
CKPT_MTM="${3:-data/checkpoints/traj_tptrans.pt}"

python -m src.eval.compare_mtm \
    --processed_dir "$PROCESSED_DIR" \
    --ckpt_baseline "$CKPT_BASELINE" \
    --ckpt_mtm "$CKPT_MTM" \
    --batch_size 128 \
    --d_model 192 \
    --nhead 4 \
    --enc_layers 4 \
    --dec_layers 2 \
    --horizon 12 \
    --out_dir metrics

echo ""
echo "✓ Comparison complete!"
echo "  - Results saved to: metrics/mtm_comparison.json"
