#!/usr/bin/env bash
set -euo pipefail

PROC_TRAJ="data/processed/traj_w64_h12"
PROC_ETA="data/processed/eta_w64"
PROC_ANOM="data/processed/anom_w64_h12"
CKPT_DIR="data/checkpoints"
FIG_DIR="data/figures"

mkdir -p "$CKPT_DIR" "$FIG_DIR"

echo "=== [1/5] Build processed datasets ==="
bash scripts/make_processed.sh \
  --interim data/interim/interim.parquet \
  --task trajectory --window 64 --horizon 12 \
  --out "$PROC_TRAJ"

bash scripts/make_processed.sh \
  --interim data/interim/interim.parquet \
  --task eta --window 64 \
  --out "$PROC_ETA"

bash scripts/make_processed.sh \
  --interim data/interim/interim.parquet \
  --task anomaly --window 64 --horizon 12 \
  --out "$PROC_ANOM"

echo "=== [2/5] Train trajectory: GRU ==="
python -m src.train.train_traj --config configs/traj_gru_small.yaml

echo "=== [3/5] Train trajectory: TPTrans ==="
python -m src.train.train_traj --config configs/traj_tptrans_base.yaml

echo "=== [4/5] Train ETA (GRU) ==="
python -m src.train.train_eta --config configs/eta_gru.yaml

echo "=== [5/5] Evaluate ==="
# Trajectory evals (+ plots)
python -m src.eval.evaluate_traj \
  --processed_dir "$PROC_TRAJ" \
  --ckpt "$CKPT_DIR/traj_gru.pt" \
  --model gru \
  --plot

python -m src.eval.evaluate_traj \
  --processed_dir "$PROC_TRAJ" \
  --ckpt "$CKPT_DIR/traj_tptrans.pt" \
  --model tptrans \
  --plot

# ETA eval
python -m src.eval.evaluate_eta \
  --processed_dir "$PROC_ETA" \
  --ckpt "$CKPT_DIR/eta_model.pt"

# Anomaly training + eval
# Train a self-supervised forecaster (uses GRU baseline under the hood)
python -m src.train.train_anom --processed_dir "$PROC_ANOM" --out_dir "$CKPT_DIR"

# Evaluate anomaly via planted anomalies + AUROC/AUPRC
python -m src.eval.evaluate_anom \
  --processed_dir "$PROC_ANOM" \
  --ckpt "$CKPT_DIR/anom_gru.pt"

echo "=== Done. See data/figures for plots and metrics/*.json for scores. ==="