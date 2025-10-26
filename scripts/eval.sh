#!/usr/bin/env bash
set -euo pipefail

python -m src.eval.evaluate_traj --processed_dir data/processed/traj_w64_h12 --ckpt data/checkpoints/traj_model.pt --model tptrans --plot
python -m src.eval.evaluate_eta   --processed_dir data/processed/eta_w64      --ckpt data/checkpoints/eta_model.pt
