#!/usr/bin/env bash
set -euo pipefail
python -m src.train.train_traj --config configs/traj_gru_small.yaml
python -m src.train.train_traj --config configs/traj_tptrans_base.yaml
python -m src.train.train_eta  --config configs/eta_gru.yaml
