from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn

from ..models import GRUSeq2Seq
from .metrics_eta import mae, mape, p95


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed/eta_w64")
    ap.add_argument("--ckpt", default="data/checkpoints/eta_model.pt")
    ap.add_argument("--val_frac", type=float, default=0.2)
    args = ap.parse_args()

    proc = Path(args.processed_dir)
    X = np.load(proc / "X.npy")
    y = np.load(proc / "y_eta.npy")

    n = len(X)
    idx_split = int(n * (1 - args.val_frac))
    Xva, yva = X[idx_split:], y[idx_split:]

    feat_dim = X.shape[-1]
    model = GRUSeq2Seq(feat_dim, d_model=128, layers=2, horizon=1)
    head = nn.Linear(2, 1)

    # Load checkpoint
    state = torch.load(args.ckpt, map_location="cpu")
    if isinstance(state, dict) and "model" in state and "head" in state:
        model.load_state_dict(state["model"])
        head.load_state_dict(state["head"])
    else:
        model.load_state_dict(state, strict=False)

    model.eval()
    head.eval()

    with torch.no_grad():
        pred_seq = model(torch.from_numpy(Xva).float())  # [B, 1, 2]
        pred = head(pred_seq[:, -1, :]).squeeze(-1).numpy()

    mae_val = float(mae(pred, yva))
    mape_val = float(mape(pred, yva))
    p95_val = float(p95(pred, yva))
    print(f"VAL: MAE={mae_val:.2f}  MAPE={mape_val*100:.2f}%  P95={p95_val:.2f}")

    # Save metrics to JSON
    mdir = Path("metrics")
    mdir.mkdir(parents=True, exist_ok=True)
    out_json = mdir / f"eta_gru_{Path(args.ckpt).stem}.json"
    res = {
        "task": "eta",
        "model": "gru",
        "ckpt": str(args.ckpt),
        "mae": mae_val,
        "mape": mape_val,
        "p95": p95_val,
    }
    out_json.write_text(json.dumps(res, indent=2))
    print(f"[metrics] wrote {out_json}")


if __name__ == "__main__":
    main()