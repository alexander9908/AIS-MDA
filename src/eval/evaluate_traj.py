# src/eval/evaluate_traj.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

from ..models import GRUSeq2Seq, TPTrans
from .metrics_traj import ade, fde


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed/traj_w64_h12")
    ap.add_argument("--ckpt", default="data/checkpoints/traj_model.pt")
    ap.add_argument("--model", choices=["gru", "tptrans"], default="tptrans")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    X = np.load(Path(args.processed_dir) / "X.npy")
    Y = np.load(Path(args.processed_dir) / "Y.npy")

    n = len(X)
    idx_split = int(n * (1 - args.val_frac))
    Xva, Yva = X[idx_split:], Y[idx_split:]

    feat_dim = X.shape[-1]
    horizon = Y.shape[1]

    if args.model == "gru":
        model = GRUSeq2Seq(feat_dim, d_model=128, layers=2, horizon=horizon)
    else:
        model = TPTrans(feat_dim, d_model=192, nhead=4, enc_layers=4, dec_layers=2, horizon=horizon)

    state = torch.load(args.ckpt, map_location="cpu")
    try:
        model.load_state_dict(state)
    except Exception:
        model.load_state_dict(state, strict=False)

    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(Xva).float()).numpy()

    print(f"VAL: ADE={ade(pred, Yva):.3f}  FDE={fde(pred, Yva):.3f}")

    if args.plot:
        import matplotlib.pyplot as plt
        sel = np.random.choice(len(Xva), size=min(6, len(Xva)), replace=False)
        for i, idx in enumerate(sel, 1):
            true = Yva[idx].cumsum(axis=0)
            predseq = pred[idx].cumsum(axis=0)
            plt.figure()
            plt.plot(true[:, 0], true[:, 1], label="true")
            plt.plot(predseq[:, 0], predseq[:, 1], label="pred", linestyle="--")
            plt.legend()
            plt.title(f"Trajectory sample {idx_split + idx}")
            out = Path("data/figures") / f"traj_sample_{idx_split + idx}.png"
            plt.savefig(out, bbox_inches="tight")
            plt.close()
            print(f"Saved {out}")


if __name__ == "__main__":
    main()