# src/eval/evaluate_traj.py
from __future__ import annotations
import argparse
from pathlib import Path
import json
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

    proc = Path(args.processed_dir)
    X = np.load(proc / "X.npy")
    Y = np.load(proc / "Y.npy")

    # Optional: apply scaler (if present)
    scaler_path = proc / "scaler.npz"
    if scaler_path.exists():
        s = np.load(scaler_path)
        X = (X - s["mean"]) / (s["std"] + 1e-8)

    # Split: prefer MMSI-based split if available
    mmsi_path = proc / "window_mmsi.npy"
    use_mmsi_split = mmsi_path.exists()
    if use_mmsi_split:
        win_mmsi = np.load(mmsi_path)
        uniq = np.unique(win_mmsi)
        rng = np.random.default_rng(42)
        rng.shuffle(uniq)
        n_val_vess = max(1, int(len(uniq) * args.val_frac))
        val_vess = set(uniq[:n_val_vess])
        val_idx = np.nonzero(np.isin(win_mmsi, list(val_vess)))[0]
        Xva, Yva = X[val_idx], Y[val_idx]
    else:
        n = len(X)
        idx_split = int(n * (1.0 - args.val_frac))
        Xva, Yva = X[idx_split:], Y[idx_split:]

    # Build model
    feat_dim = X.shape[-1]
    horizon = Y.shape[1]
    if args.model == "gru":
        model = GRUSeq2Seq(feat_dim, d_model=128, layers=2, horizon=horizon)
    else:
        model = TPTrans(feat_dim, d_model=192, nhead=4, enc_layers=4, dec_layers=2, horizon=horizon)

    # Load checkpoint with architecture auto-detection
    state = torch.load(args.ckpt, map_location="cpu")

    def _build(kind: str):
        if kind == "gru":
            return GRUSeq2Seq(feat_dim, d_model=128, layers=2, horizon=horizon)
        else:
            return TPTrans(feat_dim, d_model=192, nhead=4, enc_layers=4, dec_layers=2, horizon=horizon)

    # We already have a model built per args.model, but keep a builder for fallback
    loaded_as = args.model
    try:
        model.load_state_dict(state)
    except Exception as e_first:
        # try the other architecture
        other = "gru" if args.model == "tptrans" else "tptrans"
        model = _build(other)
        try:
            model.load_state_dict(state)
            loaded_as = other
            print(f"[info] Loaded checkpoint as {other} (auto-detected).")
        except Exception as e_second:
            # last attempt: strict=False on requested model
            model = _build(args.model)
            try:
                model.load_state_dict(state, strict=False)
                print("[warn] Loaded with strict=False; keys/sizes mismatched.")
            except Exception as e_final:
                raise RuntimeError(
                    f"Could not load checkpoint as '{args.model}' or '{other}'.\n"
                    f"First error: {e_first}\nSecond error: {e_second}\nFinal error: {e_final}"
                )

    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(Xva).float()).numpy()

    print(f"VAL: ADE={ade(pred, Yva):.3f}  FDE={fde(pred, Yva):.3f}")

    if args.plot:
        import matplotlib.pyplot as plt
        figdir = Path("data/figures")
        figdir.mkdir(parents=True, exist_ok=True)

        sel = np.random.choice(len(Xva), size=min(6, len(Xva)), replace=False)
        for i, local_idx in enumerate(sel, 1):
            true = Yva[local_idx].cumsum(axis=0)
            predseq = pred[local_idx].cumsum(axis=0)

            plt.figure()
            plt.plot(true[:, 0], true[:, 1], label="true")
            plt.plot(predseq[:, 0], predseq[:, 1], label="pred", linestyle="--")
            plt.legend()

            # Use a robust global index for filename
            if use_mmsi_split:
                global_idx = int(val_idx[local_idx])
            else:
                global_idx = int(idx_split + local_idx)

            plt.title(f"Trajectory sample {global_idx}")
            out = figdir / f"traj_sample_{global_idx}.png"
            plt.savefig(out, bbox_inches="tight")
            plt.close()
            print(f"Saved {out}")


if __name__ == "__main__":
    main()