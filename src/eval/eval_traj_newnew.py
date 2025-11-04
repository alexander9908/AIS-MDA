# src/eval/eval_traj_newnew.py
from __future__ import annotations
import argparse, json, random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models import GRUSeq2Seq, TPTrans
from ..utils.datasets import AISDataset
from .metrics_traj import ade, fde


def build_model(kind: str, feat_dim: int, horizon: int,
                d_model=192, nhead=4, enc_layers=4, dec_layers=2):
    if kind == "gru":
        return GRUSeq2Seq(feat_dim, d_model=d_model, layers=2, horizon=horizon)
    else:
        return TPTrans(
            feat_dim=feat_dim,
            d_model=d_model,
            nhead=nhead,
            enc_layers=enc_layers,
            dec_layers=dec_layers,
            horizon=horizon,
        )


# --- replace your plot_samples() with this ---
def plot_samples(
    samples,
    model_kind: str,
    out_dir: Path,
    lat_idx: int,
    lon_idx: int,
    max_plots: int = 6,
    viz_match_lengths: bool = False,   # <— NEW
):
    import numpy as np
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    sel = list(range(min(max_plots, len(samples))))

    for local_i, idx in enumerate(sel, 1):
        x, y_abs, y_pred = samples[idx]  # x:[T,F] abs, y_abs:[H,2] abs, y_pred:[H,2] deltas or abs (we convert below)

        # past absolute lon/lat
        past_lon = x[:, lon_idx]
        past_lat = x[:, lat_idx]

        # starting point at end of the input window
        x0 = past_lon[-1]
        y0 = past_lat[-1]

        # Convert predicted to absolute positions relative to last input point
        # y_pred is model output in *deltas*, so cum-sum from (x0,y0)
        pred_deltas = np.cumsum(y_pred, axis=0)
        pred_lon = x0 + pred_deltas[:, 0]
        pred_lat = y0 + pred_deltas[:, 1]

        # True future is already absolute positions in your dataset adapter
        true_lon = y_abs[:, 0]
        true_lat = y_abs[:, 1]

        # --- Optional: visually match the final *length* of true to pred (for plotting only)
        if viz_match_lengths:
            # Compute end-to-end displacement magnitudes
            d_true = np.array([true_lon[-1] - x0, true_lat[-1] - y0])
            d_pred = np.array([pred_lon[-1] - x0, pred_lat[-1] - y0])
            norm_true = np.linalg.norm(d_true) + 1e-8
            norm_pred = np.linalg.norm(d_pred)

            scale = (norm_pred / norm_true) if norm_true > 0 else 1.0
            # scale *about* the current point (x0,y0)
            true_lon = x0 + (true_lon - x0) * scale
            true_lat = y0 + (true_lat - y0) * scale

        # --- Plot
        plt.figure(figsize=(6, 5))

        # Past (thicker + arrows for heading)
        plt.plot(past_lon, past_lat, color="#1f77b4", linewidth=2.0, label="past (input)")
        plt.plot([x0], [y0], "ko", markersize=4, label="current pos")

        # True & Pred futures with same horizon
        plt.plot(true_lon, true_lat, color="#2ca02c", linewidth=2.0, label="true future")
        plt.plot(pred_lon, pred_lat, "--", color="#d62728", linewidth=2.0, label="pred future")

        # Equal aspect so distances look right-ish in lon/lat
        plt.gca().set_aspect("equal", adjustable="box")

        # Auto-zoom to the union of past tail + both futures
        Xs = np.concatenate([past_lon[-8:], true_lon, pred_lon])
        Ys = np.concatenate([past_lat[-8:], true_lat, pred_lat])
        pad_x = (Xs.max() - Xs.min()) * 0.1 + 1e-6
        pad_y = (Ys.max() - Ys.min()) * 0.1 + 1e-6
        plt.xlim(Xs.min() - pad_x, Xs.max() + pad_x)
        plt.ylim(Ys.min() - pad_y, Ys.max() + pad_y)

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"Trajectory sample {idx}")
        plt.legend(loc="upper left")
        fig_path = out_dir / f"traj_full_{model_kind}_val_{idx}.png"
        plt.savefig(fig_path, bbox_inches="tight", dpi=120)
        plt.close()
        print(f"[plot] saved {fig_path}")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True, help="e.g. data/map_reduced/val")
    ap.add_argument("--ckpt", required=True, help="checkpoint .pt")
    ap.add_argument("--model", choices=["gru", "tptrans"], default="tptrans")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_plots", type=int, default=6)
    ap.add_argument("--lat_idx", type=int, default=1, help="index of latitude in x features")
    ap.add_argument("--lon_idx", type=int, default=0, help="index of longitude in x features")
    ap.add_argument("--out_dir", default="data/figures")
    ap.add_argument("--viz_match_lengths", action="store_true",
                help="Scale the true future so its end-to-end length matches the prediction (viz only)")
    args = ap.parse_args()

    split = Path(args.split_dir)
    out_dir = Path(args.out_dir)

    # dataset from map-reduced pkl files (no features= kw)
    ds = AISDataset(str(split), max_seqlen=96)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # support (x,y) or (x,y,meta)
    first = ds[0]
    if isinstance(first, (list, tuple)) and len(first) >= 2:
        x0, y0 = first[0], first[1]
    else:
        raise RuntimeError("AISDataset.__getitem__ must return (x,y) or (x,y,meta).")

    feat_dim = x0.shape[-1]
    horizon = y0.shape[0]

    model = build_model(args.model, feat_dim, horizon)
    state = torch.load(args.ckpt, map_location="cpu")
    try:
        model.load_state_dict(state, strict=True)
    except Exception as e:
        print(f"[warn] strict load failed: {e}\n[info] retrying strict=False")
        model.load_state_dict(state, strict=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    preds, gts = [], []
    keep_for_plot = []

    with torch.no_grad():
        for batch in dl:
            # handle (x,y) or (x,y,meta)
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                xb, yb = batch[0], batch[1]
            else:
                raise RuntimeError("Dataloader must yield (x,y) or (x,y,meta).")

            xb = xb.to(device).float()
            yb = yb.to(device).float()
            yp = model(xb)  # [B,H,2]

            yp_np = yp.cpu().numpy()
            yb_np = yb.cpu().numpy()
            preds.append(yp_np)
            gts.append(yb_np)

            if len(keep_for_plot) < args.max_plots:
                b = min(4, len(xb))
                for i in range(b):
                    keep_for_plot.append((xb[i].cpu().numpy(), yb_np[i], yp_np[i]))
                    if len(keep_for_plot) >= args.max_plots:
                        break

    pred = np.concatenate(preds, axis=0)
    Y = np.concatenate(gts, axis=0)

    ade_val = float(ade(pred, Y))
    fde_val = float(fde(pred, Y))
    print(f"VAL: ADE={ade_val:.3f}  FDE={fde_val:.3f}")

    metrics_dir = Path("metrics"); metrics_dir.mkdir(exist_ok=True)
    out_json = metrics_dir / f"traj_{args.model}_{split.name}.json"
    payload = {
        "task": "trajectory",
        "model": args.model,
        "split": str(split),
        "ckpt": str(args.ckpt),
        "ade": ade_val,
        "fde": fde_val,
        "count": int(len(Y)),
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"[metrics] wrote {out_json}")

    plot_samples(keep_for_plot, args.model, out_dir, args.lat_idx, args.lon_idx, args.max_plots, viz_match_lengths=args.viz_match_lengths,)


if __name__ == "__main__":
    main()
