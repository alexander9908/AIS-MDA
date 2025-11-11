# src/eval/eval_traj_abs.py
from __future__ import annotations
import argparse, json, random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models import GRUSeq2Seq, TPTrans
from ..utils.datasets_new import AISDataset  # your adapted dataset
from .metrics_traj import ade, fde


def build_model(kind: str, feat_dim: int, horizon: int, d_model=192, nhead=4, enc_layers=4, dec_layers=2):
    if kind == "gru":
        return GRUSeq2Seq(feat_dim, d_model=d_model, layers=2, horizon=horizon)
    return TPTrans(
        feat_dim=feat_dim,
        d_model=d_model,
        nhead=nhead,
        enc_layers=enc_layers,
        dec_layers=dec_layers,
        horizon=horizon,
    )


def plot_samples_abs(
    samples,
    ds: AISDataset,
    split_dir: Path,
    model_kind: str,
    out_dir: Path,
    lat_first: bool,
    past_extra: int,
    max_plots: int,
):
    """
    Plot *absolute* lon/lat:
      - Past: show extra historical points (past_extra) before the model window,
        if available from ds.get_original_item.
      - True future: the absolute Y points
      - Pred future: the absolute predictions
    """
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    sel = random.sample(range(len(samples)), k=min(max_plots, len(samples)))

    # indices for target order
    # dataset X order is [lat, lon, sog, cog]; targets were built as first 2 features
    LAT, LON = (0, 1) if lat_first else (1, 0)

    for k, idx in enumerate(sel):
        x_win, y_abs, y_pred_abs, orig = samples[idx]
        # x_win: [window, F] (past used in model)
        # y_abs: [H, 2] absolute (lat,lon in LAT,LON order if lat_first True)
        # y_pred_abs: [H, 2] absolute (same order)
        # orig: (seq, mask, seqlen, mmsi, t0) from ds.get_original_item

        # re-order y arrays to (lon,lat) for plotting
        true_lon = y_abs[:, LON]
        true_lat = y_abs[:, LAT]
        pred_lon = y_pred_abs[:, LON]
        pred_lat = y_pred_abs[:, LAT]

        # find the last past point (current position) from x_win (absolute!)
        past_lon = x_win[:, 1] if lat_first else x_win[:, 0]
        past_lat = x_win[:, 0] if lat_first else x_win[:, 1]
        x0, y0 = past_lon[-1], past_lat[-1]

        # try to get a longer history from the original item
        # (your AdaptedDataset exposes get_original_item)
        past_lon_full, past_lat_full = past_lon, past_lat
        if hasattr(ds, "get_original_item"):
            seq, mask, seqlen, *_ = ds.get_original_item(orig["orig_idx"])
            # seq is absolute [lat,lon,sog,cog] padded to max_seqlen
            valid = int(seqlen)
            start = max(0, valid - len(past_lon) - past_extra)
            lat_full = seq[start:valid, 0].numpy()
            lon_full = seq[start:valid, 1].numpy()
            past_lon_full, past_lat_full = lon_full, lat_full

        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 5))
        # past
        plt.plot(past_lon_full, past_lat_full, "b-", label="past (input)")
        plt.plot([x0], [y0], "ko", label="current pos", markersize=4)
        # futures (absolute)
        plt.plot(true_lon, true_lat, "g-", label="true future")
        plt.plot(pred_lon, pred_lat, "r--", label="pred future")

        plt.xlabel("Longitude"); plt.ylabel("Latitude")
        plt.title(f"Trajectory sample {idx}")
        plt.legend()
        fpath = out_dir / f"traj_full_{model_kind}_val_{idx}.png"
        plt.savefig(fpath, bbox_inches="tight"); plt.close()
        print(f"[plot] {fpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True, help="e.g. data/map_reduced/val")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", choices=["gru", "tptrans"], default="tptrans")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_plots", type=int, default=8)
    ap.add_argument("--lat_first", action="store_true",
                    help="Set if targets are [lat,lon]. If unset, treats as [lon,lat].")
    ap.add_argument("--past_extra", type=int, default=48,
                    help="additional past steps to plot (pulled from original sequence when available)")
    ap.add_argument("--out_dir", default="data/figures")
    args = ap.parse_args()

    split = Path(args.split_dir)
    out_dir = Path(args.out_dir)

    # Dataset & dataloader
    ds = AISDataset(str(split), max_seqlen=96)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # infer dims
    x0, y0 = ds[0]  # adapted outputs (X, Y)
    feat_dim, horizon = x0.shape[-1], y0.shape[0]

    # model
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

    # eval loop (absolute)
    preds, gts = [], []
    keep_for_plot = []
    with torch.no_grad():
        for start_idx, (xb, yb) in enumerate(dl):
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            yp = model(xb)  # absolute targets

            yp_np = yp.cpu().numpy()
            yb_np = yb.cpu().numpy()

            preds.append(yp_np)
            gts.append(yb_np)

            # keep a few samples for plotting with link to original
            if len(keep_for_plot) < args.max_plots and hasattr(ds, "get_original_item"):
                # store the original index so we can fetch longer history
                for i in range(min(8, len(xb))):
                    orig_idx = start_idx * dl.batch_size + i
                    keep_for_plot.append((
                        xb[i].cpu().numpy(),
                        yb_np[i],
                        yp_np[i],
                        {"orig_idx": orig_idx},
                    ))
                    if len(keep_for_plot) >= args.max_plots:
                        break

    pred = np.concatenate(preds, axis=0)
    Y = np.concatenate(gts, axis=0)

    # Reorder to common (lon,lat) just for ADE/FDE (they’re rotation/translation-invariant wrt ordering swap)
    LAT, LON = (0, 1) if args.lat_first else (1, 0)
    pred_ll = pred[..., [LON, LAT]]
    Y_ll    = Y[...,    [LON, LAT]]

    ade_val, fde_val = float(ade(pred_ll, Y_ll)), float(fde(pred_ll, Y_ll))
    print(f"VAL: ADE={ade_val:.3f}  FDE={fde_val:.3f}")

    metrics_dir = Path("metrics"); metrics_dir.mkdir(exist_ok=True)
    out_json = metrics_dir / f"traj_{args.model}_{split.name}_abs.json"
    out_json.write_text(json.dumps({
        "task": "trajectory",
        "model": args.model,
        "split": str(split),
        "ckpt": str(args.ckpt),
        "ade": ade_val,
        "fde": fde_val,
        "count": int(len(Y)),
        "targets": "absolute",
        "target_order": "[lat,lon]" if args.lat_first else "[lon,lat]"
    }, indent=2))
    print(f"[metrics] wrote {out_json}")

    plot_samples_abs(
        keep_for_plot, ds, split, args.model, out_dir,
        lat_first=args.lat_first,
        past_extra=args.past_extra,
        max_plots=args.max_plots,
    )


if __name__ == "__main__":
    main()
