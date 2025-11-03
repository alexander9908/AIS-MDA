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


def build_model(kind: str, feat_dim: int, horizon: int, d_model=192, nhead=4, enc_layers=4, dec_layers=2):
    if kind == "gru":
        return GRUSeq2Seq(feat_dim, d_model=d_model, layers=2, horizon=horizon)
    return TPTrans(feat_dim=feat_dim, d_model=d_model, nhead=nhead,
                   enc_layers=enc_layers, dec_layers=dec_layers, horizon=horizon)

def plot_samples(samples, model_kind: str, out_dir: Path, lat_idx: int, lon_idx: int, past_len: int, max_plots: int = 8):
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    sel = list(range(min(max_plots, len(samples))))
    for i, idx in enumerate(sel):
        x, y_abs, y_pred_abs = samples[idx]       # x:[T,F], y_abs:[H,2], y_pred_abs:[H,2]
        past_lon = x[:past_len, lon_idx]
        past_lat = x[:past_len, lat_idx]
        x0, y0 = past_lon[-1], past_lat[-1]       # last past point

        true_lon = y_abs[:, 0]
        true_lat = y_abs[:, 1]
        pred_lon = y_pred_abs[:, 0]
        pred_lat = y_pred_abs[:, 1]

        plt.figure(figsize=(6,5))
        # past
        plt.plot(past_lon, past_lat, "b-", label="past (input)")
        plt.plot([x0],[y0], "ko", label="current pos", markersize=4)
        # futures (absolute positions)
        plt.plot(true_lon, true_lat, "g-", label="true future")
        plt.plot(pred_lon, pred_lat, "r--", label="pred future")
        plt.xlabel("Longitude"); plt.ylabel("Latitude")
        plt.title(f"Trajectory sample {idx}")
        plt.legend()
        fig_path = out_dir / f"traj_full_{model_kind}_val_{idx}.png"
        plt.savefig(fig_path, bbox_inches="tight"); plt.close()
        print(f"[plot] saved {fig_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", choices=["gru","tptrans"], default="tptrans")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_plots", type=int, default=8)
    ap.add_argument("--lat_idx", type=int, default=1)   # x feature order: [lon,lat,sog,cog]
    ap.add_argument("--lon_idx", type=int, default=0)
    ap.add_argument("--past_len", type=int, default=64) # how much of the past window to draw
    ap.add_argument("--out_dir", default="data/figures")
    args = ap.parse_args()

    split = Path(args.split_dir)
    ds = AISDataset(str(split), max_seqlen=max(96, args.past_len + 12))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    x0, y0 = ds[0]                       # y0 is ABSOLUTE positions (lon,lat) for the horizon
    feat_dim = x0.shape[-1]; horizon = y0.shape[0]

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
        for xb, yb in dl:
            xb = xb.to(device).float()
            yb = yb.to(device).float()         # ABS positions
            yp = model(xb)                     # predict ABS positions (training matched this)
            yp = yp.cpu().numpy(); yb = yb.cpu().numpy()
            preds.append(yp); gts.append(yb)
            if len(keep_for_plot) < args.max_plots:
                for i in range(min(4, len(xb))):
                    keep_for_plot.append((xb[i].cpu().numpy(), yb[i], yp[i]))
                    if len(keep_for_plot) >= args.max_plots: break

    pred = np.concatenate(preds, axis=0)
    Y    = np.concatenate(gts,   axis=0)
    ade_val = float(ade(pred, Y))
    fde_val = float(fde(pred, Y))
    print(f"VAL: ADE={ade_val:.3f}  FDE={fde_val:.3f}")

    metrics_dir = Path("metrics"); metrics_dir.mkdir(exist_ok=True)
    out_json = metrics_dir / f"traj_{args.model}_{split.name}.json"
    out_json.write_text(json.dumps({
        "task":"trajectory","model":args.model,"split":str(split),
        "ckpt":str(args.ckpt),"ade":ade_val,"fde":fde_val,"count":int(len(Y))
    }, indent=2))
    print(f"[metrics] wrote {out_json}")

    plot_samples(keep_for_plot, args.model, Path(args.out_dir),
                 args.lat_idx, args.lon_idx, past_len=args.past_len, max_plots=args.max_plots)

if __name__ == "__main__":
    main()
