# src/eval/eval_traj_abs.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models import GRUSeq2Seq, TPTrans
from ..utils.datasets_new import AISDataset
from .metrics_traj import ade, fde
import matplotlib.pyplot as plt

def build_model(kind, feat_dim, horizon, d_model=192, nhead=4, enc_layers=4, dec_layers=2):
    if kind == "gru":
        return GRUSeq2Seq(feat_dim, d_model=d_model, layers=2, horizon=horizon)
    return TPTrans(feat_dim, d_model, nhead, enc_layers, dec_layers, horizon)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", choices=["gru","tptrans"], default="tptrans")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_plots", type=int, default=8)
    ap.add_argument("--out_dir", default="data/figures")
    args = ap.parse_args()

    ds = AISDataset(args.split_dir, max_seqlen=96)  # returns (X, Y_deltas) thanks to adapter
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    X0, Y0 = ds[0]
    feat_dim, horizon = X0.shape[-1], Y0.shape[0]

    model = build_model(args.model, feat_dim, horizon)
    state = torch.load(args.ckpt, map_location="cpu")
    try:
        model.load_state_dict(state, strict=True)
    except Exception as e:
        print(f"[warn] strict load failed: {e}\n[info] retrying strict=False")
        model.load_state_dict(state, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    preds, gts = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            yp = model(xb)  # [B,H,2] (deltas)
            preds.append(yp.cpu().numpy())
            gts.append(yb.cpu().numpy())

    P = np.concatenate(preds, 0)
    Y = np.concatenate(gts, 0)
    ade_val, fde_val = float(ade(P,Y)), float(fde(P,Y))
    print(f"VAL: ADE={ade_val:.3f}  FDE={fde_val:.3f}")

    Path("metrics").mkdir(exist_ok=True)
    out_json = Path("metrics") / f"traj_{args.model}_{Path(args.split_dir).name}.json"
    out_json.write_text(json.dumps({
        "task":"trajectory","model":args.model,"split":args.split_dir,
        "ckpt":args.ckpt,"ade":ade_val,"fde":fde_val,"count":int(len(Y)),
    }, indent=2))
    print(f"[metrics] wrote {out_json}")

    # ---- Plots (absolute) ----
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    num_plots = min(args.max_plots, len(ds))
    for i in range(num_plots):
        # absolute tracks via helper
        past_abs, true_abs = ds.get_abs_segments(i)  # (T,2), (H,2) each as (lon,lat)

        # predicted absolute = last past + cumsum(pred deltas)
        # need a forward pass for this sample:
        x, y = ds[i]
        with torch.no_grad():
            yp = model(x.unsqueeze(0).to(device).float()).cpu().numpy()[0]  # (H,2)
        pred_abs = past_abs[-1] + np.cumsum(yp, axis=0)

        plt.figure(figsize=(6,5))
        plt.plot(past_abs[:,0], past_abs[:,1], "b-", label="past (input)")
        plt.plot(past_abs[-1,0], past_abs[-1,1], "ko", markersize=4, label="current pos")
        plt.plot(true_abs[:,0], true_abs[:,1], "g-", label="true future")
        plt.plot(pred_abs[:,0], pred_abs[:,1], "r--", label="pred future")
        plt.xlabel("Longitude"); plt.ylabel("Latitude")
        plt.title(f"Trajectory sample {i}")
        plt.legend()
        fp = out_dir / f"traj_full_{args.model}_{Path(args.split_dir).name}_{i}.png"
        plt.savefig(fp, bbox_inches="tight"); plt.close()
        print(f"[plot] saved {fp}")

if __name__ == "__main__":
    main()
