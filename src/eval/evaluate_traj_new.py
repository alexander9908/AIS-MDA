# src/eval/evaluate_traj_new.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.utils.datasets import AISDataset
from ..models.tptrans_new import TPTransNew
from ..models import GRUSeq2Seq  # keep GRU baseline available
from .metrics_traj import ade, fde
from ..models.tptrans_new import TPTransNew

def main():
    ap = argparse.ArgumentParser(description="Evaluate trajectory model on Map-Reduce dataset (AISDataset).")
    ap.add_argument("--split_dir", required=True, help="e.g., data/map_reduced/val or .../test")
    ap.add_argument("--ckpt", required=True, help="checkpoint path, e.g., data/checkpoints/traj_tptrans.pt")
    ap.add_argument("--model", choices=["tptrans","gru"], default="tptrans")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    # Dataset (must yield (X, Y) with Y future deltas [H,2])
    ds = AISDataset(args.split_dir, max_seqlen=96)
    if len(ds) == 0:
        raise RuntimeError(f"No samples found in {args.split_dir}")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    xb0, yb0 = next(iter(dl))
    feat_dim = xb0.shape[-1]
    horizon = yb0.shape[1]

    # inside main(), after you infer feat_dim/horizon:
    def build_tptrans(dim_ff=2048, use_posenc=True):
        return TPTransNew(
            feat_dim, d_model=192, nhead=4, enc_layers=4, dec_layers=2,
            horizon=horizon, dim_ff=dim_ff, use_posenc=use_posenc
        )
    
    if args.model == "gru":
        model = GRUSeq2Seq(feat_dim, d_model=128, layers=2, horizon=horizon)
    else:
        # First try with dim_ff=2048 (matches older checkpoints), posenc disabled for compatibility
        model = build_tptrans(dim_ff=2048, use_posenc=False)
    
    state = torch.load(args.ckpt, map_location="cpu")
    try:
        model.load_state_dict(state, strict=True)
    except Exception as e1:
        print(f"[warn] strict load failed: {e1}\n[info] retrying strict=False")
        try:
            model.load_state_dict(state, strict=False)
        except Exception as e2:
            # last resort: try alternate dim_ff=4*d_model (768)
            if args.model == "tptrans":
                alt = build_tptrans(dim_ff=4*192, use_posenc=False)
                try:
                    alt.load_state_dict(state, strict=False)
                    model = alt
                    print("[info] Loaded with dim_ff=768 (4*d_model), strict=False.")
                except Exception as e3:
                    raise RuntimeError(f"Could not load checkpoint:\n  strict err: {e1}\n  non-strict err: {e2}\n  alt err: {e3}")
            else:
                raise


    if args.model == "gru":
        model = GRUSeq2Seq(feat_dim, d_model=128, layers=2, horizon=horizon)
    else:
        model = TPTransNew(feat_dim, d_model=192, nhead=4, enc_layers=4, dec_layers=2, horizon=horizon)

    state = torch.load(args.ckpt, map_location="cpu")
    try:
        model.load_state_dict(state, strict=True)
    except Exception as e:
        # try non-strict if you changed minor things
        print(f"[warn] strict load failed: {e}\n[info] retrying strict=False")
        model.load_state_dict(state, strict=False)

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in dl:
            yhat = model(xb.float())
            preds.append(yhat.numpy())
            trues.append(yb.numpy())

    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)

    a = float(ade(pred, true))
    f = float(fde(pred, true))
    print(f"[EVAL] {args.split_dir}  ADE={a:.3f}  FDE={f:.3f}")

    Path("metrics").mkdir(exist_ok=True)
    name = f"traj_{args.model}_{Path(args.split_dir).name}.json"
    (Path("metrics")/name).write_text(json.dumps({
        "task":"trajectory", "model":args.model, "split":args.split_dir,
        "ckpt":args.ckpt, "ade":a, "fde":f
    }, indent=2))
    print(f"[metrics] wrote metrics/{name}")

    if args.plot:
        import matplotlib.pyplot as plt
        figdir = Path("data/figures"); figdir.mkdir(parents=True, exist_ok=True)

        # Randomly pick some samples
        sel = np.random.choice(len(pred), size=min(6, len(pred)), replace=False)

        for idx in sel:
            # Retrieve the input sequence (X) and true Y from the dataset
            x, y_true = ds[idx]
            # Convert tensors to numpy
            x = x.numpy()        # [T_in, F]
            y_true = y_true.numpy()  # [H, 2]
            y_pred = pred[idx]       # [H, 2]

            # --- Get base ship path in Lat/Lon ---
            # Assuming your feature order is [LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI]
            lat_idx, lon_idx = 0, 1

            past_path = x[:, [lat_idx, lon_idx]]

            # Compute absolute trajectories for true and predicted futures
            true_future = past_path[-1] + np.cumsum(y_true, axis=0)
            pred_future = past_path[-1] + np.cumsum(y_pred, axis=0)

            # --- Plot ---
            plt.figure(figsize=(6,5))
            plt.plot(past_path[:,1], past_path[:,0], 'b-', label="past (input)")
            plt.plot(true_future[:,1], true_future[:,0], 'g-', label="true future")
            plt.plot(pred_future[:,1], pred_future[:,0], 'r--', label="pred future")
            plt.scatter(past_path[-1,1], past_path[-1,0], c='k', s=20, label="current pos")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.legend()
            plt.title(f"Trajectory sample {idx}")
            out = figdir / f"traj_full_{Path(args.split_dir).name}_{idx}.png"
            plt.savefig(out, bbox_inches="tight")
            plt.close()
            print(f"Saved {out}")

# old plot
#    if args.plot:
#        import matplotlib.pyplot as plt
#        figdir = Path("data/figures"); figdir.mkdir(parents=True, exist_ok=True)
#        # Make a few plots (in delta-space cum-sum)
#        sel = np.random.choice(len(pred), size=min(6, len(pred)), replace=False)
#        for idx in sel:
#            t = true[idx].cumsum(axis=0)
#            p = pred[idx].cumsum(axis=0)
#            plt.figure()
#            plt.plot(t[:,0], t[:,1], label="true")
#            plt.plot(p[:,0], p[:,1], "--", label="pred")
#            plt.legend()
#            out = figdir / f"traj_sample_{Path(args.split_dir).name}_{int(idx)}.png"
#            plt.savefig(out, bbox_inches="tight"); plt.close()
#            print(f"Saved {out}")

if __name__ == "__main__":
    main()
