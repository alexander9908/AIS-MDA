from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..models import GRUSeq2Seq

def huber(delta=1.0):
    return nn.SmoothL1Loss(beta=delta)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed/anom_w64_h12")
    ap.add_argument("--out_dir", default="data/checkpoints")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    proc = Path(args.processed_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(proc/"X.npy")     # [N, T, F]
    Y = np.load(proc/"Y.npy")     # [N, H, 2]

    # input scaler
    s_path = proc/"scaler.npz"
    if s_path.exists():
        s = np.load(s_path); X = (X - s["mean"]) / (s["std"] + 1e-8)

    # target scaler
    ts_path = proc/"target_scaler.npz"
    y_mean = y_std = None
    if ts_path.exists():
        ts = np.load(ts_path)
        y_mean, y_std = ts["mean"], ts["std"]

    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()
    ds = TensorDataset(X, Y)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    feat_dim, horizon = X.shape[-1], Y.shape[1]
    model = GRUSeq2Seq(feat_dim, d_model=128, layers=2, horizon=horizon)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = huber(1.0)

    y_mean_t = torch.from_numpy(y_mean).view(1,1,-1).to(device) if y_mean is not None else None
    y_std_t  = torch.from_numpy(y_std).view(1,1,-1).to(device) if y_std is not None else None

    for epoch in range(1, args.epochs+1):
        model.train()
        total = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            if y_mean_t is not None:
                pred = (pred - y_mean_t) / y_std_t
                yb = (yb - y_mean_t) / y_std_t
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        print(f"epoch {epoch}: loss={total/len(ds):.4f}")

    ckpt = out_dir/"anom_gru.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"Saved anomaly model → {ckpt}")

if __name__ == "__main__":
    main()