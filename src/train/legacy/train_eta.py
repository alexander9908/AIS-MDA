from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import load_config
from ..models import GRUSeq2Seq


def main(cfg_path: str):
    cfg = load_config(cfg_path)

    processed_dir = Path(cfg.get("processed_dir", "data/processed/eta_w64"))
    out_dir = Path(cfg.get("out_dir", "data/checkpoints"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[train_eta] Loading data...")
    # Load arrays
    X = np.load(processed_dir / "X.npy")        # [N, T, F]
    y = np.load(processed_dir / "y_eta.npy")    # [N]
    print(f"[train_eta] Loaded X {X.shape}, y {y.shape}, dtype={X.dtype}")

    # Optional: feature normalization (expects scaler.npz created in make_processed.py)
    scaler_path = processed_dir / "scaler.npz"
    if scaler_path.exists():
        s = np.load(scaler_path)
        mean, std = s["mean"], s["std"]
        X = (X - mean) / (std + 1e-8)

    # -> tensors
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=cfg.get("batch_size", 256), shuffle=True)

    feat_dim = X.shape[-1]
    model = GRUSeq2Seq(
        feat_dim,
        d_model=cfg["model"].get("d_model", 128),
        layers=cfg["model"].get("layers", 2),
        horizon=1,
    )
    head = nn.Linear(2, 1)  # map last delta to ETA seconds

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    head.to(device)

    # IMPORTANT: optimize both model and head
    opt = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()),
        lr=cfg.get("lr", 1e-3)
    )
    loss_fn = nn.L1Loss()

    model.train()
    head.train()
    for epoch in range(cfg.get("epochs", 5)):
        total = 0.0
        for xb, yb in dl:
            xb = xb.float().to(device)
            yb = yb.float().to(device)

            opt.zero_grad()
            pred_seq = model(xb)                 # [B, 1, 2]
            pred = head(pred_seq[:, -1, :]).squeeze(-1)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

            total += loss.item() * xb.size(0)

        print(f"epoch {epoch+1}: mae={total/len(ds):.3f}")

    torch.save({"model": model.state_dict(), "head": head.state_dict()}, out_dir / "eta_model.pt")
    print(f"Saved model to {out_dir / 'eta_model.pt'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)