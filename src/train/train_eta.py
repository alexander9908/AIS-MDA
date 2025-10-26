from __future__ import annotations
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from ..config import load_config
from ..models import GRUSeq2Seq

def main(cfg_path: str):
    cfg = load_config(cfg_path)
    processed = Path(cfg.get("processed_dir","/mnt/data/processed/eta/"))
    X = torch.from_numpy(__import__("numpy").load(processed/"X.npy"))
    y = torch.from_numpy(__import__("numpy").load(processed/"y_eta.npy"))
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=cfg.get("batch_size",256), shuffle=True)
    feat_dim = X.shape[-1]
    model = GRUSeq2Seq(feat_dim, d_model=cfg["model"].get("d_model",128), layers=cfg["model"].get("layers",2), horizon=1)
    head = nn.Linear(2, 1)  # map last delta to ETA seconds
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr",1e-3))
    loss_fn = nn.L1Loss()
    model.train()
    for epoch in range(cfg.get("epochs",5)):
        total = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            pred_seq = model(xb.float())  # [B,1,2]
            pred = head(pred_seq[:,-1,:]).squeeze(-1)
            loss = loss_fn(pred, yb.float())
            loss.backward(); opt.step()
            total += loss.item() * xb.size(0)
        print(f"epoch {epoch+1}: mae={total/len(ds):.3f}")
    out = Path(cfg.get("out_dir","/mnt/data/checkpoints"))
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "head": head.state_dict()}, out / "eta_model.pt")
    print(f"Saved model to {out/'eta_model.pt'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
