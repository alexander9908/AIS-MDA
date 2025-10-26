from __future__ import annotations
import argparse, math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from ..config import load_config
from ..models import GRUSeq2Seq, TPTrans

def huber_loss(pred, target, delta: float = 1.0):
    return nn.SmoothL1Loss(beta=delta)(pred, target)

def main(cfg_path: str):
    cfg = load_config(cfg_path)
    # Placeholder: expect pre-saved tensors X.npy and Y.npy under processed dir
    processed = Path(cfg.get("processed_dir","/mnt/data/processed/traj/"))
    X = torch.from_numpy(__import__("numpy").load(processed/"X.npy"))
    Y = torch.from_numpy(__import__("numpy").load(processed/"Y.npy"))
    ds = TensorDataset(X, Y)
    dl = DataLoader(ds, batch_size=cfg.get("batch_size",128), shuffle=True)
    feat_dim = X.shape[-1]
    horizon = Y.shape[1]
    if cfg["model"]["name"] == "gru":
        model = GRUSeq2Seq(feat_dim, d_model=cfg["model"].get("d_model",128), layers=cfg["model"].get("layers",2), horizon=horizon)
    else:
        model = TPTrans(feat_dim, d_model=cfg["model"].get("d_model",192), nhead=cfg["model"].get("nhead",4),
                        enc_layers=cfg["model"].get("enc_layers",4), dec_layers=cfg["model"].get("dec_layers",2), horizon=horizon)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr",3e-4))
    model.train()
    for epoch in range(cfg.get("epochs",5)):
        total = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            pred = model(xb.float())
            loss = huber_loss(pred, yb.float())
            loss.backward(); opt.step()
            total += loss.item() * xb.size(0)
        print(f"epoch {epoch+1}: loss={total/len(ds):.4f}")
    out = Path(cfg.get("out_dir","/mnt/data/checkpoints"))
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "traj_model.pt")
    print(f"Saved model to {out/'traj_model.pt'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
