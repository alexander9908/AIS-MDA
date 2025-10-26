from __future__ import annotations
import argparse, numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from ..config import load_config
from ..models import GRUSeq2Seq

def main(cfg_path: str):
    cfg = load_config(cfg_path)
    processed = Path(cfg.get("processed_dir","/mnt/data/processed/anom/"))
    X = torch.from_numpy(np.load(processed/"X.npy"))
    Y = torch.from_numpy(np.load(processed/"Y.npy"))  # next-step deltas as target
    ds = TensorDataset(X, Y)
    dl = DataLoader(ds, batch_size=cfg.get("batch_size",128), shuffle=True)
    model = GRUSeq2Seq(feat_dim=X.shape[-1], d_model=cfg["model"].get("d_model",128),
                       layers=cfg["model"].get("layers",2), horizon=Y.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr",1e-3))
    loss_fn = torch.nn.SmoothL1Loss()
    for epoch in range(cfg.get("epochs",5)):
        total = 0.0
        for xb, yb in dl:
            opt.zero_grad(); pred = model(xb.float()); loss = loss_fn(pred, yb.float())
            loss.backward(); opt.step(); total += loss.item()*xb.size(0)
        print(f"epoch {epoch+1}: loss={total/len(ds):.4f}")
    out = Path(cfg.get("out_dir","/mnt/data/checkpoints"))
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "anom_forecaster.pt")
    print(f"Saved model to {out/'anom_forecaster.pt'}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
