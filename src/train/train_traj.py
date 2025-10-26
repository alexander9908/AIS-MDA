from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from ..config import load_config
from ..models import GRUSeq2Seq, TPTrans


def huber_loss(pred, target, delta: float = 1.0):
    # SmoothL1Loss uses 'beta' as the Huber delta
    return nn.SmoothL1Loss(beta=delta)(pred, target)


def main(cfg_path: str):
    cfg = load_config(cfg_path)

    processed = Path(cfg.get("processed_dir", "data/processed/traj_w64_h12"))
    out_dir = Path(cfg.get("out_dir", "data/checkpoints"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- Load arrays (NumPy first) --------
    X = np.load(processed / "X.npy")  # [N, T, F]
    Y = np.load(processed / "Y.npy")  # [N, H, 2]

    # Optional normalization (created by make_processed.py if you add it)
    scaler = processed / "scaler.npz"
    if scaler.exists():
        s = np.load(scaler)
        mean, std = s["mean"], s["std"]
        X = (X - mean) / (std + 1e-8)

    # -> tensors
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()

    ds = TensorDataset(X, Y)

    # Optional train/val split (controlled via config: val_frac)
    val_frac = float(cfg.get("val_frac", 0.0))
    if 0.0 < val_frac < 1.0 and len(ds) > 1:
        n_val = max(1, int(len(ds) * val_frac))
        n_train = len(ds) - n_val
        ds_train, ds_val = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
        has_val = True
    else:
        ds_train, ds_val = ds, None
        has_val = False

    dl_train = DataLoader(ds_train, batch_size=int(cfg.get("batch_size", 128)), shuffle=True, num_workers=0, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=int(cfg.get("batch_size", 128)), shuffle=False, num_workers=0, pin_memory=True) if has_val else None

    feat_dim = X.shape[-1]
    horizon = Y.shape[1]

    if cfg["model"]["name"] == "gru":
        model = GRUSeq2Seq(
            feat_dim,
            d_model=cfg["model"].get("d_model", 128),
            layers=cfg["model"].get("layers", 2),
            horizon=horizon,
        )
        model_name = "GRU"
    else:
        model = TPTrans(
            feat_dim,
            d_model=cfg["model"].get("d_model", 192),
            nhead=cfg["model"].get("nhead", 4),
            enc_layers=cfg["model"].get("enc_layers", 4),
            dec_layers=cfg["model"].get("dec_layers", 2),
            horizon=horizon,
        )
        model_name = "TPTrans"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 3e-4)))
    scaler_amp = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    epochs = int(cfg.get("epochs", 5))
    clip_norm = float(cfg.get("clip_norm", 1.0))
    delta = float(cfg.get("huber_delta", 1.0))

    best_val = float("inf")
    best_path = out_dir / "traj_model.pt"  # will hold best if val exists, else last

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in dl_train:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pred = model(xb)
                loss = huber_loss(pred, yb, delta=delta)

            scaler_amp.scale(loss).backward()
            # Gradient clipping for stability
            scaler_amp.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler_amp.step(opt)
            scaler_amp.update()

            total += loss.item() * xb.size(0)

        train_loss = total / len(ds_train)
        msg = f"epoch {epoch}: train_loss={train_loss:.4f}"

        if has_val:
            model.eval()
            vtotal = 0.0
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                for xb, yb in dl_val:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    pred = model(xb)
                    vtotal += huber_loss(pred, yb, delta=delta).item() * xb.size(0)
            val_loss = vtotal / len(ds_val)
            msg += f"  val_loss={val_loss:.4f}"
            # Save best
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), best_path)
        else:
            # No val set → keep overwriting; last epoch wins
            torch.save(model.state_dict(), best_path)

        print(msg)

    print(f"Saved model to {best_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)