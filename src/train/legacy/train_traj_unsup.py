# src/train/train_traj_new.py
from __future__ import annotations
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils.datasets import AISDataset

from ..config import load_config
from ..models import GRUSeq2Seq
from ..models.tptrans_new import TPTransNew  # <- new TPTrans variant with pos enc, norm_first, etc.

def huber_loss(pred, target, delta: float = 1.0):
    # SmoothL1Loss uses 'beta' as the Huber delta
    return nn.SmoothL1Loss(beta=delta)(pred, target)

def main(cfg_path: str):
    cfg = load_config(cfg_path)

    pre_dir = Path(cfg.get("processed_dir", "data/map_reduced/"))
    out_dir = Path(cfg.get("out_dir", "data/checkpoints"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # robust path joining
    train_dir = pre_dir / "train"
    val_dir   = pre_dir / "val"

    # load Map-Reduce dataset (same class as training)
    ds_train = AISDataset(str(train_dir), max_seqlen=96)
    ds_val   = AISDataset(str(val_dir),   max_seqlen=96)
    has_val = len(ds_val) > 0

    dl_train = DataLoader(ds_train,
                          batch_size=int(cfg.get("batch_size", 128)),
                          shuffle=True, num_workers=0, pin_memory=True)
    dl_val   = (DataLoader(ds_val,
                           batch_size=int(cfg.get("batch_size", 128)),
                           shuffle=False, num_workers=0, pin_memory=True)
                if has_val else None)

    # infer feature dim and horizon from a sample
    x0, y0 = ds_train[0]
    feat_dim = x0.shape[-1]
    horizon  = cfg.get("horizon", y0.shape[0])

    # build model
    model_name = cfg["model"]["name"].lower()
    if model_name == "gru":
        model = GRUSeq2Seq(
            feat_dim,
            d_model=cfg["model"].get("d_model", 128),
            layers=cfg["model"].get("layers", 2),
            horizon=horizon,
        )
        nice_name = "GRU"
    else:
        model = TPTransNew(
            feat_dim,
            d_model=cfg["model"].get("d_model", 192),
            nhead=cfg["model"].get("nhead", 4),
            enc_layers=cfg["model"].get("enc_layers", 4),
            dec_layers=cfg["model"].get("dec_layers", 2),
            horizon=horizon,
        )
        nice_name = "TPTrans"

    # optional warm-start from MSP
    warm_path = cfg.get("warm_start_msp")
    if warm_path and nice_name == "TPTrans":
        from ..models.tptrans_unsup_new import TPTransMSPNew
        msp = TPTransMSPNew(
            feat_dim=feat_dim,
            d_model=cfg["model"].get("d_model", 192),
            nhead=cfg["model"].get("nhead", 4),
            enc_layers=cfg["model"].get("enc_layers", 4),
        )
        msp.load_state_dict(torch.load(warm_path, map_location="cpu"), strict=False)
        # copy encoder stack (conv + transformer) into supervised model
        model.conv.load_state_dict(msp.conv.state_dict(), strict=False)
        model.encoder.load_state_dict(msp.encoder.state_dict(), strict=False)
        print("[init] Warm-started TPTrans from MSP weights:", warm_path)

    ckpt_path = out_dir / f"traj_{nice_name.lower()}.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 3e-4)))
    scaler_amp = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    epochs = int(cfg.get("epochs", 5))
    clip_norm = float(cfg.get("clip_norm", 1.0))
    delta = float(cfg.get("huber_delta", 1.0))

    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        # ---------- train ----------
        model.train()
        total = 0.0
        seen = 0
        for xb, yb in dl_train:
            xb = xb.to(device, non_blocking=True).float()
            yb = yb.to(device, non_blocking=True).float()

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pred = model(xb)
                loss = huber_loss(pred, yb, delta=delta)

            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler_amp.step(opt)
            scaler_amp.update()

            total += float(loss) * xb.size(0)
            seen  += xb.size(0)

        train_loss = total / max(1, seen)
        msg = f"epoch {epoch}: train_loss={train_loss:.4f}"

        # ---------- val ----------
        if has_val:
            model.eval()
            vtot = 0.0; vseen = 0
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                for xb, yb in dl_val:
                    xb = xb.to(device, non_blocking=True).float()
                    yb = yb.to(device, non_blocking=True).float()
                    pred = model(xb)
                    vtot += float(huber_loss(pred, yb, delta=delta)) * xb.size(0)
                    vseen += xb.size(0)
            val_loss = vtot / max(1, vseen)
            msg += f"  val_loss={val_loss:.4f}"
            # Save best
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), ckpt_path)
        else:
            # No val -> always save the latest
            torch.save(model.state_dict(), ckpt_path)

        print(msg)

    print(f"Saved model to {ckpt_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
