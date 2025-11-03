from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from src.utils.datasets import AISDataset

from ..config import load_config
from ..models import GRUSeq2Seq, TPTrans

def huber_loss(pred, target, delta: float = 1.0):
    # SmoothL1Loss uses 'beta' as the Huber delta
    return nn.SmoothL1Loss(beta=delta)(pred, target)


def main(cfg_path: str):
    cfg = load_config(cfg_path)

    preprocessed_dataset_dir = cfg.get('processed_dir')
    out_dir = Path(cfg.get("out_dir", "data/checkpoints"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_train = AISDataset(preprocessed_dataset_dir + "train", max_seqlen=96)
    ds_val = AISDataset(preprocessed_dataset_dir + "val", max_seqlen=96)
    has_val = True

    dl_train = DataLoader(ds_train, batch_size=int(cfg.get("batch_size", 128)), shuffle=True, num_workers=0, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=int(cfg.get("batch_size", 128)), shuffle=False, num_workers=0, pin_memory=True) if has_val else None
    
    feat_dim = ds_train[0][0].shape[-1]
    horizon = cfg.get("horizon", 12)

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

    # after you set model_name = "GRU" or "TPTrans"
    ckpt_name = f"traj_{model_name.lower()}.pt"
    best_path = out_dir / ckpt_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optional: load MTM-pretrained encoder weights for TPTrans
    pretrained_mtm = cfg.get("pretrained_mtm", None)
    if pretrained_mtm and isinstance(model, TPTrans):
        try:
            mtm_state = torch.load(pretrained_mtm, map_location="cpu")
            model_state = model.state_dict()
            loadable = {k: v for k, v in mtm_state.items() if k in model_state and v.shape == model_state[k].shape}
            if loadable:
                model_state.update(loadable)
                model.load_state_dict(model_state)
                print(f"[mtm] Loaded {len(loadable)} matching layers from {pretrained_mtm}:")
                for k in loadable.keys():
                    print(f"  ✓ {k}")
            else:
                print(f"[mtm] No matching layers found in {pretrained_mtm}; skipping preload")
                print(f"  MTM keys: {list(mtm_state.keys())[:5]}...")
                print(f"  Model keys: {list(model_state.keys())[:10]}...")
        except Exception as e:
            print(f"[mtm] Failed to load pretrained MTM weights from {pretrained_mtm}: {e}")

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 3e-4)))
    scaler_amp = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    epochs = int(cfg.get("epochs", 5))
    clip_norm = float(cfg.get("clip_norm", 1.0))
    delta = float(cfg.get("huber_delta", 1.0))

    best_val = float("inf")
    #best_path = out_dir / "traj_model.pt"  # will hold best if val exists, else last

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