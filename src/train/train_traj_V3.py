from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import load_config
from ..models import GRUSeq2Seq, TPTrans
from ..utils.datasets_V3 import make_ais_dataset
from ..models.TrAISformer import TrAISformer, BinSpec


def huber_loss(pred, target, delta: float = 1.0):
    return nn.SmoothL1Loss(beta=delta)(pred, target)



def main(cfg_path: str):
    cfg = load_config(cfg_path)
    pre_dir = cfg.get('processed_dir')  # e.g., "data/map_reduced/"
    out_dir = Path(cfg.get("out_dir", "data/checkpoints")); out_dir.mkdir(parents=True, exist_ok=True)

    window = int(cfg.get("window", 64))
    horizon = int(cfg.get("horizon", 12))

    # --- DATASETS (train/val) ---
    # If TrAISformer -> output_features=4 (lat,lon,sog,cog). Else 2 (lat,lon)
    out_feats = 4 if cfg["model"]["name"].lower() == "traisformer" else 2
    ds_train = make_ais_dataset(pre_dir + "train", window=window, horizon=horizon, output_features=out_feats, filter_short=True)
    ds_val   = make_ais_dataset(pre_dir + "val",   window=window, horizon=horizon, output_features=out_feats, filter_short=True)

    dl_train = DataLoader(ds_train, batch_size=int(cfg.get("batch_size", 128)), shuffle=True, num_workers=0, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=int(cfg.get("batch_size", 128)), shuffle=False, num_workers=0, pin_memory=True)

    feat_dim = 4  # we always pass [lat,lon,sog,cog] to encoders

    # --- MODEL ---
    name = cfg["model"]["name"].lower()
    if name == "gru":
        model = GRUSeq2Seq(feat_dim, d_model=cfg["model"].get("d_model", 128),
                           layers=cfg["model"].get("layers", 2), horizon=horizon)
        model_name = "gru"
    elif name == "tptrans":
        model = TPTrans(feat_dim=feat_dim,
                        d_model=cfg["model"].get("d_model", 192),
                        nhead=cfg["model"].get("nhead", 4),
                        enc_layers=cfg["model"].get("enc_layers", 4),
                        dec_layers=cfg["model"].get("dec_layers", 2),
                        horizon=horizon)
        model_name = "tptrans"
    elif name == "traisformer":
        bins_cfg = cfg["bins"]
        bins = BinSpec(
            lat_min=bins_cfg["lat_min"], lat_max=bins_cfg["lat_max"],
            lon_min=bins_cfg["lon_min"], lon_max=bins_cfg["lon_max"],
            sog_max=bins_cfg.get("sog_max", 50.0),
            n_lat=bins_cfg["n_lat"], n_lon=bins_cfg["n_lon"],
            n_sog=bins_cfg["n_sog"], n_cog=bins_cfg["n_cog"],
        )
        model = TrAISformer(
            bins=bins,
            d_model=cfg["model"].get("d_model", 512),
            nhead=cfg["model"].get("nhead", 8),
            num_layers=cfg["model"].get("enc_layers", 8),
            dropout=cfg["model"].get("dropout", 0.1),
            coarse_merge=cfg.get("coarse_merge", 3),
            coarse_beta=cfg.get("coarse_beta", 0.2),
        )
        model_name = "traisformer"
    else:
        raise ValueError(f"Unknown model {name}")

    ckpt_name = f"traj_{model_name}.pt"
    best_path = out_dir / ckpt_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 3e-4)))
    scaler_amp = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    epochs = int(cfg.get("epochs", 5)); clip_norm = float(cfg.get("clip_norm", 1.0))
    print(f"[train] model={model_name}, window={window}, horizon={horizon}, device={device}")

    def to_bins(X):  # X shape: [B, T, 4]
        lat, lon, sog, cog = X[...,0], X[...,1], X[...,2]*model.bins.sog_max, X[...,3]*360.0
        return {
            "lat": model.bins.lat_to_bin(lat),
            "lon": model.bins.lon_to_bin(lon),
            "sog": model.bins.sog_to_bin(sog),
            "cog": model.bins.cog_to_bin(cog),
        }

    best_val = float("inf")
    for epoch in range(1, epochs+1):
        model.train(); total = 0.0
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)  # xb:(B,window,4)  yb:(B,horizon,out_feats)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                if model_name == "traisformer":
                    past_idxs   = to_bins(xb)
                    # yb contains [lat,lon,sog,cog] in cols 0..3
                    future_idxs = to_bins(torch.cat([yb, torch.zeros_like(yb[..., :0])], dim=-1)[...,:4])  # quick split
                    logits = model(past_idxs, future_idxs)
                    loss = model.ce_loss_multi(logits, future_idxs)
                else:
                    pred = model(xb)               # (B,horizon,2)
                    loss = nn.SmoothL1Loss(beta=float(cfg.get("huber_delta", 1.0)))(pred, yb[..., :2])
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler_amp.step(opt); scaler_amp.update()
            total += float(loss.item()) * xb.size(0)
        train_loss = total / len(ds_train)

        # --- val ---
        model.eval(); vtotal = 0.0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                if model_name == "traisformer":
                    past_idxs   = to_bins(xb)
                    future_idxs = to_bins(torch.cat([yb, torch.zeros_like(yb[..., :0])], dim=-1)[...,:4])
                    logits = model(past_idxs, future_idxs)
                    vloss = model.ce_loss_multi(logits, future_idxs)
                else:
                    vloss = nn.SmoothL1Loss(beta=float(cfg.get("huber_delta", 1.0)))(model(xb), yb[..., :2])
                vtotal += float(vloss.item()) * xb.size(0)
        val_loss = vtotal / len(ds_val)
        print(f"epoch {epoch}: train={train_loss:.4f}  val={val_loss:.4f}")

        # save best
        if val_loss < best_val:
            best_val = val_loss
            if model_name == "traisformer":
                torch.save({"state_dict": model.state_dict(), "bins": model.bins.__dict__}, best_path)
            else:
                torch.save(model.state_dict(), best_path)

    print(f"Saved best to {best_path}")



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)

