from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os 
import time
import sys
from multiprocessing import cpu_count

# Adjust these imports if your folder structure is different
from ..config import load_config
from ..models.tptrans_V2 import TPTrans
from ..utils.datasets_V3 import make_ais_dataset
from ..utils.logging import CustomLogger

# --- CONFIGURATION FOR NORMALIZATION ---
# These MUST match the region you are training/evaluating on.
# Based on your eval command: --lat_min 54.0 --lat_max 58.0 --lon_min 6.0 --lon_max 16.0
NORM_CONFIG = {
    "LAT_MIN": 54.0,
    "LAT_MAX": 59.0,
    "LON_MIN": 5.0,
    "LON_MAX": 17.0,
    "SOG_MAX": 30.0,  # Max speed in knots
}


def normalize_batch(batch_x, cfg):
    """
    Normalizes a batch of AIS data [B, T, 4] to [0, 1].
    Channels: 0=Lat, 1=Lon, 2=SOG, 3=COG
    """
    norm_x = batch_x.clone()
    
    # 1. Lat: (val - min) / (max - min)
    norm_x[:, :, 0] = (batch_x[:, :, 0] - cfg["LAT_MIN"]) / (cfg["LAT_MAX"] - cfg["LAT_MIN"])
    
    # 2. Lon: (val - min) / (max - min)
    norm_x[:, :, 1] = (batch_x[:, :, 1] - cfg["LON_MIN"]) / (cfg["LON_MAX"] - cfg["LON_MIN"])
    
    # 3. SOG: val / max (Clamped to 1.0)
    norm_x[:, :, 2] = torch.clamp(batch_x[:, :, 2] / cfg["SOG_MAX"], 0.0, 1.0)
    
    # 4. COG: val / 360.0 (Cyclic 0-360)
    norm_x[:, :, 3] = batch_x[:, :, 3] / 360.0
    
    return norm_x

def main(cfg_path: str):
    cfg = load_config(cfg_path)
    pre_dir = cfg.get('processed_dir')
    out_dir = Path(cfg.get("out_dir", "data/checkpoints"))
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = CustomLogger(project_name="AIS-MDA", group=cfg.get("wandb_group", None), run_name=cfg.get("run_name", None))

    window = int(cfg.get("window", 64))
    horizon = int(cfg.get("horizon", 12))
    
    # Scale factor for Output Deltas (Physical Degree Difference * 100)
    # This keeps gradients healthy.
    SCALE_FACTOR = 100.0

    logger.log_config(cfg)
    logger.log_config({
        "window": window, 
        "horizon": horizon, 
        "mode": "delta_training_scaled",
        "scale_factor": SCALE_FACTOR,
        "norm_bounds": NORM_CONFIG
    })

    # --- DATASETS ---
    out_feats = 4 
    
    start_mode = cfg.get("start_mode", "head")
    kmeans_cfg = cfg.get("kmeans", None)
    epoch_samples = int(cfg.get("epoch_samples", 20))

    ds_train = make_ais_dataset(pre_dir + "train",
                                window=window, horizon=horizon,
                                output_features=out_feats,
                                filter_short=True,
                                start_mode=start_mode,
                                kmeans_config=kmeans_cfg,
                                epoch_samples=epoch_samples)
    
    ds_val = make_ais_dataset(pre_dir + "val",
                              window=window, horizon=horizon,
                              output_features=out_feats,
                              filter_short=True,
                              start_mode="uniform",
                              epoch_samples=max(2, epoch_samples // 4))

    # --- WINDOWS FIX ---
    if os.name == 'nt':
        print("[System] Detected Windows. Setting num_workers=0.")
        n_workers = 0
    else:
        n_workers = cpu_count()

    dl_train = DataLoader(ds_train, batch_size=int(cfg.get("batch_size", 128)), 
                          shuffle=True, num_workers=n_workers, pin_memory=True)
    dl_val   = DataLoader(ds_val, batch_size=int(cfg.get("batch_size", 128)), 
                          shuffle=False, num_workers=n_workers//2, pin_memory=True)

    # --- MODEL (TPTrans) ---
    name = cfg["model"]["name"].lower()
    if name != "tptrans":
        raise ValueError(f"This V2 script is specifically for TPTrans Delta training. Got {name}")

    feat_dim = 4
    model = TPTrans(feat_dim=feat_dim,
                    d_model=cfg["model"].get("d_model", 192),
                    nhead=cfg["model"].get("nhead", 4),
                    enc_layers=cfg["model"].get("enc_layers", 4),
                    dec_layers=cfg["model"].get("dec_layers", 2),
                    horizon=horizon)
    model_name = "tptrans"

    ckpt_name = f"traj_{model_name}_delta.pt"
    best_path = out_dir / ckpt_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    target_lr = float(cfg.get("lr", 3e-4))
    opt = torch.optim.AdamW(model.parameters(), lr=target_lr)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=max(2, int(cfg.get("early_stop_patience", 0))//4)
    )

    try:
        scaler_amp = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    except AttributeError:
        scaler_amp = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    criterion = nn.SmoothL1Loss(beta=float(cfg.get("huber_delta", 1.0)))

    epochs = int(cfg.get("epochs", 5))
    clip_norm = float(cfg.get("clip_norm", 1.0))
    
    print(f"[Train] Model={model_name} (Delta Mode), Device={device}, Scale={SCALE_FACTOR}")
    print(f"[Train] Normalization Bounds: {NORM_CONFIG}")
    
    warmup_epochs = int(cfg.get("warmup_epochs", max(1, int(epochs//40))))
    best_val = float("inf")
    
    final_train_loss = 0.0
    final_val_loss = 0.0

    for epoch in range(1, epochs+1):
        if epoch <= warmup_epochs:
            warmup_lr = target_lr * (epoch / warmup_epochs)
            for param_group in opt.param_groups:
                param_group['lr'] = warmup_lr

        model.train()
        total_loss = 0.0
        
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            
            # --- NORMALIZATION STEP ---
            # We normalize the INPUT (xb) so the Transformer sees [0, 1]
            xb_norm = normalize_batch(xb, NORM_CONFIG)
            # --------------------------

            opt.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                # 1. Predict Deltas using NORMALIZED input
                pred_deltas = model(xb_norm) # [B, H, 2]

                # 2. Compute Target Deltas using RAW PHYSICAL coordinates (Degrees)
                # We want the model to learn "Change in Degrees * 100"
                last_past = xb[:, -1, :2] 
                future_pos = yb[:, :, :2]
                
                full_seq = torch.cat([last_past.unsqueeze(1), future_pos], dim=1)
                
                # Calculate Raw Deltas (Degrees)
                raw_target_deltas = full_seq[:, 1:] - full_seq[:, :-1]
                
                # Scale up (0.001 deg -> 0.1 unitless target)
                target_deltas = raw_target_deltas * SCALE_FACTOR

                loss = criterion(pred_deltas, target_deltas)

            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler_amp.step(opt)
            scaler_amp.update()

            total_loss += loss.item() * xb.size(0)

        train_loss = total_loss / len(ds_train)
        final_train_loss = train_loss

        # Validation Loop
        model.eval()
        vtotal = 0.0
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                
                # --- NORMALIZATION STEP (Val) ---
                xb_norm = normalize_batch(xb, NORM_CONFIG)
                # --------------------------------
                
                pred_deltas = model(xb_norm)
                
                last_past = xb[:, -1, :2]
                future_pos = yb[:, :, :2]
                full_seq = torch.cat([last_past.unsqueeze(1), future_pos], dim=1)
                
                raw_target_deltas = full_seq[:, 1:] - full_seq[:, :-1]
                target_deltas = raw_target_deltas * SCALE_FACTOR

                vloss = criterion(pred_deltas, target_deltas)
                vtotal += vloss.item() * xb.size(0)
        
        val_loss = vtotal / len(ds_val)
        final_val_loss = val_loss
        
        current_lr = opt.param_groups[0]['lr']
        print(f"Epoch {epoch}: Train Loss={train_loss:.6f} | Val Loss={val_loss:.6f} | LR={current_lr:.2e}")
        
        if epoch > warmup_epochs:
            scheduler.step(val_loss)

        logger.log_metrics({
            "train_loss": train_loss, 
            "val_loss": val_loss, 
            "lr": current_lr
        }, step=epoch)

        if val_loss < best_val:
            best_val = val_loss
            state = {
                "state_dict": model.state_dict(),
                "scale_factor": SCALE_FACTOR,
                "model_type": "delta_tptrans",
                "norm_config": NORM_CONFIG # Save bounds so Eval knows!
            }
            torch.save(state, best_path)

    print(f"Saved best model to {best_path}")
    
    logger.log_summary({
        'best_val_loss': best_val,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'epochs_completed': epochs
    })
    
    logger.artifact(artifact=best_path, name=f"{model_name}_delta_model", type="model")
    logger.finish()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)