# src/train/train_traj_ES_V2.py
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

# Adjust imports to match your project structure
from ..config import load_config
from ..models import TPTrans
from ..utils.datasets_V3 import make_ais_dataset
from ..utils.logging import CustomLogger
from multiprocessing import cpu_count

def main(cfg_path: str):
    cfg = load_config(cfg_path)
    pre_dir = cfg.get('processed_dir')  # e.g., "data/map_reduced/"
    out_dir = Path(cfg.get("out_dir", "data/checkpoints"))
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = CustomLogger(project_name="AIS-MDA", group=cfg.get("wandb_group", None), run_name=cfg.get("run_name", None))

    window = int(cfg.get("window", 64))
    horizon = int(cfg.get("horizon", 12))

    logger.log_config(cfg)
    logger.log_config({"window": window, "horizon": horizon, "mode": "delta_training"})

    # --- DATASETS ---
    # TPTrans uses 4 features input [lat,lon,sog,cog], predicts 2 features [dlat, dlon]
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

    # --- WINDOWS MULTIPROCESSING FIX ---
    # Windows cannot pickle nested functions used in datasets_V3. 
    # Setting num_workers=0 runs on the main process (slower but works).
    if os.name == 'nt':
        print("[System] Detected Windows. Setting num_workers=0 to avoid pickling errors.")
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

    # Use modern GradScaler if available, else fallback
    try:
        scaler_amp = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    except AttributeError:
        scaler_amp = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # Delta Loss function
    criterion = nn.SmoothL1Loss(beta=float(cfg.get("huber_delta", 1.0)))

    epochs = int(cfg.get("epochs", 5))
    clip_norm = float(cfg.get("clip_norm", 1.0))
    
    print(f"[Train] Model={model_name} (Delta Mode), Device={device}")
    
    warmup_epochs = int(cfg.get("warmup_epochs", max(1, int(epochs//40))))
    best_val = float("inf")
    
    # Store these for final summary logging
    final_train_loss = 0.0
    final_val_loss = 0.0

    for epoch in range(1, epochs+1):
        # Linear Warmup
        if epoch <= warmup_epochs:
            warmup_lr = target_lr * (epoch / warmup_epochs)
            for param_group in opt.param_groups:
                param_group['lr'] = warmup_lr

        model.train()
        total_loss = 0.0
        
        # Train Loop
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device) # xb:[B, W, 4], yb:[B, H, 4]
            
            opt.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                # 1. Predict Deltas
                pred_deltas = model(xb) # [B, H, 2]

                # 2. Compute Target Deltas (y_t - y_{t-1})
                # We need the last past point to calculate the first delta
                last_past = xb[:, -1, :2] # [B, 2]
                future_pos = yb[:, :, :2] # [B, H, 2]
                
                # Concatenate [last_past, future_pos] -> [B, H+1, 2]
                full_seq = torch.cat([last_past.unsqueeze(1), future_pos], dim=1)
                
                # Target is diff between adjacent steps
                target_deltas = full_seq[:, 1:] - full_seq[:, :-1] # [B, H, 2]

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
                
                pred_deltas = model(xb)
                
                last_past = xb[:, -1, :2]
                future_pos = yb[:, :, :2]
                full_seq = torch.cat([last_past.unsqueeze(1), future_pos], dim=1)
                target_deltas = full_seq[:, 1:] - full_seq[:, :-1]

                vloss = criterion(pred_deltas, target_deltas)
                vtotal += vloss.item() * xb.size(0)
        
        val_loss = vtotal / len(ds_val)
        final_val_loss = val_loss
        
        current_lr = opt.param_groups[0]['lr']
        print(f"Epoch {epoch}: Train Loss={train_loss:.6f} | Val Loss={val_loss:.6f} | LR={current_lr:.2e}")
        
        if epoch > warmup_epochs:
            scheduler.step(val_loss)

        # Log metrics to WandB History (Curves)
        logger.log_metrics({
            "train_loss": train_loss, 
            "val_loss": val_loss, 
            "lr": current_lr
        }, step=epoch)

        # Check improvement
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)

    print(f"Saved best model to {best_path}")
    
    # --- EXPLICIT SUMMARY LOGGING ---
    # This ensures these values appear in the "Summary" columns of the WandB table
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