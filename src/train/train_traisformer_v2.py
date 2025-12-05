import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
import os
import numpy as np

# Adjust imports based on your project structure
from ..config import load_config
from ..models.traisformer import TrAISformer, BinSpec
from ..utils.datasets import make_ais_dataset  # Using V5 for TrAISformer
from ..utils.logging import CustomLogger

# --- CONFIG: DATA BOUNDS ---
# These must match the bounds used in your evaluation script and preprocessing
# Normalized data [0,1] will be scaled back to these ranges before binning.
DATA_BOUNDS = {
    "LAT_MIN": 54.0, "LAT_MAX": 59.0,
    "LON_MIN": 5.0,  "LON_MAX": 17.0,
    "SOG_MAX": 30.0,
}

def to_bins(batch_norm, bins: BinSpec, device):
    """
    Converts Normalized Batch [B, T, 4] -> Dictionary of Bin Indices
    Input channels: 0:Lat, 1:Lon, 2:SOG, 3:COG
    """
    # 1. Denormalize [0, 1] -> Physical Units
    # We clone to avoid modifying the tensor in place if it's needed elsewhere
    lat_deg = batch_norm[:, :, 0] * (DATA_BOUNDS["LAT_MAX"] - DATA_BOUNDS["LAT_MIN"]) + DATA_BOUNDS["LAT_MIN"]
    lon_deg = batch_norm[:, :, 1] * (DATA_BOUNDS["LON_MAX"] - DATA_BOUNDS["LON_MIN"]) + DATA_BOUNDS["LON_MIN"]
    sog_kn  = batch_norm[:, :, 2] * DATA_BOUNDS["SOG_MAX"]
    cog_deg = batch_norm[:, :, 3] * 360.0
    
    # 2. Convert Physical Units -> Bin Indices
    # The bins class methods return tensors on the same device as input if coded correctly,
    # but we ensure they are on the correct device.
    return {
        "lat": bins.lat_to_bin(lat_deg).to(device),
        "lon": bins.lon_to_bin(lon_deg).to(device),
        "sog": bins.sog_to_bin(sog_kn).to(device),
        "cog": bins.cog_to_bin(cog_deg).to(device)
    }

def main(cfg_path):
    # 1. Load Configuration
    cfg = load_config(cfg_path)
    out_dir = Path(cfg.get("out_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Setup Logging
    logger = CustomLogger("AIS-MDA", group=cfg.get("wandb_group"), run_name=cfg.get("run_name"))
    logger.log_config(cfg)
    
    # 3. Setup Bins (The Vocabulary size)
    bin_cfg = cfg.get("bins")
    bins = BinSpec(
        lat_min=DATA_BOUNDS["LAT_MIN"], lat_max=DATA_BOUNDS["LAT_MAX"],
        lon_min=DATA_BOUNDS["LON_MIN"], lon_max=DATA_BOUNDS["LON_MAX"],
        sog_max=DATA_BOUNDS["SOG_MAX"],
        n_lat=bin_cfg["n_lat"], n_lon=bin_cfg["n_lon"],
        n_sog=bin_cfg["n_sog"], n_cog=bin_cfg["n_cog"]
    )
    
    # 4. Initialize Model
    model_cfg = cfg["model"]
    model = TrAISformer(
        bins=bins,
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg.get("num_layers", model_cfg.get("enc_layers", 6)), # Handle key variation
        dropout=model_cfg["dropout"],
        emb_lat=model_cfg.get("emb_lat", 128),
        emb_lon=model_cfg.get("emb_lon", 128),
        emb_sog=model_cfg.get("emb_sog", 64),
        emb_cog=model_cfg.get("emb_cog", 64)
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print("-" * 40)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"Device: {device}")
    print("-" * 40)

    # 5. Data Loading with KMeans Support
    pre_dir = cfg.get("processed_dir")
    
    # Extract KMeans settings from config
    start_mode = cfg.get("start_mode", "head") # defaults to 'head' if missing
    kmeans_cfg = cfg.get("kmeans", None)
    
    print(f"[Data] Initializing Train Dataset with mode='{start_mode}'...")
    ds_train = make_ais_dataset(
        data_dir=pre_dir + "train", 
        window=cfg["window"], 
        horizon=cfg["horizon"], 
        output_features=4, # Need all 4 features to bin them
        epoch_samples=cfg.get("epoch_samples", 5),
        start_mode=start_mode,
        kmeans_config=kmeans_cfg
    )
    
    print(f"[Data] Initializing Val Dataset (Uniform sampling)...")
    ds_val = make_ais_dataset(
        data_dir=pre_dir + "val", 
        window=cfg["window"], 
        horizon=cfg["horizon"], 
        output_features=4,
        epoch_samples=2,
        start_mode="uniform" # Usually standard sampling for validation
    )
    
    dl_train = DataLoader(ds_train, batch_size=cfg["batch_size"], shuffle=True, 
                          num_workers=cpu_count(), pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=cfg["batch_size"], shuffle=False, 
                        num_workers=cpu_count(), pin_memory=True)
    
    # 6. Optimization
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = cfg.get("early_stop_patience", 20)
    
    # 7. Training Loop
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        sum_loss = 0
        
        for i, (x, y) in enumerate(dl_train):
            x, y = x.to(device), y.to(device)
            
            # Convert normalized floats to bin indices
            past_idxs = to_bins(x, bins, device)
            future_idxs = to_bins(y, bins, device)
            
            opt.zero_grad()
            
            # Forward pass
            logits = model(past_idxs, future_idxs)
            loss = model.compute_loss(logits, future_idxs)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            sum_loss += loss.item()
            
        avg_train = sum_loss / len(dl_train)
        
        # Validation
        model.eval()
        sum_val = 0
        with torch.no_grad():
            for x, y in dl_val:
                x, y = x.to(device), y.to(device)
                past_idxs = to_bins(x, bins, device)
                future_idxs = to_bins(y, bins, device)
                
                logits = model(past_idxs, future_idxs)
                loss = model.compute_loss(logits, future_idxs)
                sum_val += loss.item()
                
        avg_val = sum_val / len(dl_val)
        
        # Logging & Scheduling
        scheduler.step(avg_val)
        current_lr = opt.param_groups[0]['lr']
        
        print(f"Epoch {epoch:03d} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | LR: {current_lr:.2e}")
        logger.log_metrics({
            "train_loss": avg_train, 
            "val_loss": avg_val,
            "lr": current_lr
        }, step=epoch)
        
        # Checkpointing
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            early_stop_counter = 0
            
            save_dict = {
                "state_dict": model.state_dict(),
                "bins": bins.to_dict(), # Save bins for eval!
                "config": cfg,
                "data_bounds": DATA_BOUNDS, # Save bounds for eval!
                "model_type": "traisformer"
            }
            torch.save(save_dict, out_dir / "traj_traisformer.pt")
            print("  -> Saved Best Model")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs.")
                break

    logger.finish()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    main(args.config)