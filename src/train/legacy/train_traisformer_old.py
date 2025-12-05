# src/train/train_traisformer.py
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
import os

from ...config import load_config
from ...models.traisformer import TrAISformer, BinSpec
from ...utils import make_ais_dataset
from ...utils.logging import CustomLogger

# --- CONFIG: MATCH YOUR DATA GENERATION ---
# This must match what you used in preprocessing.py!
DATA_BOUNDS = {
    "LAT_MIN": 54.0, "LAT_MAX": 59.0,
    "LON_MIN": 5.0,  "LON_MAX": 17.0,
    "SOG_MAX": 30.0,
}

def to_bins(batch_norm, bins: BinSpec):
    """
    Converts Normalized Batch [B, T, 4] -> Dictionary of Bin Indices
    Input channels: 0:Lat, 1:Lon, 2:SOG, 3:COG
    """
    # Denormalize first using the DATA_BOUNDS from preprocessing
    lat_deg = batch_norm[:, :, 0] * (DATA_BOUNDS["LAT_MAX"] - DATA_BOUNDS["LAT_MIN"]) + DATA_BOUNDS["LAT_MIN"]
    lon_deg = batch_norm[:, :, 1] * (DATA_BOUNDS["LON_MAX"] - DATA_BOUNDS["LON_MIN"]) + DATA_BOUNDS["LON_MIN"]
    sog_kn  = batch_norm[:, :, 2] * DATA_BOUNDS["SOG_MAX"]
    cog_deg = batch_norm[:, :, 3] * 360.0
    
    return {
        "lat": bins.lat_to_bin(lat_deg),
        "lon": bins.lon_to_bin(lon_deg),
        "sog": bins.sog_to_bin(sog_kn),
        "cog": bins.cog_to_bin(cog_deg)
    }

def main(cfg_path):
    cfg = load_config(cfg_path)
    out_dir = Path(cfg.get("out_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger = CustomLogger("AIS-MDA", group="traisformer", run_name=cfg.get("run_name"))
    
    # 1. Setup Bins
    bin_cfg = cfg.get("bins")
    bins = BinSpec(
        lat_min=DATA_BOUNDS["LAT_MIN"], lat_max=DATA_BOUNDS["LAT_MAX"],
        lon_min=DATA_BOUNDS["LON_MIN"], lon_max=DATA_BOUNDS["LON_MAX"],
        sog_max=DATA_BOUNDS["SOG_MAX"],
        n_lat=bin_cfg["n_lat"], n_lon=bin_cfg["n_lon"],
        n_sog=bin_cfg["n_sog"], n_cog=bin_cfg["n_cog"]
    )
    
    # 2. Model
    model = TrAISformer(
        bins=bins,
        d_model=cfg["model"]["d_model"],
        nhead=cfg["model"]["nhead"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
        emb_lat=cfg["model"]["emb_lat"],
        emb_lon=cfg["model"]["emb_lon"],
        emb_sog=cfg["model"]["emb_sog"],
        emb_cog=cfg["model"]["emb_cog"]
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # 3. Data
    pre_dir = cfg.get("processed_dir")
    ds_train = make_ais_dataset(pre_dir + "train", cfg["window"], cfg["horizon"], 4, epoch_samples=cfg.get("epoch_samples", 5))
    ds_val = make_ais_dataset(pre_dir + "val", cfg["window"], cfg["horizon"], 4, epoch_samples=2)
    
    dl_train = DataLoader(ds_train, batch_size=cfg["batch_size"], shuffle=True, num_workers=cpu_count(), pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=cfg["batch_size"], shuffle=False, num_workers=cpu_count(), pin_memory=True)
    
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]))
    
    best_loss = float('inf')
    
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        sum_loss = 0
        
        for x, y in dl_train:
            x, y = x.to(device), y.to(device)
            past_idxs = to_bins(x, bins)
            future_idxs = to_bins(y, bins)
            
            opt.zero_grad()
            logits = model(past_idxs, future_idxs)
            loss = model.compute_loss(logits, future_idxs)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sum_loss += loss.item()
            
        avg_train = sum_loss / len(dl_train)
        
        model.eval()
        sum_val = 0
        with torch.no_grad():
            for x, y in dl_val:
                x, y = x.to(device), y.to(device)
                past_idxs = to_bins(x, bins)
                future_idxs = to_bins(y, bins)
                logits = model(past_idxs, future_idxs)
                loss = model.compute_loss(logits, future_idxs)
                sum_val += loss.item()
                
        avg_val = sum_val / len(dl_val)
        print(f"Epoch {epoch} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")
        logger.log_metrics({"train_loss": avg_train, "val_loss": avg_val}, step=epoch)
        
        if avg_val < best_loss:
            best_loss = avg_val
            # Save everything needed for eval
            torch.save({
                "state_dict": model.state_dict(),
                "bins": bins.to_dict(),
                "config": cfg,
                "data_bounds": DATA_BOUNDS
            }, Path(cfg["out_dir"]) / "traj_traisformer.pt")
            print("  -> Saved Best Model")

    logger.finish()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    main(args.config)