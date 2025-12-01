"""Evaluate MTM pretrained model quality and learned representations."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import os
import pickle
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.masking import make_time_mask, apply_time_mask


# Indices consistent with preprocessing
LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI = list(range(9))


class PickleMTMDataset(Dataset):
    """Same as training dataset."""
    def __init__(self, final_dir: str, window: int = 64, min_len: int = 64):
        self.window = int(window)
        self.paths: List[Path] = [Path(final_dir) / p for p in os.listdir(final_dir) if p.endswith("_processed.pkl")]
        self.index: List[tuple[int, int]] = []
        self._traj_cache: List[np.ndarray | None] = [None] * len(self.paths)

        for fi, p in enumerate(self.paths):
            with open(p, "rb") as f:
                item = pickle.load(f)
            traj = item["traj"]
            if traj.shape[0] < max(min_len, self.window):
                continue
            for s in range(0, traj.shape[0] - self.window + 1):
                self.index.append((fi, s))

    def __len__(self):
        return len(self.index)

    def _load_traj(self, fi: int) -> np.ndarray:
        if self._traj_cache[fi] is None:
            with open(self.paths[fi], "rb") as f:
                item = pickle.load(f)
            self._traj_cache[fi] = item["traj"]
        return self._traj_cache[fi]

    def __getitem__(self, idx: int):
        fi, s = self.index[idx]
        traj = self._load_traj(fi)
        sl = traj[s:s + self.window]
        x = sl[:, [LON, LAT, SOG, COG]].astype(np.float32)
        return torch.from_numpy(x)


class MTMEncoder(nn.Module):
    """Same architecture as training."""
    def __init__(self, in_dim: int = 4, d_model: int = 192, nhead: int = 4, enc_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)
        self.proj = nn.Linear(d_model, in_dim)

    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h = self.encoder(h)
        return self.proj(h)


def torch_haversine_m(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
    R = 6371000.0
    lat1 = torch.deg2rad(lat1_deg)
    lon1 = torch.deg2rad(lon1_deg)
    lat2 = torch.deg2rad(lat2_deg)
    lon2 = torch.deg2rad(lon2_deg)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.arcsin(torch.sqrt(torch.clamp(a, 0.0, 1.0)))
    return R * c


def compute_reconstruction_metrics(pred, true, mask_time, lat_min, lat_max, lon_min, lon_max, speed_max):
    """Compute reconstruction errors on masked positions."""
    lon_p, lat_p, sog_p, cog_p = pred[..., 0], pred[..., 1], pred[..., 2], pred[..., 3]
    lon_t, lat_t, sog_t, cog_t = true[..., 0], true[..., 1], true[..., 2], true[..., 3]

    # Denormalize
    lon_p_deg = lon_p * (lon_max - lon_min) + lon_min
    lon_t_deg = lon_t * (lon_max - lon_min) + lon_min
    lat_p_deg = lat_p * (lat_max - lat_min) + lat_min
    lat_t_deg = lat_t * (lat_max - lat_min) + lat_min
    sog_p_kn = sog_p * speed_max
    sog_t_kn = sog_t * speed_max
    cog_p_deg = cog_p * 360.0
    cog_t_deg = cog_t * 360.0

    m = mask_time  # [B,T]
    
    metrics = {}
    if m.any():
        # Spatial error (meters)
        d_m = torch_haversine_m(lat_p_deg[m], lon_p_deg[m], lat_t_deg[m], lon_t_deg[m])
        metrics['spatial_mae_m'] = d_m.mean().item()
        metrics['spatial_p50_m'] = d_m.median().item()
        metrics['spatial_p95_m'] = torch.quantile(d_m, 0.95).item()
        
        # SOG error (knots)
        sog_err = (sog_p_kn[m] - sog_t_kn[m]).abs()
        metrics['sog_mae_kn'] = sog_err.mean().item()
        
        # COG error (degrees, wrapped)
        delta_cog = (cog_p_deg[m] - cog_t_deg[m] + 180.0) % 360.0 - 180.0
        cog_err = delta_cog.abs()
        metrics['cog_mae_deg'] = cog_err.mean().item()
    else:
        metrics = {k: 0.0 for k in ['spatial_mae_m', 'spatial_p50_m', 'spatial_p95_m', 'sog_mae_kn', 'cog_mae_deg']}
    
    return metrics


def visualize_reconstruction(model, dataset, device, num_samples=3, mask_ratio=0.12, out_dir='data/figures'):
    """Visualize original vs. masked vs. reconstructed trajectories."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i in range(num_samples):
            idx = np.random.randint(len(dataset))
            x = dataset[idx].unsqueeze(0).to(device)  # [1,T,4]
            
            # Create mask
            mask_time = make_time_mask(1, x.size(1), mask_ratio=mask_ratio, device=device)
            x_masked = apply_time_mask(x, mask_time, mask_value=0.0)
            
            # Reconstruct
            pred = model(x_masked)
            
            # Convert to numpy
            x_np = x.cpu().numpy()[0]  # [T,4]
            x_masked_np = x_masked.cpu().numpy()[0]
            pred_np = pred.cpu().numpy()[0]
            mask_np = mask_time.cpu().numpy()[0]
            
            # Plot lon/lat trajectory
            axes[i, 0].plot(x_np[:, 0], x_np[:, 1], 'b-', label='Original', alpha=0.7)
            axes[i, 0].scatter(x_np[mask_np, 0], x_np[mask_np, 1], c='red', s=50, label='Masked', zorder=5)
            axes[i, 0].set_xlabel('Longitude (normalized)')
            axes[i, 0].set_ylabel('Latitude (normalized)')
            axes[i, 0].set_title(f'Sample {i+1}: Original Trajectory')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
            
            # Plot masked trajectory
            axes[i, 1].plot(x_masked_np[:, 0], x_masked_np[:, 1], 'g-', label='Masked input', alpha=0.7)
            axes[i, 1].scatter(x_masked_np[mask_np, 0], x_masked_np[mask_np, 1], c='red', s=50, label='Masked positions', zorder=5)
            axes[i, 1].set_xlabel('Longitude (normalized)')
            axes[i, 1].set_ylabel('Latitude (normalized)')
            axes[i, 1].set_title(f'Sample {i+1}: Masked Input')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)
            
            # Plot reconstruction
            axes[i, 2].plot(x_np[:, 0], x_np[:, 1], 'b-', label='Ground truth', alpha=0.5, linewidth=2)
            axes[i, 2].plot(pred_np[:, 0], pred_np[:, 1], 'r--', label='Reconstructed', alpha=0.7)
            axes[i, 2].scatter(x_np[mask_np, 0], x_np[mask_np, 1], c='blue', s=50, label='True masked', zorder=5)
            axes[i, 2].scatter(pred_np[mask_np, 0], pred_np[mask_np, 1], c='red', s=50, marker='x', label='Pred masked', zorder=5)
            axes[i, 2].set_xlabel('Longitude (normalized)')
            axes[i, 2].set_ylabel('Latitude (normalized)')
            axes[i, 2].set_title(f'Sample {i+1}: Reconstruction')
            axes[i, 2].legend()
            axes[i, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = out_dir / 'mtm_reconstruction_samples.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved reconstruction visualization to {save_path}")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Evaluate MTM pretrained encoder")
    ap.add_argument("--final_dir", default="data/map_reduce_final")
    ap.add_argument("--ckpt", default="data/checkpoints/traj_mtm.pt")
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--d_model", type=int, default=192)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--enc_layers", type=int, default=4)
    ap.add_argument("--mask_ratio", type=float, default=0.12)
    ap.add_argument("--num_samples", type=int, default=1000, help="Number of samples to evaluate")
    ap.add_argument("--vis_samples", type=int, default=3, help="Number of samples to visualize")
    ap.add_argument("--out_dir", default="data/figures")
    ap.add_argument("--metrics_dir", default="metrics")
    # Denorm constants
    ap.add_argument("--lat_min", type=float, default=54.0)
    ap.add_argument("--lat_max", type=float, default=58.0)
    ap.add_argument("--lon_min", type=float, default=6.0)
    ap.add_argument("--lon_max", type=float, default=16.0)
    ap.add_argument("--speed_max", type=float, default=30.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print(f"Loading MTM model from {args.ckpt}")
    model = MTMEncoder(in_dim=4, d_model=args.d_model, nhead=args.nhead, enc_layers=args.enc_layers).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    print("✓ Model loaded successfully")
    
    # Load dataset
    print(f"Loading dataset from {args.final_dir}")
    ds = PickleMTMDataset(args.final_dir, window=args.window, min_len=args.window)
    if len(ds) == 0:
        raise SystemExit(f"No windows found in {args.final_dir}")
    
    # Limit number of samples for faster evaluation
    eval_size = min(args.num_samples, len(ds))
    indices = np.random.choice(len(ds), eval_size, replace=False)
    eval_ds = torch.utils.data.Subset(ds, indices)
    dl = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"✓ Evaluating on {eval_size} samples")
    
    # Evaluate reconstruction quality
    print("\n" + "="*60)
    print("Evaluating reconstruction quality on masked positions...")
    print("="*60)
    
    all_metrics = []
    with torch.no_grad():
        for xb in tqdm(dl, desc="Evaluating"):
            xb = xb.to(device)
            mask_time = make_time_mask(xb.size(0), xb.size(1), mask_ratio=args.mask_ratio, device=device)
            x_masked = apply_time_mask(xb, mask_time, mask_value=0.0)
            
            pred = model(x_masked)
            
            metrics = compute_reconstruction_metrics(
                pred, xb, mask_time,
                args.lat_min, args.lat_max, args.lon_min, args.lon_max, args.speed_max
            )
            all_metrics.append(metrics)
    
    # Aggregate metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = float(np.mean(values))
        avg_metrics[key + "_std"] = float(np.std(values))
    
    # Print results
    print("\n" + "="*60)
    print("MTM RECONSTRUCTION METRICS (on masked positions)")
    print("="*60)
    print(f"Spatial MAE:        {avg_metrics['spatial_mae_m']:.2f} ± {avg_metrics['spatial_mae_m_std']:.2f} meters")
    print(f"Spatial P50:        {avg_metrics['spatial_p50_m']:.2f} ± {avg_metrics['spatial_p50_m_std']:.2f} meters")
    print(f"Spatial P95:        {avg_metrics['spatial_p95_m']:.2f} ± {avg_metrics['spatial_p95_m_std']:.2f} meters")
    print(f"SOG MAE:            {avg_metrics['sog_mae_kn']:.2f} ± {avg_metrics['sog_mae_kn_std']:.2f} knots")
    print(f"COG MAE:            {avg_metrics['cog_mae_deg']:.2f} ± {avg_metrics['cog_mae_deg_std']:.2f} degrees")
    print("="*60)
    
    # Save metrics
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "mtm_reconstruction.json"
    with open(metrics_path, 'w') as f:
        json.dump(avg_metrics, f, indent=2)
    print(f"\n✓ Saved metrics to {metrics_path}")
    
    # Visualize reconstructions
    print(f"\nGenerating {args.vis_samples} reconstruction visualizations...")
    visualize_reconstruction(model, ds, device, num_samples=args.vis_samples, 
                           mask_ratio=args.mask_ratio, out_dir=args.out_dir)
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
