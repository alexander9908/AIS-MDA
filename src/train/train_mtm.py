from __future__ import annotations
import argparse
from pathlib import Path
import os
import pickle
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ..utils.masking import make_time_mask, apply_time_mask


# Indices consistent with src/preprocessing/preprocessing.py
LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI = list(range(9))


class PickleMTMDataset(Dataset):
    """Streams map-reduce final pickles and yields fixed-length windows of [lon, lat, sog, cog].
    Assumes traj is already normalized as in preprocessing (0..1 for lat/lon/sog, cog/360).
    """
    def __init__(self, final_dir: str, window: int = 64, min_len: int = 64):
        self.window = int(window)
        self.paths: List[Path] = [Path(final_dir) / p for p in os.listdir(final_dir) if p.endswith("_processed.pkl")]
        self.index: List[tuple[int, int]] = []  # (file_idx, start_idx)
        self._traj_cache: List[np.ndarray | None] = [None] * len(self.paths)

        for fi, p in enumerate(self.paths):
            with open(p, "rb") as f:
                item = pickle.load(f)
            traj = item["traj"]  # shape [N, 9]
            if traj.shape[0] < max(min_len, self.window):
                continue
            # windows
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
        # channels: [lon, lat, sog, cog] = [1,0,2,3]
        x = sl[:, [LON, LAT, SOG, COG]].astype(np.float32)
        return torch.from_numpy(x)  # [T, 4]


class MTMEncoder(nn.Module):
    """Conv1d + Transformer encoder + linear head to reconstruct 4 channels per time step."""
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

    def forward(self, x):  # x: [B,T,4]
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)  # [B,T,D]
        h = self.encoder(h)  # [B,T,D]
        return self.proj(h)  # [B,T,4]


def torch_haversine_m(lat1_deg: torch.Tensor, lon1_deg: torch.Tensor, lat2_deg: torch.Tensor, lon2_deg: torch.Tensor) -> torch.Tensor:
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


def compute_mtm_loss(pred: torch.Tensor, true: torch.Tensor, mask_time: torch.Tensor,
                     lat_min: float, lat_max: float, lon_min: float, lon_max: float,
                     speed_max: float, lambda_spatial: float = 1.0, lambda_kin: float = 1.0) -> torch.Tensor:
    # pred/true in normalized space, channels [lon, lat, sog, cog]
    lon_p, lat_p, sog_p, cog_p = pred[..., 0], pred[..., 1], pred[..., 2], pred[..., 3]
    lon_t, lat_t, sog_t, cog_t = true[..., 0], true[..., 1], true[..., 2], true[..., 3]

    # denormalize
    lon_p_deg = lon_p * (lon_max - lon_min) + lon_min
    lon_t_deg = lon_t * (lon_max - lon_min) + lon_min
    lat_p_deg = lat_p * (lat_max - lat_min) + lat_min
    lat_t_deg = lat_t * (lat_max - lat_min) + lat_min

    sog_p_kn = sog_p * speed_max
    sog_t_kn = sog_t * speed_max

    cog_p_deg = cog_p * 360.0
    cog_t_deg = cog_t * 360.0

    m = mask_time  # [B,T]
    if m.any():
        d_m = torch_haversine_m(lat_p_deg[m], lon_p_deg[m], lat_t_deg[m], lon_t_deg[m])
        loss_spatial = d_m.mean()

        sog_err = (sog_p_kn[m] - sog_t_kn[m]).abs()
        delta_cog = (cog_p_deg[m] - cog_t_deg[m] + 180.0) % 360.0 - 180.0
        cog_err = delta_cog.abs()
        loss_kin = (sog_err + cog_err).mean()
        return lambda_spatial * loss_spatial + lambda_kin * loss_kin
    else:
        return torch.zeros((), device=pred.device)


def main():
    ap = argparse.ArgumentParser(description="Masked Trajectory Modeling pretraining from MapReduce pickles")
    ap.add_argument("--final_dir", default="data/map_reduce_final")
    ap.add_argument("--out_dir", default="data/checkpoints")
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=192)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--enc_layers", type=int, default=4)
    ap.add_argument("--mask_ratio", type=float, default=0.12)
    ap.add_argument("--span_len", type=int, default=1)
    ap.add_argument("--noise_std", type=float, default=0.0)
    ap.add_argument("--lambda_spatial", type=float, default=1.0)
    ap.add_argument("--lambda_kin", type=float, default=1.0)
    # geodesic/denorm constants from preprocessing
    ap.add_argument("--lat_min", type=float, default=54.0)
    ap.add_argument("--lat_max", type=float, default=58.0)
    ap.add_argument("--lon_min", type=float, default=6.0)
    ap.add_argument("--lon_max", type=float, default=16.0)
    ap.add_argument("--speed_max", type=float, default=30.0)
    args = ap.parse_args()

    ds = PickleMTMDataset(args.final_dir, window=args.window, min_len=args.window)
    if len(ds) == 0:
        raise SystemExit(f"No windows found in {args.final_dir} with window={args.window}")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                    num_workers=args.num_workers, pin_memory=True)
    print(f"[MTM] windows={len(ds)}  batches/epoch={len(dl)}  batch_size={args.batch_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MTMEncoder(in_dim=4, d_model=args.d_model, nhead=args.nhead, enc_layers=args.enc_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "traj_mtm.pt"

    model.train()
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        count = 0
        pbar = tqdm(dl, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for step, xb in enumerate(pbar, start=1):
            xb = xb.to(device)  # [B,T,4]
            mask_time = make_time_mask(xb.size(0), xb.size(1), mask_ratio=args.mask_ratio, span_len=args.span_len, device=device)
            x_masked = apply_time_mask(xb, mask_time, mask_value=0.0, noise_std=args.noise_std)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pred = model(x_masked)
                loss = compute_mtm_loss(pred, xb, mask_time,
                                        lat_min=args.lat_min, lat_max=args.lat_max,
                                        lon_min=args.lon_min, lon_max=args.lon_max,
                                        speed_max=args.speed_max,
                                        lambda_spatial=args.lambda_spatial, lambda_kin=args.lambda_kin)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()
            count += 1
            if step % max(1, args.log_every) == 0:
                pbar.set_postfix({"mtm_loss": f"{total / count:.4f}"})

        print(f"epoch {epoch}: mtm_loss={total / max(1,count):.4f}")

    torch.save(model.state_dict(), ckpt)
    print(f"Saved MTM checkpoint → {ckpt}")


if __name__ == "__main__":
    main()


