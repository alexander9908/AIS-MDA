# src/utils/datasets.py
# coding=utf-8
import numpy as np
import os
import pickle
import torch
from torch.utils.data import Dataset

def pipeline_adapter(window=64, horizon=12, output_features=2, filter_short=True,
                     lat_col=0, lon_col=1,  # indices inside seq for lat/lon (your V["traj"] is [lat,lon,sog,cog,...])
                     y_order=("lon","lat")):
    """
    Adapts AISDataset to output (X, Y) for the trainer:
      - X: first `window` timesteps, all 4 features (lat,lon,sog,cog) as float32
      - Y: next `horizon` **deltas** in the order specified by y_order (default [Δlon, Δlat])

    Also adds:
      - get_original_item(idx): returns (seq, mask, seqlen, mmsi, time_start)
      - get_abs_segments(idx): returns (past_abs, future_abs) as absolute (lon,lat)
    """
    want_lon_first = (y_order == ("lon", "lat"))

    def decorator(cls):
        class AdaptedDataset(cls):
            def __init__(self, *args, **kwargs):
                kwargs['max_seqlen'] = max(kwargs.get('max_seqlen', 96), window + horizon)
                super().__init__(*args, **kwargs)

                self._window = window
                self._horizon = horizon
                self._lat_col = lat_col
                self._lon_col = lon_col
                self._want_lon_first = want_lon_first

                # Build index mapping to valid samples
                self._valid_indices = []
                if filter_short:
                    print("Filtering trajectories...")
                    for i in range(super().__len__()):
                        _, _, seqlen, _, _ = super().__getitem__(i)
                        if int(seqlen) >= window + horizon:
                            self._valid_indices.append(i)
                    print(f"Valid samples: {len(self._valid_indices)}/{super().__len__()}")
                else:
                    self._valid_indices = list(range(super().__len__()))

            def __len__(self):
                return len(self._valid_indices)

            def __getitem__(self, idx):
                """Return (X, Y_deltas) suitable for training:
                   X: (window, 4)  float32
                   Y: (horizon, 2) float32  -> [Δlon, Δlat] (default)
                """
                orig_idx = self._valid_indices[idx]
                seq, mask, seqlen, mmsi, time_start = super().__getitem__(orig_idx)
                # seq is (max_seqlen, 4) -> [lat, lon, sog, cog] in your current loader

                # Input window
                X = seq[:self._window, :]  # (T,4)

                # Absolute future positions from the sequence
                fut_abs_lat = seq[self._window:self._window + self._horizon, self._lat_col]  # (H,)
                fut_abs_lon = seq[self._window:self._window + self._horizon, self._lon_col]  # (H,)

                # Last past absolute point
                last_lat = seq[self._window - 1, self._lat_col]
                last_lon = seq[self._window - 1, self._lon_col]

                # Build absolute future as (lon,lat) for consistent delta order
                fut_abs = torch.stack([fut_abs_lon, fut_abs_lat], dim=1)  # (H,2)
                last_abs = torch.tensor([last_lon.item(), last_lat.item()], dtype=torch.float32).unsqueeze(0)  # (1,2)

                # Per-step deltas relative to previous absolute sample
                # Δ = future[t] - prev, with prev = last_abs for t=0, and then rolling
                fut_all = torch.cat([last_abs, fut_abs], dim=0)  # (H+1,2)
                Y = fut_all[1:] - fut_all[:-1]                   # (H,2) deltas

                # If the user preferred [Δlat,Δlon], swap here (we default to [Δlon,Δlat])
                if not self._want_lon_first:
                    Y = Y[:, [1, 0]]

                return X.float(), Y.float()

            def get_original_item(self, idx):
                """Access original tuple from base dataset."""
                orig_idx = self._valid_indices[idx]
                return super().__getitem__(orig_idx)

            def get_abs_segments(self, idx):
                """Return absolute past and future tracks as (lon,lat) numpy arrays for plotting."""
                seq, mask, seqlen, mmsi, time_start = self.get_original_item(idx)  # seq: (max_seqlen,4)
                T, H = self._window, self._horizon
                # Absolute past and future
                past_lon = seq[:T, self._lon_col].numpy()
                past_lat = seq[:T, self._lat_col].numpy()
                fut_lon = seq[T:T+H, self._lon_col].numpy()
                fut_lat = seq[T:T+H, self._lat_col].numpy()
                past_abs = np.stack([past_lon, past_lat], axis=1)  # (T,2)
                future_abs = np.stack([fut_lon, fut_lat], axis=1)  # (H,2)
                return past_abs, future_abs

        AdaptedDataset.__name__ = f"Adapted{cls.__name__}"
        AdaptedDataset.__qualname__ = f"Adapted{cls.__qualname__}"
        return AdaptedDataset

    return decorator


@pipeline_adapter(window=64, horizon=12, output_features=2, filter_short=True,
                  lat_col=0, lon_col=1, y_order=("lon","lat"))
class AISDataset(Dataset):
    """Loads individual *_processed.pkl and pads to max_seqlen.
       Expects V['traj'] to have columns: [lat, lon, sog, cog, ...].
    """
    def __init__(self, data_dir, max_seqlen=96, dtype=torch.float32,
                 device=torch.device("cpu"), file_extension=".pkl"):
        self.max_seqlen = max_seqlen
        self.device = device
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith(file_extension)]
        self.file_list.sort()

    def __len__(self):
        return len(self.file_list)

    def _load_file(self, filepath):
        with open(filepath, 'rb') as f:
            V = pickle.load(f)
        return V

    def __getitem__(self, idx):
        filepath = os.path.join(self.data_dir, self.file_list[idx])
        V = self._load_file(filepath)

        # Your preprocessor stored traj with at least [lat,lon,sog,cog,...]
        m_v = V["traj"][:, :4]              # (N,4)
        m_v[m_v > 0.9999] = 0.9999          # your previous clamp
        seqlen = min(len(m_v), self.max_seqlen)

        seq = np.zeros((self.max_seqlen, 4), dtype=np.float32)
        seq[:seqlen, :] = m_v[:seqlen, :]

        seq = torch.tensor(seq, dtype=torch.float32)
        mask = torch.zeros(self.max_seqlen, dtype=torch.float32)
        mask[:seqlen] = 1.0

        seqlen_t = torch.tensor(seqlen, dtype=torch.int32)
        mmsi = torch.tensor(int(V["mmsi"]), dtype=torch.int64)

        # if your 5th column is timestamp, keep this; else use 0
        time_start = torch.tensor(int(V["traj"][0, 4]) if V["traj"].shape[1] > 4 else 0, dtype=torch.int64)

        return seq, mask, seqlen_t, mmsi, time_start
