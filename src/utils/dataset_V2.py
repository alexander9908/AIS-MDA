import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class AISDatasetV2(Dataset):
    """
    Full-trip AIS dataset loader for evaluation & rollout.
    Returns complete trajectories without truncation.

    Expects files to be pickles with keys:
      - 'traj': ndarray with columns [lat, lon, sog, cog, heading, rot, nav, timestamp, mmsi]
      - 'mmsi': vessel MMSI (int)
    """

    def __init__(self, data_dir: str, file_extension: str = ".pkl", dtype: torch.dtype = torch.float32):
        self.data_dir = data_dir
        self.file_list = sorted([f for f in os.listdir(data_dir) if f.endswith(file_extension)])
        self.dtype = dtype

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.file_list[idx])
        with open(path, "rb") as f:
            data = pickle.load(f)
        traj = np.asarray(data["traj"], dtype=np.float32)
        mmsi = int(data["mmsi"]) if isinstance(data, dict) and "mmsi" in data else int(traj[0, 8])
        traj_tensor = torch.tensor(traj, dtype=self.dtype)
        return traj_tensor, mmsi, self.file_list[idx]

