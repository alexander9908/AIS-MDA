# src/utils/datasets.py
import os, pickle, numpy as np, torch
from torch.utils.data import Dataset

def pipeline_adapter(window=64, horizon=12, output_features=2, filter_short=True):
    def decorator(cls):
        class AdaptedDataset(cls):
            def __init__(self, *args, **kwargs):
                kwargs['max_seqlen'] = max(kwargs.get('max_seqlen', 96), window + horizon)
                super().__init__(*args, **kwargs)
                self._window = window
                self._horizon = horizon
                self._output_features = output_features
                self._valid_indices = []
                if filter_short:
                    for i in range(super().__len__()):
                        _, _, seqlen, _, _ = super().__getitem__(i)
                        if seqlen >= window + horizon:
                            self._valid_indices.append(i)
                else:
                    self._valid_indices = list(range(super().__len__()))
            def __len__(self): return len(self._valid_indices)
            def __getitem__(self, idx):
                orig_idx = self._valid_indices[idx]
                seq, mask, seqlen, mmsi, time_start = super().__getitem__(orig_idx)
                X = seq[:self._window]                                 # (window, 4)  [lat,lon,sog,cog]
                Y = seq[self._window:self._window+self._horizon, :self._output_features]
                return X, Y
            def get_original_item(self, idx):
                orig_idx = self._valid_indices[idx]
                return super().__getitem__(orig_idx)
        AdaptedDataset.__name__ = f"Adapted{cls.__name__}"
        return AdaptedDataset
    return decorator

class AISDatasetBase(Dataset):
    """Loads <MMSI>_<id>_processed.pkl with dict['traj'] columns:
       [LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI] (normalized lat/lon/sog/cog ok)
       Returns (seq, mask, seqlen, mmsi, time_start) where seq = (max_seqlen, 4) = [lat,lon,sog,cog].
    """
    def __init__(self, data_dir, max_seqlen=96, file_extension=".pkl"):
        self.data_dir = data_dir
        self.max_seqlen = max_seqlen
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith(file_extension)]
        self.file_list.sort()
    def __len__(self): return len(self.file_list)
    def _load_file(self, path):
        with open(path, 'rb') as f: return pickle.load(f)
    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.file_list[idx])
        V = self._load_file(path)
        traj = V["traj"]  # np.ndarray
        m_v = traj[:, :4] # [lat,lon,sog,cog]
        m_v[m_v > 0.9999] = 0.9999
        seqlen = min(len(m_v), self.max_seqlen)
        seq = np.zeros((self.max_seqlen, 4), dtype=np.float32)
        seq[:seqlen] = m_v[:seqlen]
        mask = np.zeros((self.max_seqlen,), dtype=np.float32); mask[:seqlen] = 1.0
        mmsi = int(V["mmsi"]); time_start = int(traj[0, 7])
        return torch.tensor(seq), torch.tensor(mask), torch.tensor(seqlen), torch.tensor(mmsi), torch.tensor(time_start)

# Backward-compatible default (64/12 and 2 output feat = lat/lon)
AISDataset = pipeline_adapter(window=64, horizon=12, output_features=2)(AISDatasetBase)

# Factory so train/eval can pick window/horizon/output_features at runtime
def make_ais_dataset(data_dir, window, horizon, output_features=2, filter_short=True, max_seqlen=None):
    Adapted = pipeline_adapter(window=window, horizon=horizon, output_features=output_features, filter_short=filter_short)(AISDatasetBase)
    return Adapted(data_dir, max_seqlen=(max_seqlen or (window + horizon)))
