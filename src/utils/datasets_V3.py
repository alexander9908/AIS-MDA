# src/utils/datasets.py
import os, pickle, numpy as np, torch
from torch.utils.data import Dataset
from sklearn.cluster import KMeans


def pipeline_adapter(window=64, horizon=12, output_features=2, filter_short=True):
    def decorator(cls):
        class AdaptedDataset(cls):
            def __init__(self, *args, **kwargs):
                required = window + horizon
                kwargs['max_seqlen'] = max(kwargs.get('max_seqlen', 96), required)
                kwargs.setdefault('min_required_len', required)
                super().__init__(*args, **kwargs)
                self._window = window
                self._horizon = horizon
                self._output_features = output_features
                self._valid_indices = []
                if filter_short:
                    for i in range(super().__len__()):
                        if super().original_length(i) >= required:
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
    def __init__(self, data_dir, max_seqlen=96, file_extension=".pkl",
                 min_required_len=None, start_mode="head", kmeans_config=None):
        self.data_dir = data_dir
        self.max_seqlen = max_seqlen
        self.min_required_len = min_required_len or max_seqlen
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith(file_extension)]
        self.file_list.sort()
        self.start_mode = (start_mode or "head").lower()
        self.kmeans_config = kmeans_config or {}
        self._lengths = []
        self._centroids = None
        self._kmeans_rng = np.random.default_rng(self.kmeans_config.get("random_state"))
        self._start_rng = np.random.default_rng(self.kmeans_config.get("random_state"))
        self._prepare_metadata()
    def __len__(self): return len(self.file_list)
    def _load_file(self, path):
        with open(path, 'rb') as f: return pickle.load(f)
    def original_length(self, idx):
        return self._lengths[idx]
    def _prepare_metadata(self):
        """Scan trajectories once for lengths and optional KMeans fitting."""
        coords_samples = []
        samples_per_traj = int(self.kmeans_config.get("samples_per_traj", 128))
        max_points = int(self.kmeans_config.get("max_points", 200000))
        for fname in self.file_list:
            path = os.path.join(self.data_dir, fname)
            V = self._load_file(path)
            traj = V["traj"]
            self._lengths.append(len(traj))
            if self.start_mode == "kmeans":
                lat_lon = traj[:, :2]
                if samples_per_traj > 0 and len(lat_lon) > samples_per_traj:
                    idx = self._kmeans_rng.choice(len(lat_lon), size=samples_per_traj, replace=False)
                    sample = lat_lon[idx]
                else:
                    sample = lat_lon
                coords_samples.append(sample.astype(np.float64, copy=False))
        if self.start_mode == "kmeans" and coords_samples:
            coords = np.concatenate(coords_samples, axis=0)
            if max_points > 0 and len(coords) > max_points:
                idx = self._kmeans_rng.choice(len(coords), size=max_points, replace=False)
                coords = coords[idx]
            n_clusters = int(self.kmeans_config.get("n_clusters", 32))
            n_init = int(self.kmeans_config.get("n_init", 10))
            random_state = self.kmeans_config.get("random_state", None)
            self._kmeans_model = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
            self._kmeans_model.fit(coords)
            self._centroids = self._kmeans_model.cluster_centers_
    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.file_list[idx])
        V = self._load_file(path)
        traj = V["traj"]  # np.ndarray
        start_idx = self._pick_start_index(traj)
        end_idx = min(len(traj), start_idx + self.max_seqlen)
        slice_traj = traj[start_idx:end_idx, :4]  # [lat,lon,sog,cog]
        slice_traj[slice_traj > 0.9999] = 0.9999
        seqlen = len(slice_traj)
        seq = np.zeros((self.max_seqlen, 4), dtype=np.float32)
        seq[:seqlen] = slice_traj
        mask = np.zeros((self.max_seqlen,), dtype=np.float32); mask[:seqlen] = 1.0
        mmsi = int(V["mmsi"]); time_start = int(traj[start_idx, 7])
        return (
            torch.tensor(seq),
            torch.tensor(mask),
            torch.tensor(seqlen),
            torch.tensor(mmsi),
            torch.tensor(time_start),
        )
    def _pick_start_index(self, traj):
        total_len = len(traj)
        required = min(self.min_required_len, total_len)
        max_start = max(0, total_len - required)
        if self.start_mode != "kmeans" or self._centroids is None or max_start == 0 or self._kmeans_model is None:
            return 0
        coords = traj[:, :2]
        # Restrict to clusters the voyage actually visits
        labels = self._kmeans_model.predict(coords)
        unique_labels = np.unique(labels)
        if len(unique_labels) == 0:
            return 0
        centroid_idx = int(self._start_rng.choice(unique_labels))
        target = self._centroids[centroid_idx]
        dists = np.linalg.norm(coords - target, axis=1)
        order = np.argsort(dists)
        for idx in order:
            if idx <= max_start:
                return int(idx)
        return int(max_start)

# Backward-compatible default (64/12 and 2 output feat = lat/lon)
AISDataset = pipeline_adapter(window=64, horizon=12, output_features=2)(AISDatasetBase)

# Factory so train/eval can pick window/horizon/output_features at runtime
def make_ais_dataset(data_dir, window, horizon, output_features=2,
                     filter_short=True, max_seqlen=None,
                     start_mode="head", kmeans_config=None):
    Adapted = pipeline_adapter(window=window, horizon=horizon,
                               output_features=output_features,
                               filter_short=filter_short)(AISDatasetBase)
    return Adapted(
        data_dir,
        max_seqlen=(max_seqlen or (window + horizon)),
        start_mode=start_mode,
        kmeans_config=kmeans_config,
    )
