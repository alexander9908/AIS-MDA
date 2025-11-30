# src/utils/datasets_V3.py
import os, pickle, numpy as np, torch
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from collections import defaultdict
import joblib
from pickle import UnpicklingError

class AISDatasetBase(Dataset):
    """
    Loads <MMSI>_<id>_processed.pkl.
    Supports 'epoch_samples' to oversample files within one epoch.
    Precomputes valid start indices and KMeans clusters to avoid runtime overhead.
    """
    def __init__(self, data_dir, max_seqlen=96, file_extension=".pkl",
                 min_required_len=None, start_mode="head", kmeans_config=None,
                 epoch_samples=1):
        self.data_dir = data_dir
        self.joblib_used = False 
        self.max_seqlen = max_seqlen
        self.min_required_len = min_required_len or max_seqlen
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith(file_extension)]
        self.file_list.sort()
        self.start_mode = (start_mode or "head").lower()
        self.kmeans_config = kmeans_config or {}
        self.epoch_samples = epoch_samples 
        
        self._lengths = []
        self._centroids = None
        self._kmeans_model = None
        self.file_meta = [] 
        
        # RNGs
        self._kmeans_rng = np.random.default_rng(self.kmeans_config.get("random_state"))
        self._start_rng = np.random.default_rng(self.kmeans_config.get("random_state"))
        
        self._prepare_metadata()

    def __len__(self): 
        return len(self.file_list) * self.epoch_samples

    def _load_file(self, path):
        if self.joblib_used:
            return joblib.load(path)
        else:
            try:
                with open(path, 'rb') as f: return pickle.load(f)
            except UnpicklingError:
                self.joblib_used = True
                return joblib.load(path)

    def original_length(self, idx):
        file_idx = idx // self.epoch_samples
        return self._lengths[file_idx]

    def _prepare_metadata(self):
        print(f"[AISDataset] Scanning {len(self.file_list)} files. Mode={self.start_mode}, Oversample={self.epoch_samples}x")
        
        coords_samples = []
        samples_per_traj_fit = int(self.kmeans_config.get("samples_per_traj", 128))
        max_points_fit = int(self.kmeans_config.get("max_points", 200000))
        
        temp_trajs = []
        
        for fname in self.file_list:
            path = os.path.join(self.data_dir, fname)
            V = self._load_file(path)
            traj = V["traj"]
            self._lengths.append(len(traj))
            temp_trajs.append(traj)

            if self.start_mode == "kmeans":
                lat_lon = traj[:, :2]
                if samples_per_traj_fit > 0 and len(lat_lon) > samples_per_traj_fit:
                    idx = self._kmeans_rng.choice(len(lat_lon), size=samples_per_traj_fit, replace=False)
                    sample = lat_lon[idx]
                else:
                    sample = lat_lon
                coords_samples.append(sample.astype(np.float64, copy=False))

        if self.start_mode == "kmeans" and coords_samples:
            coords = np.concatenate(coords_samples, axis=0)
            if max_points_fit > 0 and len(coords) > max_points_fit:
                idx = self._kmeans_rng.choice(len(coords), size=max_points_fit, replace=False)
                coords = coords[idx]
            
            n_clusters = int(self.kmeans_config.get("n_clusters", 32))
            n_init = int(self.kmeans_config.get("n_init", 10))
            random_state = self.kmeans_config.get("random_state", 42)
            
            print(f"[AISDataset] Fitting KMeans({n_clusters}) on {len(coords)} points...")
            self._kmeans_model = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
            self._kmeans_model.fit(coords)
            self._centroids = self._kmeans_model.cluster_centers_

        print("[AISDataset] Building start-index lookup tables...")
        for i, traj in enumerate(temp_trajs):
            total_len = len(traj)
            max_start = max(0, total_len - self.min_required_len)
            valid_starts = np.arange(max_start + 1, dtype=np.int32)
            
            meta = {'starts': valid_starts}
            
            if self.start_mode == "kmeans" and self._kmeans_model is not None and len(valid_starts) > 0:
                start_coords = traj[valid_starts, :2]
                labels = self._kmeans_model.predict(start_coords)
                cluster_map = defaultdict(list)
                for start_idx, label in zip(valid_starts, labels):
                    cluster_map[label].append(start_idx)
                meta['clusters'] = {k: np.array(v) for k, v in cluster_map.items()}
                meta['cluster_keys'] = np.array(list(meta['clusters'].keys()))
            
            self.file_meta.append(meta)
        
        del temp_trajs
        print("[AISDataset] Metadata ready.")

    def __getitem__(self, idx):
        file_idx = idx // self.epoch_samples
        path = os.path.join(self.data_dir, self.file_list[file_idx])
        V = self._load_file(path)
        traj = V["traj"]
        
        start_idx = self._pick_start_index_fast(file_idx)
        
        end_idx = min(len(traj), start_idx + self.max_seqlen)
        slice_traj = traj[start_idx:end_idx, :4]
        slice_traj[slice_traj > 0.9999] = 0.9999
        
        seqlen = len(slice_traj)
        seq = np.zeros((self.max_seqlen, 4), dtype=np.float32)
        seq[:seqlen] = slice_traj
        mask = np.zeros((self.max_seqlen,), dtype=np.float32); mask[:seqlen] = 1.0
        
        mmsi = int(V["mmsi"])
        time_start = int(traj[start_idx, 7])
        
        return (
            torch.tensor(seq),
            torch.tensor(mask),
            torch.tensor(seqlen),
            torch.tensor(mmsi),
            torch.tensor(time_start),
        )

    def _pick_start_index_fast(self, file_idx):
        meta = self.file_meta[file_idx]
        starts = meta['starts']
        
        if len(starts) == 0: return 0
            
        if self.start_mode == "kmeans" and 'clusters' in meta:
            keys = meta['cluster_keys']
            if len(keys) > 0:
                chosen_cluster = self._start_rng.choice(keys)
                candidates = meta['clusters'][chosen_cluster]
                return int(self._start_rng.choice(candidates))
        
        if self.start_mode == "head" and self.epoch_samples == 1:
             return 0 
        
        return int(self._start_rng.choice(starts))


# --- FIX: Explicit Inheritance instead of Decorator ---
class AISDataset(AISDatasetBase):
    """
    Standard implementation that slices Window + Horizon.
    """
    def __init__(self, data_dir, window=64, horizon=12, output_features=2, 
                 filter_short=True, max_seqlen=None, **kwargs):
        
        required = window + horizon
        # Pass parameters up to Base
        super().__init__(data_dir, 
                         max_seqlen=(max_seqlen or required), 
                         min_required_len=required, 
                         **kwargs)
        
        self._window = window
        self._horizon = horizon
        self._output_features = output_features
        self._valid_indices = []
        
        # Filter logic to handle short files
        base_len = super().__len__()
        if filter_short:
            for i in range(base_len):
                if super().original_length(i) >= required:
                    self._valid_indices.append(i)
        else:
            self._valid_indices = list(range(base_len))

    def __len__(self):
        return len(self._valid_indices)
    
    def __getitem__(self, idx):
        # 1. Map index to valid file index
        orig_idx = self._valid_indices[idx]
        
        # 2. Get full sequence from Base
        seq, mask, seqlen, mmsi, time_start = super().__getitem__(orig_idx)
        
        # 3. Slice X (Past) and Y (Future)
        # seq shape is [max_seqlen, 4]
        X = seq[:self._window]
        Y = seq[self._window : self._window + self._horizon, :self._output_features]
        
        return X, Y

def make_ais_dataset(data_dir, window, horizon, output_features=2,
                     filter_short=True, max_seqlen=None,
                     start_mode="head", kmeans_config=None, epoch_samples=20):
    """
    Factory function to create the AISDataset.
    """
    return AISDataset(
        data_dir=data_dir,
        window=window, 
        horizon=horizon,
        output_features=output_features,
        filter_short=filter_short,
        max_seqlen=max_seqlen,
        start_mode=start_mode,
        kmeans_config=kmeans_config,
        epoch_samples=epoch_samples,
    )