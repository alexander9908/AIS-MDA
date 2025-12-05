# coding=utf-8
# Copyright 2021, Duong Nguyen
#
# Licensed under the CECILL-C License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.cecill.info
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Customized Pytorch Dataset.
"""

import numpy as np
import os
import pickle

import torch
from torch.utils.data import Dataset, DataLoader

import os
import pickle  # or use np.load, json.load, etc. depending on your file format

def pipeline_adapter(window=64, horizon=12, output_features=2, filter_short=True):
    """
    Decorator that adapts AISDataset to output pipeline-compatible format.
    
    Converts __getitem__ output from:
        (seq, mask, seqlen, mmsi, time_start)
    To:
        (X, Y) where X is input window and Y is prediction horizon
    
    Args:
        window: Input sequence length (default 64)
        horizon: Prediction horizon length (default 12)
        output_features: Number of features in target (2 for lat/lon)
        filter_short: If True, skip trajectories shorter than window+horizon
    
    Usage:
        @pipeline_adapter(window=64, horizon=12)
        class AISDataset(Dataset):
            ...
    """
    def decorator(cls):
        class AdaptedDataset(cls):
            def __init__(self, *args, **kwargs):
                # Override max_seqlen to ensure we load enough data
                kwargs['max_seqlen'] = max(
                    kwargs.get('max_seqlen', 96),
                    window + horizon
                )
                super().__init__(*args, **kwargs)
                
                self._window = window
                self._horizon = horizon
                self._output_features = output_features
                
                # Build index mapping to valid samples
                self._valid_indices = []
                if filter_short:
                    print("Filtering trajectories...")
                    for i in range(super().__len__()):
                        _, _, seqlen, _, _ = super().__getitem__(i)
                        if seqlen >= window + horizon:
                            self._valid_indices.append(i)
                    print(f"Valid samples: {len(self._valid_indices)}/{super().__len__()}")
                else:
                    self._valid_indices = list(range(super().__len__()))
            
            def __len__(self):
                return len(self._valid_indices)
            
            def __getitem__(self, idx):
                """
                Returns:
                    X: (window, 4) - input features [lat, lon, sog, cog]
                    Y: (horizon, output_features) - target features (typically lat, lon)
                """
                # Get original data from parent class
                orig_idx = self._valid_indices[idx]
                seq, mask, seqlen, mmsi, time_start = super().__getitem__(orig_idx)
                
                # Extract input window: first 'window' timesteps, all features
                X = seq[:self._window]  # (window, 4)
                
                # Extract target horizon: next 'horizon' timesteps, only specified features
                Y = seq[self._window:self._window + self._horizon, :self._output_features]
                
                return X, Y
            
            def get_original_item(self, idx):
                """Access original __getitem__ if needed."""
                orig_idx = self._valid_indices[idx]
                return super().__getitem__(orig_idx)
        
        AdaptedDataset.__name__ = f"Adapted{cls.__name__}"
        AdaptedDataset.__qualname__ = f"Adapted{cls.__qualname__}"
        return AdaptedDataset
    
    return decorator

@pipeline_adapter(window=64, horizon=12, output_features=2, filter_short=True)
class AISDataset(Dataset):
    """Customized Pytorch dataset that loads from disk.
    """
    def __init__(self, 
                 data_dir,
                 max_seqlen=96,
                 dtype=torch.float32,
                 device=torch.device("cpu"),
                 file_extension=".pkl"):
        """
        Args
            data_dir: path to directory containing data files
            max_seqlen: max sequence length
            file_extension: file extension to look for
        """    
            
        self.max_seqlen = max_seqlen
        self.device = device
        self.data_dir = data_dir
        
        # Build list of filenames
        self.file_list = [
            f for f in os.listdir(data_dir) 
            if f.endswith(file_extension)
        ]
        self.file_list.sort()

    def __len__(self):
        return len(self.file_list)
    
    def _load_file(self, filepath):
        """Load a single data file. Modify based on your file format."""
        # For pickle files:
        with open(filepath, 'rb') as f:
            V = pickle.load(f)
        
        return V
        
    def __getitem__(self, idx):
        """Gets items by loading from disk.
        
        Returns:
            seq: Tensor of (max_seqlen, [lat,lon,sog,cog]).
            mask: Tensor of (max_seqlen, 1). mask[i] = 0.0 if x[i] is padding.
            seqlen: sequence length.
            mmsi: vessel's MMSI.
            time_start: timestamp of the starting time of the trajectory.
        """
        # Load data from disk
        filepath = os.path.join(self.data_dir, self.file_list[idx])
        V = self._load_file(filepath)
        
        m_v = V["traj"][:,:4]  # lat, lon, sog, cog
        m_v[m_v>0.9999] = 0.9999
        seqlen = min(len(m_v), self.max_seqlen)
        seq = np.zeros((self.max_seqlen, 4))
        seq[:seqlen, :] = m_v[:seqlen, :]
        seq = torch.tensor(seq, dtype=torch.float32)
        
        mask = torch.zeros(self.max_seqlen)
        mask[:seqlen] = 1.
        
        seqlen = torch.tensor(seqlen, dtype=torch.int)
        mmsi = torch.tensor(int(V["mmsi"]), dtype=torch.int)
        time_start = torch.tensor(V["traj"][0, 4], dtype=torch.int)
        
        return seq, mask, seqlen, mmsi, time_start