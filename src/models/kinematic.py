"""Classic kinematic baselines."""
from __future__ import annotations
import numpy as np

def constant_velocity_predict(history_dxdy: np.ndarray, horizon: int = 12) -> np.ndarray:
    """Given past deltas [T,2], predict next horizon steps by last observed velocity."""
    if len(history_dxdy) < 1:
        return np.zeros((horizon, 2), dtype="float32")
    v = history_dxdy[-1]  # last delta
    return np.tile(v, (horizon, 1)).astype("float32")
