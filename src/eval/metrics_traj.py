from __future__ import annotations
import numpy as np

def ade(pred: np.ndarray, true: np.ndarray) -> float:
    """Average displacement error across horizon."""
    d = np.linalg.norm(pred - true, axis=-1)  # [B,H]
    return float(d.mean())

def fde(pred: np.ndarray, true: np.ndarray) -> float:
    d = np.linalg.norm(pred[:,-1,:] - true[:,-1,:], axis=-1)  # [B]
    return float(d.mean())
