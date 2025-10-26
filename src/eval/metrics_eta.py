from __future__ import annotations
import numpy as np

def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.abs(pred - true).mean())

def mape(pred: np.ndarray, true: np.ndarray, eps: float = 1e-6) -> float:
    return float((np.abs((pred - true) / (np.abs(true) + eps))).mean())

def p95(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.percentile(np.abs(pred - true), 95))
