from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores))

def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(average_precision_score(labels, scores))
