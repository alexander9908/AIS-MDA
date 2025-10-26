from __future__ import annotations
import pandas as pd
import numpy as np

def inject_synthetic_anomalies(df: pd.DataFrame, prob: float = 0.01, spike_scale: float = 5.0) -> pd.DataFrame:
    """Simple synthetic anomalies by injecting dx/dy spikes."""
    df = df.copy()
    mask = (np.random.rand(len(df)) < prob)
    if {"dx","dy"}.issubset(df.columns):
        df.loc[mask, "dx"] = df.loc[mask, "dx"] * spike_scale
        df.loc[mask, "dy"] = df.loc[mask, "dy"] * spike_scale
    return df
