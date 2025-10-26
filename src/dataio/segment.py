"""Trajectory segmentation by time gaps and minimal length."""
from __future__ import annotations
import pandas as pd
import numpy as np

def segment_trajectories(df: pd.DataFrame, gap_hours: float = 6.0, min_len: int = 10) -> pd.DataFrame:
    df = df.copy().sort_values(["mmsi","timestamp"]).reset_index(drop=True)
    df["dt"] = df.groupby("mmsi")["timestamp"].diff().dt.total_seconds().fillna(0.0)
    gap = gap_hours * 3600.0
    new_seg = (df["dt"] > gap).astype(int)
    df["segment_id"] = df.groupby("mmsi")["dt"].apply(lambda s: new_seg.loc[s.index].cumsum())
    # filter short segments
    counts = df.groupby(["mmsi","segment_id"])["timestamp"].transform("size")
    df = df[counts >= min_len].drop(columns=["dt"])
    return df.reset_index(drop=True)
