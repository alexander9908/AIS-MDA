"""Trajectory segmentation by time gaps and minimal length."""
#from __future__ import annotations
#import pandas as pd
#import numpy as np
#
#def segment_trajectories(df: pd.DataFrame, gap_hours: float = 6.0, min_len: int = 10) -> pd.DataFrame:
#    df = df.copy().sort_values(["mmsi","timestamp"]).reset_index(drop=True)
#    df["dt"] = df.groupby("mmsi")["timestamp"].diff().dt.total_seconds().fillna(0.0)
#    gap = gap_hours * 3600.0
#    new_seg = (df["dt"] > gap).astype(int)
#    df["segment_id"] = df.groupby("mmsi")["dt"].apply(lambda s: new_seg.loc[s.index].cumsum())
#    # filter short segments
#    counts = df.groupby(["mmsi","segment_id"])["timestamp"].transform("size")
#    df = df[counts >= min_len].drop(columns=["dt"])
#    return df.reset_index(drop=True)


# src/dataio/segment.py
from __future__ import annotations
import pandas as pd
import numpy as np

def segment_trajectories(df: pd.DataFrame, gap_hours: float = 6.0, min_len: int = 10) -> pd.DataFrame:
    """
    Segment trajectories per MMSI when time gaps exceed `gap_hours`.
    Produces an integer `segment_id` starting at 0 within each MMSI.
    """
    df = df.copy()

    # Ensure proper types & ordering
    if "timestamp" not in df.columns:
        raise ValueError("Missing 'timestamp' column")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "mmsi"])

    # Sort and reset index
    df = df.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)

    # Δt in seconds per MMSI
    df["dt"] = (
        df.groupby("mmsi", sort=False)["timestamp"]
          .diff()
          .dt.total_seconds()
          .fillna(0.0)
          .clip(lower=0.0)
    )

    gap = float(gap_hours) * 3600.0

    # --- key change: use transform so the index aligns with df ---
    df["segment_id"] = df.groupby("mmsi", sort=False)["dt"].transform(lambda s: (s > gap).cumsum().astype(int))

    # Filter out short segments
    counts = df.groupby(["mmsi", "segment_id"], sort=False)["timestamp"].transform("size")
    df = df[counts >= int(min_len)]

    return df.reset_index(drop=True)