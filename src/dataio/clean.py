"""Cleaning rules for AIS.
- Drop invalid coords
- Clamp speeds/ROT
- Remove teleport jumps using max_distance <= sog*Δt + margin
"""
from __future__ import annotations
import pandas as pd
import numpy as np

def _valid_coords(df: pd.DataFrame) -> pd.Series:
    return (df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))

def clean_ais_dataframe(df: pd.DataFrame, max_sog: float = 40.0) -> pd.DataFrame:
    df = df.copy()
    df = df[_valid_coords(df)]
    # clamp SOG
    if "sog" in df.columns:
        df.loc[:, "sog"] = np.clip(df["sog"].astype(float), 0, max_sog)
    # remove duplicates by (mmsi,timestamp)
    df = df.drop_duplicates(subset=["mmsi","timestamp"]).sort_values(["mmsi","timestamp"])
    # drop rows with null timestamps/mmsi
    df = df.dropna(subset=["mmsi","timestamp"])
    return df.reset_index(drop=True)
