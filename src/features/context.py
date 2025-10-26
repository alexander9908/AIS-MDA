from __future__ import annotations
import pandas as pd
import numpy as np

def add_directional_encodings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "cog" in df.columns:
        rad = np.deg2rad(df["cog"].astype(float))
        df["cog_sin"], df["cog_cos"] = np.sin(rad), np.cos(rad)
    return df

def add_grid_features(df: pd.DataFrame, cell_size_m: float = 1000.0) -> pd.DataFrame:
    """Simple square-grid index using projected x,y if present."""
    if not {"x","y"}.issubset(df.columns):
        return df
    df = df.copy()
    df["cell_x"] = (df["x"] // cell_size_m).astype(int)
    df["cell_y"] = (df["y"] // cell_size_m).astype(int)
    df["cell_id"] = (df["cell_x"].astype(str) + "_" + df["cell_y"].astype(str))
    return df
