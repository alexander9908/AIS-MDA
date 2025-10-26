from __future__ import annotations
import pandas as pd
import numpy as np
from ..utils.geo import haversine_distance_m, project_to_local_xy

def add_kinematic_features(df: pd.DataFrame, use_projection: bool = True) -> pd.DataFrame:
    df = df.copy().sort_values(["mmsi","segment_id","timestamp"])
    # time delta
    df["dt"] = df.groupby(["mmsi","segment_id"])["timestamp"].diff().dt.total_seconds().fillna(0.0).clip(lower=0.0)
    if use_projection:
        x, y = project_to_local_xy(df["lat"].values, df["lon"].values)
        df["x"], df["y"] = x, y
        df["dx"] = df.groupby(["mmsi","segment_id"])["x"].diff().fillna(0.0)
        df["dy"] = df.groupby(["mmsi","segment_id"])["y"].diff().fillna(0.0)
    else:
        # fallback approximate meters using haversine step
        df["step_m"] = df.groupby(["mmsi","segment_id"]).apply(
            lambda g: pd.Series([0.0] + [haversine_distance_m(g["lat"].iloc[i-1], g["lon"].iloc[i-1], g["lat"].iloc[i], g["lon"].iloc[i]) for i in range(1,len(g))], index=g.index)
        ).reset_index(level=[0,1], drop=True)
    # derivatives
    for comp in ["dx","dy"]:
        if comp in df.columns:
            df[f"{comp}_dt"] = df[comp] / df["dt"].replace(0, np.nan)
    if "cog" in df.columns:
        rad = np.deg2rad(df["cog"].astype(float))
        df["cog_sin"], df["cog_cos"] = np.sin(rad), np.cos(rad)
    # acceleration approximations
    if "sog" in df.columns:
        sog = df["sog"].astype(float).values
        dt = df["dt"].replace(0, np.nan).values
        accel = np.zeros_like(sog, dtype=float)
        accel[1:] = np.diff(sog) / dt[1:]
        accel[np.isinf(accel) | np.isnan(accel)] = 0.0
        df["accel"] = accel
    return df
