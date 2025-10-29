# src/labeling/eta_labels.py
from __future__ import annotations
from typing import Iterable, Tuple, Dict
import numpy as np
import pandas as pd

_DEFAULT_FEATURES = [
    "sog", "cog_sin", "cog_cos", "accel", "dt", "dx", "dy",
    "dist_to_port", "bearing_to_port"
]

def _choose_features(df: pd.DataFrame, features: Iterable[str] | None):
    if features:
        F = [f for f in features if f in df.columns]
    else:
        F = [f for f in _DEFAULT_FEATURES if f in df.columns]
    if not F:
        raise ValueError("No usable features found for ETA features.")
    return F

def _clean_numeric(g: pd.DataFrame, F: list[str]) -> pd.DataFrame:
    """Coerce features to numeric, fill NaN reasonably, and drop invalids."""
    g = g.copy()
    for c in F:
        g[c] = pd.to_numeric(g[c], errors="coerce")

    # Simple local interpolation to fix short gaps
    g[F] = g[F].interpolate(method="linear", limit_direction="both", limit=3).ffill().bfill()
    return g

def make_eta_windows(
    df: pd.DataFrame,
    window: int,
    features: Iterable[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Make sliding windows for ETA regression.
    Each window has X [window, F] and scalar target y = time_to_port_sec (if available).
    """
    if "segment_id" not in df.columns:
        raise ValueError("segment_id missing. Did you run segment_trajectories first?")

    F = _choose_features(df, features)
    need_cols = set(F) | {"mmsi", "segment_id"}
    if "time_to_port_sec" in df.columns:
        need_cols.add("time_to_port_sec")
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    Xs, Ys, mmsi_list = [], [], []
    for (_, _), g0 in df.groupby(["mmsi", "segment_id"], sort=False):
        g0 = g0.sort_values("timestamp").reset_index(drop=True)
        g = _clean_numeric(g0, F)

        T = len(g)
        if T < window + 1:
            continue

        feat_arr = g[F].to_numpy(dtype="float32", copy=False)
        # Use real ETA label if available, else pseudo from horizon sum of dt
        if "time_to_port_sec" in g.columns and g["time_to_port_sec"].notna().any():
            y_all = g["time_to_port_sec"].to_numpy(dtype="float32", copy=False)
        else:
            # fallback: pseudo ETA by cumulative dt backwards
            dt = pd.to_numeric(g["dt"], errors="coerce").fillna(0).to_numpy(dtype="float32")
            y_all = np.flip(np.cumsum(np.flip(dt)))

        for i in range(T - window):
            x = feat_arr[i:i + window, :]
            y = y_all[i + window - 1]  # scalar target for last step in window
            # Skip windows with NaN
            if pd.isna(x).any() or pd.isna(y):
                continue
            Xs.append(x)
            Ys.append(y)
            mmsi_list.append(g["mmsi"].iloc[0])

    if Xs:
        X = np.stack(Xs)
        y = np.array(Ys, dtype="float32")
    else:
        X = np.empty((0, window, len(F)), dtype="float32")
        y = np.empty((0,), dtype="float32")

    meta = {"features": F, "window": window}
    print(f"[eta_labels] Built {len(X)} ETA windows using features {F}")
    return X, y, meta