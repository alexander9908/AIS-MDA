from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Dict

def make_traj_windows(df: pd.DataFrame, window: int = 64, horizon: int = 12, features=None
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Create sliding windows of features X and future deltas Y.
    Returns X [N, window, F], Y [N, horizon, 2] (dx,dy), and meta dict.
    """
    if features is None:
        features = ["dx","dy","sog","cog_sin","cog_cos","accel","dt"]
    # ensure required columns
    for col in ["dx","dy","dt"]:
        if col not in df.columns:
            raise ValueError(f"Required column missing: {col}")
    Xs, Ys, metas = [], [], []
    for (mmsi, seg), g in df.groupby(["mmsi","segment_id"], sort=False):
        g = g.reset_index(drop=True)
        # build windows
        for i in range(len(g) - (window + horizon)):
            x = g.loc[i:i+window-1, features].values.astype("float32")
            fut = g.loc[i+window:i+window+horizon-1, ["dx","dy"]].values.astype("float32")
            Xs.append(x); Ys.append(fut); metas.append((mmsi, seg, i))
    if not Xs:
        return np.empty((0,window,len(features)),dtype="float32"), np.empty((0,horizon,2),dtype="float32"), {"features":features}
    X = np.stack(Xs, axis=0); Y = np.stack(Ys, axis=0)
    return X, Y, {"features": features, "window": window, "horizon": horizon}
