from __future__ import annotations
from typing import Iterable, Tuple, Dict, List
import numpy as np
import pandas as pd

_DEFAULT_FEATURES = ["dx", "dy", "sog", "cog_sin", "cog_cos", "accel", "dt", "cell_id"]

def _choose_features(df: pd.DataFrame, features: Iterable[str] | None) -> List[str]:
    if features:
        F = [f for f in features if f in df.columns]
    else:
        F = [f for f in _DEFAULT_FEATURES if f in df.columns]
    if not F:
        raise ValueError("No usable features found in dataframe.")
    return F

def _unique_preserve_order(cols: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

def _prep_types_and_fill(df: pd.DataFrame, cols_for_fill: List[str]) -> pd.DataFrame:
    g = df.copy()
    cols_for_fill = _unique_preserve_order(cols_for_fill)

    # Handle cell_id (categorical-like)
    if "cell_id" in g.columns and "cell_id" in cols_for_fill:
        g["cell_id"] = g["cell_id"].fillna(-1).astype("int32", errors="ignore")

    # Try to coerce all other columns to numeric if possible
    for c in cols_for_fill:
        if c == "cell_id" or c not in g.columns:
            continue
        g[c] = pd.to_numeric(g[c], errors="coerce")

    # Numeric interpolation for remaining numeric columns
    num_cols = [c for c in cols_for_fill if c != "cell_id" and pd.api.types.is_numeric_dtype(g[c])]
    if num_cols:
        g[num_cols] = (
            g[num_cols]
            .interpolate(method="linear", limit_direction="both", limit=3)
            .ffill()
            .bfill()
        )
    return g

def make_traj_windows(
    df: pd.DataFrame,
    window: int,
    horizon: int,
    features: Iterable[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    if "segment_id" not in df.columns:
        raise ValueError("segment_id missing. Did you run segment_trajectories first?")

    F = _choose_features(df, features)
    need_cols = set(F) | {"mmsi", "segment_id", "dx", "dy"}
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Filter out non-numeric feature columns (warn once)
    num_feats = []
    for f in F:
        if pd.api.types.is_numeric_dtype(df[f]):
            num_feats.append(f)
        else:
            print(f"[warn] Dropping non-numeric feature '{f}' (dtype={df[f].dtype})")

    F = num_feats
    if not F:
        raise ValueError("No numeric features available for trajectory modeling.")

    Xs, Ys = [], []
    cols_for_fill = _unique_preserve_order(F + ["dx", "dy"])

    for (_, _), g0 in df.groupby(["mmsi", "segment_id"], sort=False):
        g0 = g0.sort_values("timestamp").reset_index(drop=True)
        g = _prep_types_and_fill(g0, cols_for_fill)

        T = len(g)
        max_i = T - (window + horizon)
        if max_i < 0:
            continue

        feat_arr = g[F].to_numpy(dtype="float32", copy=False)
        target_arr = g[["dx", "dy"]].to_numpy(dtype="float32", copy=False)

        for i in range(max_i + 1):
            x = feat_arr[i:i + window]
            y = target_arr[i + window:i + window + horizon]
            # pandas.isna works even if array is float32
            if pd.isna(x).any() or pd.isna(y).any():
                continue
            Xs.append(x)
            Ys.append(y)

    if Xs:
        X = np.stack(Xs)
        Y = np.stack(Ys)
    else:
        X = np.empty((0, window, len(F)), dtype="float32")
        Y = np.empty((0, horizon, 2), dtype="float32")

    meta = {"features": F, "window": window, "horizon": horizon}
    print(f"[traj_labels] Built {len(X)} windows using features {F}")
    return X, Y, meta