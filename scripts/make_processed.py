from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure package import works when running from scripts/
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.labeling.traj_labels import make_traj_windows
from src.labeling.eta_labels import make_eta_windows  # <-- add this import


def _save_scaler(X: np.ndarray, out_dir: Path):
    flat = X.reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0).astype("float32")
    std = flat.std(axis=0).astype("float32") + 1e-8
    np.savez(out_dir / "scaler.npz", mean=mean, std=std)


def _save_window_mmsi(df: pd.DataFrame, window: int, horizon: int, out_dir: Path):
    mmsis: list[int] = []
    for (mmsi, seg), g in df.groupby(["mmsi", "segment_id"], sort=False):
        g = g.reset_index(drop=True)
        for _i in range(len(g) - (window + horizon)):
            mmsis.append(int(mmsi))
    np.save(out_dir / "window_mmsi.npy", np.array(mmsis, dtype="int64"))


def build_traj(interim_path: Path, out_dir: Path, window: int, horizon: int, features: list[str] | None):
    df = pd.read_parquet(interim_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, Y, meta = make_traj_windows(df, window=window, horizon=horizon, features=features)

    if X.size > 0:
        _save_scaler(X, out_dir)
        _save_window_mmsi(df, window=window, horizon=horizon, out_dir=out_dir)

    np.save(out_dir / "X.npy", X.astype("float32", copy=False))
    np.save(out_dir / "Y.npy", Y.astype("float32", copy=False))

    (out_dir / "meta.json").write_text(json_dumps({"features": meta.get("features", features),
                                                   "window": window, "horizon": horizon}))
    print(f"[trajectory] Saved X.npy {X.shape}, Y.npy {Y.shape} to {out_dir}")


def build_eta(interim_path: Path, out_dir: Path, window: int, features: list[str] | None):
    """Use the robust ETA window builder (handles NaNs and dtypes)."""
    df = pd.read_parquet(interim_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, meta = make_eta_windows(df, window=window, features=features)

    if X.size > 0:
        _save_scaler(X, out_dir)

    np.save(out_dir / "X.npy", X.astype("float32", copy=False))
    np.save(out_dir / "y_eta.npy", y.astype("float32", copy=False))
    (out_dir / "meta.json").write_text(json_dumps({"features": meta.get("features", features),
                                                   "window": window}))
    print(f"[eta] Saved X.npy {X.shape}, y_eta.npy {y.shape} to {out_dir}")


def build_anom(interim_path: Path, out_dir: Path, window: int, horizon: int, features: list[str] | None):
    df = pd.read_parquet(interim_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, Y, meta = make_traj_windows(df, window=window, horizon=horizon, features=features)

    if X.size > 0:
        _save_scaler(X, out_dir)
        _save_window_mmsi(df, window=window, horizon=horizon, out_dir=out_dir)

    np.save(out_dir / "X.npy", X.astype("float32", copy=False))
    np.save(out_dir / "Y.npy", Y.astype("float32", copy=False))
    (out_dir / "meta.json").write_text(json_dumps({"features": meta.get("features", features),
                                                   "window": window, "horizon": horizon}))
    print(f"[anomaly] Saved X.npy {X.shape}, Y.npy {Y.shape} to {out_dir}")


def json_dumps(obj):
    import json as _json
    return _json.dumps(obj, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Build processed tensors (X.npy/Y.npy) from interim parquet")
    ap.add_argument("--interim", required=True, help="Path to interim parquet (from make_interim.py)")
    ap.add_argument("--task", required=True, choices=["trajectory", "eta", "anomaly"])
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--features", nargs="*", default=None)
    ap.add_argument("--out", required=True, help="Output directory for processed tensors")
    args = ap.parse_args()

    interim_path = Path(args.interim)
    out_dir = Path(args.out)

    if args.task == "trajectory":
        build_traj(interim_path, out_dir, args.window, args.horizon, args.features)
    elif args.task == "eta":
        build_eta(interim_path, out_dir, args.window, args.features)
    else:
        build_anom(interim_path, out_dir, args.window, args.horizon, args.features)


if __name__ == "__main__":
    main()