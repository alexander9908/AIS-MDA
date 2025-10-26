from __future__ import annotations
import argparse, sys, numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.labeling.traj_labels import make_traj_windows

def build_traj(interim_path: Path, out_dir: Path, window: int, horizon: int, features: list[str] | None):
    df = pd.read_parquet(interim_path)
    X, Y, meta = make_traj_windows(df, window=window, horizon=horizon, features=features)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "Y.npy", Y)
    (out_dir / "meta.json").write_text(json_dumps(meta))
    print(f"[trajectory] Saved X.npy {X.shape}, Y.npy {Y.shape} to {out_dir}")

def build_eta(interim_path: Path, out_dir: Path, window: int, features: list[str] | None):
    df = pd.read_parquet(interim_path)
    # For demo: fabricate simple distance-to-destination and ETA if not present
    if "time_to_port_sec" not in df.columns:
        df["time_to_port_sec"] = np.nan  # Placeholder
        df = df.dropna(subset=["dx","dy","dt"])  # ensure features exist
    # Build sliding windows for features only; predict ETA at window end (placeholder random demo if missing)
    groups = list(df.groupby(["mmsi","segment_id"]))
    Xs, ys = [], []
    F = features or ["sog","cog_sin","cog_cos","accel","dt","dx","dy"]
    for (_, _), g in groups:
        g = g.reset_index(drop=True)
        for i in range(len(g) - window - 1):
            Xs.append(g.loc[i:i+window-1, F].values.astype("float32"))
            # Placeholder: if no real label, use next dt*cumulative steps as pseudo-ETA (demo only)
            if g["time_to_port_sec"].notna().any():
                y = g.loc[i+window, "time_to_port_sec"]
            else:
                y = float(g.loc[i+1:i+window+1, "dt"].sum())
            ys.append(y)
    X = np.stack(Xs, axis=0) if Xs else np.empty((0,window,len(F)), dtype="float32")
    y = np.array(ys, dtype="float32") if ys else np.empty((0,), dtype="float32")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y_eta.npy", y)
    (out_dir / "meta.json").write_text(json_dumps({"features":F,"window":window}))
    print(f"[eta] Saved X.npy {X.shape}, y_eta.npy {y.shape} to {out_dir}")

def build_anom(interim_path: Path, out_dir: Path, window: int, horizon: int, features: list[str] | None):
    df = pd.read_parquet(interim_path)
    X, Y, meta = make_traj_windows(df, window=window, horizon=horizon, features=features)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "Y.npy", Y)
    (out_dir / "meta.json").write_text(json_dumps(meta))
    print(f"[anomaly] Saved X.npy {X.shape}, Y.npy {Y.shape} to {out_dir}")

def json_dumps(obj):
    import json as _json
    return _json.dumps(obj, indent=2)

def main():
    import argparse, json
    ap = argparse.ArgumentParser(description="Build processed tensors (X.npy/Y.npy) from interim parquet")
    ap.add_argument("--interim", required=True, help="Path to interim parquet (from make_interim.py)")
    ap.add_argument("--task", required=True, choices=["trajectory","eta","anomaly"])
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
