from __future__ import annotations
import argparse, sys, os
from pathlib import Path

# allow "src" imports when running from scripts/
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.dataio.load_ais import load_ais_df
from src.dataio.clean import clean_ais_dataframe
from src.dataio.segment import segment_trajectories
from src.features.kinematics import add_kinematic_features
from src.features.context import add_directional_encodings, add_grid_features


def _pipeline_one(group_df: pd.DataFrame, gap_hours: float, max_sog: float) -> pd.DataFrame:
    """Run the full pipeline on a single MMSI group."""
    if group_df.empty:
        return group_df
    g = clean_ais_dataframe(group_df, max_sog=max_sog)
    g = segment_trajectories(g, gap_hours=gap_hours, min_len=10)
    if g.empty:
        return g
    g = add_kinematic_features(g, use_projection=True)
    g = add_directional_encodings(g)
    g = add_grid_features(g, cell_size_m=1000.0)
    return g


def main():
    ap = argparse.ArgumentParser(description="Clean and segment AIS into interim parquet (parallel)")
    ap.add_argument("--raw", nargs="+", required=True, help="Paths to raw CSV/Parquet files")
    ap.add_argument("--out", required=True, help="Output interim directory")
    ap.add_argument("--gap_hours", type=float, default=6.0)
    ap.add_argument("--max_sog", type=float, default=40.0)
    ap.add_argument("--workers", type=int, default=0, help="Parallel workers (0=auto; 1=serial)")
    ap.add_argument("--engine", default="pyarrow", choices=["pyarrow", "fastparquet"], help="Parquet engine")
    args = ap.parse_args()

    # Decide worker count
    if args.workers <= 0:
        try:
            import multiprocessing as mp
            workers = max(1, mp.cpu_count() - 1)
        except Exception:
            workers = 1
    else:
        workers = args.workers

    print(f"[make_interim] Loading raw files ({len(args.raw)}) ...")
    df = load_ais_df(args.raw)  # already normalizes/renames and sorts by mmsi,timestamp
    if df.empty:
        raise SystemExit("No data after load.")

    # Split into MMSI groups
    if "mmsi" not in df.columns:
        raise SystemExit("mmsi column missing after load.")
    groups = list(df.groupby("mmsi", sort=False))
    del df  # free memory

    print(f"[make_interim] Processing {len(groups)} vessels with workers={workers} ...")

    # Try pandarallel first (nice API + progress bar)
    use_pandarallel = False
    if workers > 1:
        try:
            from pandarallel import pandarallel
            pandarallel.initialize(nb_workers=workers, progress_bar=False)
            use_pandarallel = True
        except Exception:
            use_pandarallel = False

    if use_pandarallel:
        # Rebuild a df with an MMSI index for parallel_apply
        big = pd.concat([g for _, g in groups], axis=0, ignore_index=True)
        # Ensure sorted for stable windows
        big = big.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)
        # Parallel per MMSI
        out = (
            big.groupby("mmsi", sort=False)
               .parallel_apply(lambda g: _pipeline_one(g, args.gap_hours, args.max_sog))
        )
        # parallel_apply preserves a groupby index -> flatten
        if isinstance(out.index, pd.MultiIndex):
            out = out.reset_index(level=0, drop=True)
        df_proc = out.reset_index(drop=True)
    else:
        # Fallback: multiprocessing pool (robust everywhere)
        import multiprocessing as mp
        def _wrap(tup):
            _mmsi, g = tup
            return _pipeline_one(g, args.gap_hours, args.max_sog)

        if workers > 1:
            with mp.get_context("spawn").Pool(processes=workers) as pool:
                parts = list(tqdm(pool.imap_unordered(_wrap, groups), total=len(groups), desc="per-vessel"))
        else:
            parts = [ _wrap(g) for g in tqdm(groups, total=len(groups), desc="per-vessel (serial)") ]

        if parts:
            df_proc = pd.concat(parts, ignore_index=True)
        else:
            df_proc = pd.DataFrame(columns=["mmsi", "timestamp", "lat", "lon"])

    if df_proc.empty:
        raise SystemExit("No data after cleaning/segmenting.")

    # Final sort & write with PyArrow (fast)
    df_proc = df_proc.sort_values(["mmsi", "segment_id", "timestamp"]).reset_index(drop=True)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "interim.parquet"
    print(f"[make_interim] Writing {len(df_proc):,} rows → {outpath} (engine={args.engine})")
    df_proc.to_parquet(outpath, engine=args.engine, index=False)
    print(f"Saved interim dataset to {outpath} with {len(df_proc):,} rows")

if __name__ == "__main__":
    main()