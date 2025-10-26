from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.dataio.load_ais import load_ais_df
from src.dataio.clean import clean_ais_dataframe
from src.dataio.segment import segment_trajectories
from src.features.kinematics import add_kinematic_features
from src.features.context import add_directional_encodings, add_grid_features

def main():
    ap = argparse.ArgumentParser(description="Clean and segment AIS into interim parquet")
    ap.add_argument("--raw", nargs="+", required=True, help="Paths to raw CSV/Parquet files")
    ap.add_argument("--out", required=True, help="Output interim directory")
    ap.add_argument("--gap_hours", type=float, default=6.0)
    ap.add_argument("--max_sog", type=float, default=40.0)
    args = ap.parse_args()

    df = load_ais_df(args.raw)
    df = clean_ais_dataframe(df, max_sog=args.max_sog)
    df = segment_trajectories(df, gap_hours=args.gap_hours, min_len=10)
    df = add_kinematic_features(df, use_projection=True)
    df = add_directional_encodings(df)
    df = add_grid_features(df, cell_size_m=1000.0)

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "interim.parquet"
    df.to_parquet(outpath, index=False)
    print(f"Saved interim dataset to {outpath} with {len(df):,} rows")

if __name__ == "__main__":
    main()
