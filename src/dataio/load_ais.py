"""Robust AIS loader for CSV/Parquet with minimal assumptions.
Expected columns (best-effort): mmsi, timestamp, lat, lon, sog, cog, heading, nav_status, shiptype, draught, destination
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

expected_columns = [
    "mmsi","timestamp","lat","lon","sog","cog","heading","nav_status","shiptype","draught","destination",
]

def load_ais_df(paths) -> pd.DataFrame:
    """Load multiple CSV/Parquet files into a single DataFrame.
    - Parses timestamps to pandas datetime (UTC-naive)
    - Keeps unknown columns as-is
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    frames = []
    for p in map(Path, paths):
        if p.suffix.lower() in [".parquet", ".pq"]:
            df = pd.read_parquet(p)
        elif p.suffix.lower() in [".csv", ".gz"]:
            df = pd.read_csv(p)
        else:
            raise ValueError(f"Unsupported file type: {p.suffix}")
        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]
        # Parse timestamp
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
        frames.append(df)
    if not frames:
        raise ValueError("No files loaded")
    df = pd.concat(frames, ignore_index=True)
    # Basic required columns check
    req = {"mmsi","timestamp","lat","lon"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df
