# src/dataio/load_ais.py
"""Fast, robust AIS loader for CSV/Parquet with minimal assumptions.
Expected canonical columns (best-effort mapping):
mmsi, timestamp, lat, lon, sog, cog, heading, nav_status, shiptype, draught, destination
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import pandas as pd
import numpy as np
import re

# Try optional high-performance backend
try:
    import pyarrow as pa
    import pyarrow.csv as pacsv
    HAVE_ARROW = True
except Exception:
    HAVE_ARROW = False

# Canonical (pipeline) column names we try to produce
expected_columns = [
    "mmsi", "timestamp", "lat", "lon", "sog", "cog",
    "heading", "nav_status", "shiptype", "draught", "destination",
]

# Common variants seen in AIS CSV exports -> canonical names
_VARIANTS_MAP = {
    # canonical
    "mmsi": "mmsi",
    "timestamp": "timestamp",
    "lat": "lat",
    "lon": "lon",
    "sog": "sog",
    "cog": "cog",
    "heading": "heading",
    "nav_status": "nav_status",
    "shiptype": "shiptype",
    "draught": "draught",
    "destination": "destination",

    # vendor variants
    "latitude": "lat",
    "longitude": "lon",
    "navigational_status": "nav_status",
    "ship_type": "shiptype",
    "cargo_type": "cargo_type",
    "rot": "rot",
    "imo": "imo",
    "callsign": "callsign",
    "name": "name",
    "eta": "eta",
    "type_of_mobile": "type_of_mobile",
    "type_of_position_fixing_device": "pos_fixing_device",
    "data_source_type": "data_source_type",
    "width": "width",
    "length": "length",
    "size_a": "size_a",
    "size_b": "size_b",
    "size_c": "size_c",
    "size_d": "size_d",

    # alternate timestamp labels
    "base_datetime": "timestamp",
    "basedatetime": "timestamp",
    "time": "timestamp",

    # your file's trailing columns 'A','B','C','D'
    "a": "size_a",
    "b": "size_b",
    "c": "size_c",
    "d": "size_d",
}

_REQUIRED = {"mmsi", "timestamp", "lat", "lon"}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _normalize_columns(cols: Iterable[str]) -> list[str]:
    """Lowercase, remove BOM/zero-width/NBSP, collapse unicode whitespace to '_',
    replace dashes with underscores, and strip leading punctuation like '#'."""
    normalized = []
    for c in cols:
        c = str(c)
        # remove BOM / zero-width / non-breaking space
        c = re.sub(r"[\uFEFF\u200B\u00A0]", "", c)
        # collapse any whitespace to underscore
        c = re.sub(r"\s+", "_", c)
        c = c.strip().lower().replace("-", "_")
        # strip leading non-alphanumeric chars (e.g., '#_timestamp' -> 'timestamp')
        c = re.sub(r"^[^a-z0-9]+", "", c)
        normalized.append(c)
    return normalized


def _rename_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    cols_norm = _normalize_columns(df.columns)
    df = df.set_axis(cols_norm, axis=1, copy=False)
    rename_dict = {c: _VARIANTS_MAP[c] for c in df.columns if c in _VARIANTS_MAP}
    if rename_dict:
        df = df.rename(columns=rename_dict)
    return df


def _parse_timestamp(series: pd.Series) -> pd.Series:
    # try dayfirst (31/12/2015 23:59:59), then fallback
    ts = pd.to_datetime(series, errors="coerce", dayfirst=True, infer_datetime_format=True)
    if ts.isna().mean() > 0.5:
        ts2 = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
        ts = ts.fillna(ts2)
    return ts


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


# ----------------------------------------------------------------------
# High-speed CSV reader
# ----------------------------------------------------------------------
def _read_csv_fast(path: Path) -> pd.DataFrame:
    """Try PyArrow CSV first (multi-threaded). Fallback to pandas C engine."""
    if HAVE_ARROW:
        for delim in [",", ";", "\t", "|"]:
            try:
                tbl = pacsv.read_csv(
                    path.as_posix(),
                    read_options=pacsv.ReadOptions(autogenerate_column_names=False),
                    parse_options=pacsv.ParseOptions(delimiter=delim),
                    convert_options=pacsv.ConvertOptions(strings_can_be_null=True),
                )
                df = tbl.to_pandas(types_mapper=pd.ArrowDtype)
                # normalize immediately (handles '# Timestamp' -> 'timestamp')
                df.columns = _normalize_columns(df.columns)
                return df
            except Exception:
                continue
    # Fallbacks
    for delim in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=delim, engine="c")
            df.columns = _normalize_columns(df.columns)
            return df
        except Exception:
            continue
    # Last resort
    df = pd.read_csv(path, engine="python")
    df.columns = _normalize_columns(df.columns)
    return df


# ----------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------
def load_ais_df(paths) -> pd.DataFrame:
    """Load one or more CSV/Parquet files and return a unified DataFrame with canonical columns."""
    if isinstance(paths, (str, Path)):
        paths = [paths]

    frames: list[pd.DataFrame] = []
    for p in map(Path, paths):
        if p.suffix.lower() in (".parquet", ".pq"):
            df = pd.read_parquet(p, engine="pyarrow")
            df.columns = _normalize_columns(df.columns)
        elif p.suffix.lower() in (".csv", ".gz"):
            df = _read_csv_fast(p)
        else:
            raise ValueError(f"Unsupported file type: {p.suffix}")

        df = _rename_to_canonical(df)

        if "timestamp" in df.columns:
            df["timestamp"] = _parse_timestamp(df["timestamp"])

        _coerce_numeric(df, ["lat", "lon", "sog", "cog", "heading", "draught", "rot"])
        if "mmsi" in df.columns:
            df["mmsi"] = pd.to_numeric(df["mmsi"], errors="coerce").astype("Int64")
        frames.append(df)

    if not frames:
        raise ValueError("No files loaded")

    df_all = pd.concat(frames, ignore_index=True)

    # Required column check
    missing = _REQUIRED - set(df_all.columns)
    if missing:
        hint = []
        if "latitude" in df_all.columns or "longitude" in df_all.columns:
            hint.append("Detected 'Latitude/Longitude' — mapped to 'lat/lon'.")
        if "timestamp" not in df_all.columns:
            hint.append("Header like '# Timestamp' is now normalized → 'timestamp'.")
        raise ValueError(f"Missing required columns: {missing}. " + " ".join(hint))

    # Drop invalid rows
    df_all = df_all.dropna(subset=list(_REQUIRED))
    if df_all["mmsi"].isna().any():
        df_all = df_all.dropna(subset=["mmsi"])
    df_all["mmsi"] = df_all["mmsi"].astype(np.int64)

    # Clip out-of-range coordinates
    df_all = df_all[(df_all["lat"].between(-90, 90)) & (df_all["lon"].between(-180, 180))]

    # Sort for segmentation
    if "mmsi" in df_all.columns and "timestamp" in df_all.columns:
        df_all = df_all.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)

    return df_all