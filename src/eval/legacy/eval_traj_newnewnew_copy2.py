# src/eval/eval_traj_newnewnew.py
from __future__ import annotations
import argparse, json, os, csv, datetime as dt
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Iterable, Optional, Sequence, Tuple

from ..models import GRUSeq2Seq, TPTrans
from ..utils.datasets import AISDataset
from .metrics_traj import ade, fde

# Optional import for de-normalization from preprocessing
try:
    from ..preprocessing.preprocessing import de_normalize_track as _de_normalize_track
    _HAS_DENORM_FN = True
except Exception:
    _HAS_DENORM_FN = False

DEFAULT_EUROPE_EXTENT: Tuple[float, float, float, float] = (-25.0, 45.0, 30.0, 72.0)

# Try to import normalization bounds from the preprocessing pipeline.
_DEFAULT_LAT_MIN = None
_DEFAULT_LAT_MAX = None
_DEFAULT_LON_MIN = None
_DEFAULT_LON_MAX = None
try:
    from ..preprocessing.preprocessing import LAT_MIN as _LAT_MIN, LAT_MAX as _LAT_MAX, LON_MIN as _LON_MIN, LON_MAX as _LON_MAX  # type: ignore
    _DEFAULT_LAT_MIN, _DEFAULT_LAT_MAX = float(_LAT_MIN), float(_LAT_MAX)
    _DEFAULT_LON_MIN, _DEFAULT_LON_MAX = float(_LON_MIN), float(_LON_MAX)
except Exception:
    # Fallback: None → must be provided by CLI if denorm is desired
    pass


def build_model(kind: str, feat_dim: int, horizon: int, d_model=192, nhead=4, enc_layers=4, dec_layers=2):
    if kind == "gru":
        return GRUSeq2Seq(feat_dim, d_model=d_model, layers=2, horizon=horizon)
    return TPTrans(feat_dim=feat_dim, d_model=d_model, nhead=nhead,
                   enc_layers=enc_layers, dec_layers=dec_layers, horizon=horizon)


def split_lat_lon(arr: np.ndarray, lat_idx: int, lon_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (lats, lons) from a [T,C] array given channel indices."""
    return arr[:, lat_idx], arr[:, lon_idx]


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1 = np.radians([lat1, lon1])
    p2 = np.radians([lat2, lon2])
    dlat = p2[0] - p1[0]
    dlon = p2[1] - p1[1]
    a = np.sin(dlat/2.0)**2 + np.cos(p1[0])*np.cos(p2[0])*np.sin(dlon/2.0)**2
    return float(2*R*np.arcsin(np.sqrt(a)))


def to_iso(ts: float, fmt: str = "%Y-%m-%d %H:%M:%S UTC") -> str:
    return dt.datetime.utcfromtimestamp(float(ts)).strftime(fmt)


def fname_ts(ts: float) -> str:
    return dt.datetime.utcfromtimestamp(float(ts)).strftime("%Y-%m-%dT%H%M%SZ")


def extract_id_and_span(arrays: Sequence[np.ndarray], idx_timestamp: int = 7, idx_mmsi: int = 8) -> Tuple[int, float, float]:
    stacks = [a for a in arrays if a is not None and getattr(a, 'size', 0) > 0]
    if not stacks:
        return 0, 0.0, 0.0
    all_stacks = np.concatenate(stacks, axis=0)
    ts = all_stacks[:, idx_timestamp]
    mmsi_col = all_stacks[:, idx_mmsi]
    vals, counts = np.unique(mmsi_col.astype(np.int64), return_counts=True)
    mmsi = int(vals[np.argmax(counts)])
    t_start = float(np.nanmin(ts)); t_end = float(np.nanmax(ts))
    return mmsi, t_start, t_end


def _resolve_basemap_candidates(basemap_path: Optional[str]) -> Iterable[Path]:
    """
    Yield potential paths to a Natural Earth land dataset. We prioritise user-supplied
    paths, but fall back to common package locations so that the script works out
    of the box without extra downloads.
    """
    if basemap_path:
        yield Path(basemap_path).expanduser()

    # GeoPandas < 1.0 bundled the dataset; try it first if still available.
    try:
        import geopandas

        path = getattr(geopandas.datasets, "get_path", None)
        if path:
            yield Path(path("naturalearth_lowres"))
    except Exception:
        pass

    # GeoDatasets package (new official home for the sample data).
    try:
        import geodatasets  # type: ignore

        geodatasets_path = getattr(geodatasets, "get_path", None)
        if geodatasets_path:
            yield Path(geodatasets_path("naturalearth.land"))
    except Exception:
        pass

    # Pyogrio bundles Natural Earth for its test-suite; handy as an offline fallback.
    try:
        import pyogrio

        pyogrio_path = Path(pyogrio.__file__).resolve().parent / "tests" / "fixtures" / "naturalearth_lowres" / "naturalearth_lowres.shp"
        yield pyogrio_path
    except Exception:
        pass


def load_europe_basemap(basemap_path: Optional[str] = None):
    """
    Attempt to load a Natural Earth basemap as a GeoDataFrame (EPSG:4326).
    Returns None if geopandas is unavailable or no dataset can be located.
    """
    try:
        import geopandas as gpd  # Local import to avoid heavy dependency unless plotting
    except ImportError:
        print("[warn] geopandas not available; skipping basemap rendering.")
        return None

    for candidate in _resolve_basemap_candidates(basemap_path):
        if not candidate or not candidate.exists():
            continue
        try:
            europe = gpd.read_file(candidate)
        except Exception as exc:
            print(f"[warn] failed to read basemap from {candidate}: {exc}")
            continue

        # Ensure geographic CRS for plotting lon/lat directly.
        try:
            if europe.crs is None:
                europe = europe.set_crs("EPSG:4326")
            else:
                europe = europe.to_crs("EPSG:4326")
        except Exception:
            print(f"[warn] unable to reproject basemap from {candidate}; skipping.")
            continue

        # Restrict to Europe when continent metadata is available; otherwise clip by extent.
        if "continent" in europe.columns:
            mask = europe["continent"].isin(["Europe"])
            # Keep a few transcontinental countries that matter for AIS tracks.
            if "name" in europe.columns:
                mask |= europe["name"].isin(["Russia", "Turkey", "Cyprus"])
            europe = europe[mask]

        if hasattr(europe, "cx"):
            clipped = europe.cx[-30:45, 30:75]
            if not clipped.empty:
                europe = clipped

        if europe.empty:
            continue

        europe = europe.reset_index(drop=True)
        print(f"[info] Using basemap from {candidate}")
        return europe

    print("[warn] Could not locate a Natural Earth basemap; proceeding without map background.")
    return None


def robust_extent(lats: np.ndarray, lons: np.ndarray, pad: float = 0.75,
                  clamp: Tuple[float, float, float, float] = DEFAULT_EUROPE_EXTENT,
                  sigma: float = 3.0) -> Tuple[float, float, float, float]:
    """Compute an outlier-robust extent with padding and clamping to Europe."""
    lats = lats[np.isfinite(lats)]
    lons = lons[np.isfinite(lons)]
    if lats.size == 0 or lons.size == 0:
        return clamp

    def clip(arr):
        m = float(np.nanmean(arr))
        s = float(np.nanstd(arr))
        if not np.isfinite(s) or s == 0.0:
            return arr
        return arr[(arr >= m - sigma * s) & (arr <= m + sigma * s)]

    lats_c = clip(lats)
    lons_c = clip(lons)
    if lats_c.size >= 2 and lons_c.size >= 2:
        lat_min, lat_max = float(np.min(lats_c)), float(np.max(lats_c))
        lon_min, lon_max = float(np.min(lons_c)), float(np.max(lons_c))
    else:
        lat_min, lat_max = float(np.min(lats)), float(np.max(lats))
        lon_min, lon_max = float(np.min(lons)), float(np.max(lons))

    if abs(lat_max - lat_min) < 0.2:
        lat_min -= 0.5; lat_max += 0.5
    if abs(lon_max - lon_min) < 0.2:
        lon_min -= 0.5; lon_max += 0.5

    lat_min -= pad; lat_max += pad; lon_min -= pad; lon_max += pad
    lon_min = max(clamp[0], lon_min); lon_max = min(clamp[1], lon_max)
    lat_min = max(clamp[2], lat_min); lat_max = min(clamp[3], lat_max)
    return (lon_min, lon_max, lat_min, lat_max)

def maybe_denorm(track: np.ndarray, lat_idx: int, lon_idx: int, name: str = "array",
                 lat_min: Optional[float] = None, lat_max: Optional[float] = None,
                 lon_min: Optional[float] = None, lon_max: Optional[float] = None) -> np.ndarray:
    """Apply preprocessing.de_normalize_track if values look normalized in [0..1].
    Ensures lat is column 0 and lon is column 1 for the denorm function.
    Validates output ranges and logs actions.
    """
    if track.ndim != 2 or track.shape[0] == 0:
        return track
    lat = track[:, lat_idx]
    lon = track[:, lon_idx]
    looks_norm = (
        np.nanmin(lat) >= -0.1 and np.nanmax(lat) <= 1.1 and
        np.nanmin(lon) >= -0.1 and np.nanmax(lon) <= 1.1
    )
    if looks_norm and _HAS_DENORM_FN:
        # build temporary [lat,lon,sog,cog]
        tmp = np.zeros((track.shape[0], max(4, track.shape[1])), dtype=float)
        tmp[:, 0] = track[:, lat_idx]
        tmp[:, 1] = track[:, lon_idx]
        tmp = _de_normalize_track(tmp)
        track = track.copy()
        track[:, lat_idx] = tmp[:, 0]
        track[:, lon_idx] = tmp[:, 1]
        print(f"[denorm] Applied de_normalize_track to {name}.")
    else:
        print(f"[denorm] {name} already in degrees or denorm unavailable.")
    # Validate
    lat = track[:, lat_idx]; lon = track[:, lon_idx]
    try:
        if not (-90 <= float(np.nanmin(lat)) <= 90 and -90 <= float(np.nanmax(lat)) <= 90):
            print(f"[warn] {name}: latitude out of bounds after (de)norm; check indices/normalization.")
        if not (-180 <= float(np.nanmin(lon)) <= 180 and -180 <= float(np.nanmax(lon)) <= 180):
            print(f"[warn] {name}: longitude out of bounds after (de)norm; check indices/normalization.")
    except Exception:
        pass
    return track


def maybe_denorm_deltas(lat_delta: np.ndarray, lon_delta: np.ndarray,
                        lat_min: Optional[float] = None, lat_max: Optional[float] = None,
                        lon_min: Optional[float] = None, lon_max: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """If deltas look normalized, scale them by the [min,max] range without shifting.
    Falls back to no-op if bounds are not available.
    """
    def looks_norm(v):
        return np.nanmin(v) >= -0.1 and np.nanmax(v) <= 1.1
    if (lat_min is None or lat_max is None or lon_min is None or lon_max is None):
        return lat_delta, lon_delta
    if looks_norm(lat_delta) and looks_norm(lon_delta):
        lat_range = float(lat_max - lat_min)
        lon_range = float(lon_max - lon_min)
        return lat_delta * lat_range, lon_delta * lon_range
    return lat_delta, lon_delta


def plot_samples(
    samples,
    ds,
    model_kind: str,
    out_dir: Path,
    lat_idx: int,
    lon_idx: int,
    past_len: int,
    max_plots: int = 8,
    basemap=None,
    map_extent: Optional[Tuple[float, float, float, float]] = None,
    auto_extent: bool = False,
    extent_source: str = "actual",
    extent_outlier_sigma: float = 3.0,
    y_order: str = "latlon",  # order in Y arrays
    # prediction handling
    pred_is_delta: bool = False,
    anchor_pred: bool = True,
    # denorm bounds
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None,
    # stamping / metadata
    stamp_titles: bool = True,
    stamp_filename: bool = True,
    save_meta: bool = True,
    meta_path: str = "data/figures/traj_eval_meta.csv",
    timefmt: str = "%Y-%m-%d %H:%M:%S UTC",
    annotate_id: bool = False,
    # full trip context
    full_trip: bool = False,
    mmsi_filter: Optional[int] = None,
    max_hours: float = 24.0,
):
    import matplotlib.pyplot as plt
    # Optional cartopy import
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        HAS_CARTOPY = True
        proj = ccrs.PlateCarree()
    except Exception:
        HAS_CARTOPY = False
        proj = None

    out_dir.mkdir(parents=True, exist_ok=True)
    sel = list(range(min(max_plots, len(samples))))
    for i, idx in enumerate(sel):
        ds_idx, x, y_abs, y_pred_abs = samples[idx]       # x:[T,F], y_abs:[H,2], y_pred_abs:[H,2]

        # Build actual track with columns aligned to lat_idx/lon_idx
        T = min(past_len, x.shape[0])
        C = max(4, x.shape[1])
        actual_stack = np.zeros((T + y_abs.shape[0], C), dtype=float)
        # Past
        actual_stack[:T, lat_idx] = x[:T, lat_idx]
        actual_stack[:T, lon_idx] = x[:T, lon_idx]
        # True future from Y order
        if y_order.lower() == "lonlat":
            actual_stack[T:, lat_idx] = y_abs[:, 1]
            actual_stack[T:, lon_idx] = y_abs[:, 0]
        else:  # latlon
            actual_stack[T:, lat_idx] = y_abs[:, 0]
            actual_stack[T:, lon_idx] = y_abs[:, 1]

        # Predicted future track - map channels according to y_order
        pred_track = np.zeros((y_pred_abs.shape[0], C), dtype=float)
        if y_order.lower() == "lonlat":
            pred_track[:, lat_idx] = y_pred_abs[:, 1]
            pred_track[:, lon_idx] = y_pred_abs[:, 0]
        else:  # latlon
            pred_track[:, lat_idx] = y_pred_abs[:, 0]
            pred_track[:, lon_idx] = y_pred_abs[:, 1]

        # De-normalize actual
        actual_stack = maybe_denorm(actual_stack, lat_idx, lon_idx, name=f"actual[{idx}]",
                                    lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)

        # Extract arrays consistently for actual
        lats_a = actual_stack[:, lat_idx]
        lons_a = actual_stack[:, lon_idx]
        lats_true = lats_a[T:]
        lons_true = lons_a[T:]

        # Prepare predictions: delta or absolute
        if pred_is_delta:
            # Use raw deltas then scale to degrees if needed
            lats_p, lons_p = split_lat_lon(pred_track, lat_idx, lon_idx)
            lats_p, lons_p = maybe_denorm_deltas(lats_p, lons_p, lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)
            # Anchor to last past point
            cur_lat = lats_a[T-1]
            cur_lon = lons_a[T-1]
            lats_p = cur_lat + np.cumsum(lats_p)
            lons_p = cur_lon + np.cumsum(lons_p)
            print("[pred] treated as deltas (anchored by definition).")
        else:
            # Absolute predictions: de-normalize and optionally anchor
            pred_track = maybe_denorm(pred_track, lat_idx, lon_idx, name=f"pred[{idx}]",
                                      lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)
            lats_p, lons_p = split_lat_lon(pred_track, lat_idx, lon_idx)
            cur_lat = lats_a[T-1]
            cur_lon = lons_a[T-1]
            if anchor_pred and np.isfinite(lats_p[0]) and np.isfinite(lons_p[0]):
                d0 = haversine_km(cur_lat, cur_lon, float(lats_p[0]), float(lons_p[0]))
                if d0 > 5.0:
                    dlat = lats_p - lats_p[0]
                    dlon = lons_p - lons_p[0]
                    lats_p = cur_lat + dlat
                    lons_p = cur_lon + dlon
                    print(f"[anchor] Shifted absolute predictions to start at current pos (Δ≈{d0:.1f} km).")

        # Diagnostics
        try:
            med_a = float(np.nanmedian(lats_a))
            med_p = float(np.nanmedian(lats_p))
            print(f"[diag] sample {idx}: med(lat) actual={med_a:.2f}, pred={med_p:.2f}")
            if med_a > 50 and med_p < 20:
                print("[warn] Prediction latitude far from actual; check normalization or index order.")
        except Exception:
            pass

        # Additional diagnostics and cleaning
        print(f"[diag] cur=({cur_lat:.5f},{cur_lon:.5f}) pred[0]=({lats_p[0]:.5f},{lons_p[0]:.5f}) "
              f"pred[min/max lat]=({np.nanmin(lats_p):.2f},{np.nanmax(lats_p):.2f}) "
              f"pred[min/max lon]=({np.nanmin(lons_p):.2f},{np.nanmax(lons_p):.2f}) n_pred={len(lats_p)}")
        if np.allclose(lats_p, lats_p[0]) and np.allclose(lons_p, lons_p[0]):
            print("[warn] flat prediction (single point or zero-length line).")
        finite_mask = np.isfinite(lats_p) & np.isfinite(lons_p)
        if finite_mask.sum() < 2:
            print("[warn] no valid predicted points to plot after filtering.")
        lats_p = lats_p[finite_mask]; lons_p = lons_p[finite_mask]

        # Figure + map background
        if HAS_CARTOPY:
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=proj))
            ax.add_feature(cfeature.LAND, zorder=0)
            ax.add_feature(cfeature.OCEAN, zorder=0)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
            try:
                gl.top_labels = False; gl.right_labels = False
            except Exception:
                pass
        else:
            fig, ax = plt.subplots(figsize=(7.5, 6.5))
            if basemap is not None:
                basemap.plot(ax=ax, color="#f2f2f2", edgecolor="#a6a6a6", linewidth=0.6, zorder=0)
                ax.set_facecolor("#d6e6f5")
            else:
                ax.set_facecolor("#f6f7fb")

        # Extent selection
        if auto_extent:
            if extent_source == "actual":
                lats_src, lons_src = lats_a, lons_a
            elif extent_source == "pred":
                lats_src, lons_src = lats_p, lons_p
            else:
                lats_src = np.concatenate([lats_a, lats_p])
                lons_src = np.concatenate([lons_a, lons_p])
            ext = robust_extent(lats_src, lons_src, sigma=extent_outlier_sigma)
            # Ensure predictions remain visible when using actual-only extent
            if extent_source == "actual":
                lon_min, lon_max, lat_min_e, lat_max_e = ext
                in_lon = (lons_p >= lon_min) & (lons_p <= lon_max)
                in_lat = (lats_p >= lat_min_e) & (lats_p <= lat_max_e)
                if not np.any(in_lon & in_lat) and len(lats_p) > 0:
                    # enlarge once
                    cx = (lon_min + lon_max)/2.0; cy = (lat_min_e + lat_max_e)/2.0
                    w = (lon_max - lon_min) * 1.5; h = (lat_max_e - lat_min_e) * 1.5
                    new_ext = (cx - w/2, cx + w/2, cy - h/2, cy + h/2)
                    # clamp to Europe
                    lon_min = max(DEFAULT_EUROPE_EXTENT[0], new_ext[0])
                    lon_max = min(DEFAULT_EUROPE_EXTENT[1], new_ext[1])
                    lat_min_e = max(DEFAULT_EUROPE_EXTENT[2], new_ext[2])
                    lat_max_e = min(DEFAULT_EUROPE_EXTENT[3], new_ext[3])
                    ext = (lon_min, lon_max, lat_min_e, lat_max_e)
        else:
            ext = map_extent if map_extent is not None else DEFAULT_EUROPE_EXTENT

        if HAS_CARTOPY:
            ax.set_extent(ext, crs=proj)
        else:
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        print(f"[extent] Using extent {ext} (auto={auto_extent}, source={extent_source})")

        # Optional full trip context (draw entire track for this sample's source file)
        if full_trip:
            try:
                fpath = os.path.join(ds.data_dir, ds.file_list[ds_idx])
                import pickle
                Vfull = pickle.load(open(fpath, 'rb'))
                full = Vfull["traj"]
                if mmsi_filter is None or int(Vfull.get("mmsi", 0)) == int(mmsi_filter):
                    full_dn = maybe_denorm(full.copy(), lat_idx=0, lon_idx=1, name="full_trip",
                                           lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)
                    full_lats = full_dn[:, 0]; full_lons = full_dn[:, 1]
                    if HAS_CARTOPY:
                        ax.plot(full_lons, full_lats, color="#999999", linewidth=1.0, alpha=0.6, transform=proj, label="full trip (context)")
                    else:
                        ax.plot(full_lons, full_lats, color="#999999", linewidth=1.0, alpha=0.6, label="full trip (context)")
            except Exception as e:
                print(f"[warn] full_trip context failed for idx {ds_idx}: {e}")

        # Plot tracks
        if HAS_CARTOPY:
            ax.plot(lons_a[:T], lats_a[:T], '-', marker='o', markersize=2, color="#1f77b4", linewidth=1.6, transform=proj, label="past (input)")
            ax.plot([lons_a[T-1]], [lats_a[T-1]], 'o', color='k', markersize=4, transform=proj, label='current pos')
            ax.plot(lons_true, lats_true, '-', color="#2ca02c", linewidth=1.8, transform=proj, label="true future")
            if len(lats_p) >= 2:
                ax.plot(lons_p, lats_p, '--', color="#d62728", linewidth=1.8, transform=proj, label="pred future", zorder=5)
        else:
            ax.plot(lons_a[:T], lats_a[:T], '-', marker='o', markersize=2, color="#1f77b4", linewidth=1.6, label="past (input)")
            ax.plot([lons_a[T-1]], [lats_a[T-1]], 'o', color='k', markersize=4, label='current pos')
            ax.plot(lons_true, lats_true, '-', color="#2ca02c", linewidth=1.8, label="true future")
            if len(lats_p) >= 2:
                ax.plot(lons_p, lats_p, '--', color="#d62728", linewidth=1.8, label="pred future", zorder=5)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        # Identification: MMSI + time span
        mmsi = 0; t0 = 0.0; t1 = 0.0
        try:
            fpath = os.path.join(ds.data_dir, ds.file_list[ds_idx])
            import pickle
            Vraw = pickle.load(open(fpath, 'rb'))
            raw = Vraw["traj"]
            Hlen = y_abs.shape[0]
            past_raw = raw[:T, :]
            true_raw = raw[T:T+Hlen, :]
            mmsi, t0, t1 = extract_id_and_span([past_raw, true_raw], idx_timestamp=7, idx_mmsi=8)
            t0_iso = to_iso(t0, fmt=timefmt)
            t1_iso = to_iso(t1, fmt=timefmt)
        except Exception as e:
            print(f"[warn] failed to extract id/span for idx {ds_idx}: {e}")
            t0_iso = t1_iso = ""

        if stamp_titles:
            ax.set_title(f"Trajectory sample {ds_idx} ({model_kind}) — MMSI {mmsi} — {t0_iso} → {t1_iso}")
        else:
            ax.set_title(f"Trajectory sample {ds_idx} ({model_kind})")
        ax.legend(loc="upper left", frameon=True)

        if annotate_id:
            if HAS_CARTOPY:
                ax.text(lons_a[T-1], lats_a[T-1], f"MMSI {mmsi}\n{t0_iso} → {t1_iso}",
                        transform=proj, fontsize=8, ha="left", va="bottom",
                        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2), zorder=10)
            else:
                ax.text(lons_a[T-1], lats_a[T-1], f"MMSI {mmsi}\n{t0_iso} → {t1_iso}",
                        fontsize=8, ha="left", va="bottom",
                        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2), zorder=10)

        # Filename and save
        if stamp_filename and t0 and t1 and mmsi:
            out_name = f"traj_{model_kind}_mmsi-{mmsi}_{fname_ts(t0)}_{fname_ts(t1)}_idx-{ds_idx}.png"
        else:
            out_name = f"traj_{model_kind}_idx-{ds_idx}.png"
        fig_path = out_dir / out_name
        plt.savefig(fig_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        print(f"[plot] saved {fig_path}")
        print(f"[counts] n_past={T} n_true_future={len(lats_true)} n_pred={len(lats_p)} mode={'full_trip' if full_trip else 'window'}")

        # Append metadata CSV
        try:
            lats_all = lats_a
            lons_all = lons_a
            meta_row = {
                "sample_idx": int(ds_idx),
                "model": model_kind,
                "mmsi": int(mmsi),
                "t_start": float(t0),
                "t_end": float(t1),
                "t_start_iso": to_iso(t0, fmt=timefmt) if t0 else "",
                "t_end_iso": to_iso(t1, fmt=timefmt) if t1 else "",
                "n_past": int(T),
                "n_true_future": int(len(lats_true)),
                "n_pred": int(len(lats_p)),
                "out_path": str(fig_path),
                "lat_min": float(min(np.nanmin(lats_all), np.nanmin(lats_p))) if len(lats_p) else float(np.nanmin(lats_all)),
                "lat_max": float(max(np.nanmax(lats_all), np.nanmax(lats_p))) if len(lats_p) else float(np.nanmax(lats_all)),
                "lon_min": float(min(np.nanmin(lons_all), np.nanmin(lons_p))) if len(lons_p) else float(np.nanmin(lons_all)),
                "lon_max": float(max(np.nanmax(lons_all), np.nanmax(lons_p))) if len(lons_p) else float(np.nanmax(lons_all)),
            }
            if save_meta:
                os.makedirs(out_dir, exist_ok=True)
                file_exists = os.path.exists(meta_path)
                with open(meta_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(meta_row.keys()))
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(meta_row)
        except Exception as e:
            print(f"[warn] failed to write meta row for idx {ds_idx}: {e}")

def main():
    ap = argparse.ArgumentParser(description="Trajectory evaluation and plotting. Note: Past is a window of length --past_len, not the full MMSI trip.")
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", choices=["gru","tptrans"], default="tptrans")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_plots", type=int, default=8)
    ap.add_argument("--lat_idx", type=int, default=0)   # x feature order: [lat,lon,sog,cog]
    ap.add_argument("--lon_idx", type=int, default=1)
    ap.add_argument("--past_len", type=int, default=64) # how much of the past window to draw
    ap.add_argument("--out_dir", default="data/figures")
    ap.add_argument("--basemap_path", default=None,
                    help="Optional path to a Natural Earth land file (shp/geojson). Defaults to bundled fixtures if available.")
    ap.add_argument("--map_extent", nargs=4, type=float, metavar=("MIN_LON", "MAX_LON", "MIN_LAT", "MAX_LAT"),
                    help="Fix the map extent; by default the Europe view (-25 45 30 72) is used.")
    ap.add_argument("--auto_extent", action="store_true",
                    help="If set, zoom to each trajectory with padding instead of using the fixed Europe extent.")
    ap.add_argument("--denorm", action="store_true",
                    help="If set, convert normalized [0..1] lat/lon back to geographic degrees for plotting.")
    ap.add_argument("--lat_min", type=float, default=_DEFAULT_LAT_MIN, help="Lat min bound used during normalization (if denorm).")
    ap.add_argument("--lat_max", type=float, default=_DEFAULT_LAT_MAX, help="Lat max bound used during normalization (if denorm).")
    ap.add_argument("--lon_min", type=float, default=_DEFAULT_LON_MIN, help="Lon min bound used during normalization (if denorm).")
    ap.add_argument("--lon_max", type=float, default=_DEFAULT_LON_MAX, help="Lon max bound used during normalization (if denorm).")
    ap.add_argument("--y_order", choices=["latlon", "lonlat"], default="latlon",
                    help="Column order of Y/YP tensors. Use 'latlon' if Y[:,0]=lat, Y[:,1]=lon.")
    ap.add_argument("--extent_source", choices=["both","actual","pred"], default="actual",
                    help="Which points control auto-zoom extent (default: actual).")
    ap.add_argument("--extent_outlier_sigma", type=float, default=3.0,
                    help="Sigma for outlier clipping when computing auto-extent.")
    ap.add_argument("--pred_is_delta", action="store_true", help="Set if model outputs per-step deltas instead of absolute coords.")
    ap.add_argument("--anchor_pred", dest="anchor_pred", action="store_true", help="Anchor absolute predictions to current position if first point is far.")
    ap.add_argument("--no_anchor_pred", dest="anchor_pred", action="store_false")
    ap.set_defaults(anchor_pred=True)
    # stamping / meta
    ap.add_argument("--stamp_titles", dest="stamp_titles", action="store_true")
    ap.add_argument("--no_stamp_titles", dest="stamp_titles", action="store_false")
    ap.set_defaults(stamp_titles=True)
    ap.add_argument("--stamp_filename", dest="stamp_filename", action="store_true")
    ap.add_argument("--no_stamp_filename", dest="stamp_filename", action="store_false")
    ap.set_defaults(stamp_filename=True)
    ap.add_argument("--save_meta", dest="save_meta", action="store_true")
    ap.add_argument("--no_save_meta", dest="save_meta", action="store_false")
    ap.set_defaults(save_meta=True)
    ap.add_argument("--meta_path", type=str, default="data/figures/traj_eval_meta.csv")
    ap.add_argument("--timefmt", type=str, default="%Y-%m-%d %H:%M:%S UTC")
    ap.add_argument("--annotate_id", action="store_true", help="Draw MMSI + time span near current point.")
    # full trip
    ap.add_argument("--full_trip", action="store_true", help="Overlay full trip context for the sample's source file.")
    ap.add_argument("--mmsi_filter", type=int, default=None, help="When set, only overlay context if MMSI matches.")
    ap.add_argument("--max_hours", type=float, default=24.0, help="Max hours for context (currently advisory).")
    args = ap.parse_args()

    split = Path(args.split_dir)
    ds = AISDataset(str(split), max_seqlen=max(96, args.past_len + 12))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    x0, y0 = ds[0]                       # y0 is ABSOLUTE positions (lon,lat) for the horizon
    feat_dim = x0.shape[-1]; horizon = y0.shape[0]

    model = build_model(args.model, feat_dim, horizon)
    state = torch.load(args.ckpt, map_location="cpu")
    try:
        model.load_state_dict(state, strict=True)
    except Exception as e:
        print(f"[warn] strict load failed: {e}\n[info] retrying strict=False")
        model.load_state_dict(state, strict=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    preds, gts = [], []
    keep_for_plot = []
    seen = 0
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device).float()
            yb = yb.to(device).float()         # Ground-truth absolute positions (normalized), order given by dataset

            yp_raw = model(xb)                 # [B,H,2]: abs or deltas depending on training

            # Determine window length (past T) and last absolute point
            T = xb.size(1)
            last_lat = xb[:, T - 1, args.lat_idx]
            last_lon = xb[:, T - 1, args.lon_idx]
            if args.y_order == "lonlat":
                last_abs = torch.stack([last_lon, last_lat], dim=1)  # [B,2]
            else:  # latlon
                last_abs = torch.stack([last_lat, last_lon], dim=1)  # [B,2]

            if args.pred_is_delta:
                yp_abs = torch.cumsum(yp_raw, dim=1) + last_abs.unsqueeze(1)  # [B,H,2]
            else:
                yp_abs = yp_raw

            # Align pred order with ground-truth order for metrics (dataset Y is latlon)
            if args.y_order == "lonlat":
                yp_abs_for_metrics = yp_abs[:, :, [1, 0]]
            else:
                yp_abs_for_metrics = yp_abs

            yp_np_plot = yp_abs.cpu().numpy()  # keep order per y_order for plotting
            yb_np = yb.cpu().numpy()
            yp_metric = yp_abs_for_metrics.cpu().numpy()

            preds.append(yp_metric); gts.append(yb_np)
            if len(keep_for_plot) < args.max_plots:
                bsz = len(xb)
                take = min(4, bsz)
                for i in range(take):
                    keep_for_plot.append((seen + i, xb[i].cpu().numpy(), yb_np[i], yp_np_plot[i]))
                    if len(keep_for_plot) >= args.max_plots:
                        break
            seen += len(xb)

    pred = np.concatenate(preds, axis=0)
    Y    = np.concatenate(gts,   axis=0)
    ade_val = float(ade(pred, Y))
    fde_val = float(fde(pred, Y))
    print(f"VAL: ADE={ade_val:.3f}  FDE={fde_val:.3f}")

    metrics_dir = Path("metrics"); metrics_dir.mkdir(exist_ok=True)
    out_json = metrics_dir / f"traj_{args.model}_{split.name}.json"
    out_json.write_text(json.dumps({
        "task":"trajectory","model":args.model,"split":str(split),
        "ckpt":str(args.ckpt),"ade":ade_val,"fde":fde_val,"count":int(len(Y))
    }, indent=2))
    print(f"[metrics] wrote {out_json}")

    basemap = load_europe_basemap(args.basemap_path)
    map_extent = tuple(args.map_extent) if args.map_extent is not None else None

    # Note: Past is a window of length --past_len, not the full MMSI trip.
    print("[info] Note: Past plotted is a window of length --past_len, not the full MMSI trip. Use --full_trip for context.")

    plot_samples(keep_for_plot, ds, args.model, Path(args.out_dir),
                 args.lat_idx, args.lon_idx, past_len=args.past_len, max_plots=args.max_plots,
                 basemap=basemap, map_extent=map_extent, auto_extent=args.auto_extent,
                 extent_source=args.extent_source, extent_outlier_sigma=args.extent_outlier_sigma,
                 y_order=args.y_order,
                 pred_is_delta=args.pred_is_delta, anchor_pred=args.anchor_pred,
                 lat_min=args.lat_min, lat_max=args.lat_max,
                 lon_min=args.lon_min, lon_max=args.lon_max,
                 stamp_titles=args.stamp_titles, stamp_filename=args.stamp_filename,
                 save_meta=args.save_meta, meta_path=args.meta_path, timefmt=args.timefmt,
                 annotate_id=args.annotate_id, full_trip=args.full_trip,
                 mmsi_filter=args.mmsi_filter, max_hours=args.max_hours)

if __name__ == "__main__":
    main()
