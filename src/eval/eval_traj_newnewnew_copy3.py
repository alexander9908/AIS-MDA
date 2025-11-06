# src/eval/eval_traj_newnewnew.py
from __future__ import annotations
import argparse, json, os, csv, sys, glob, pickle, datetime as dt
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Iterable, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _HAS_CARTOPY = True
except Exception:
    _HAS_CARTOPY = False

from ..models import GRUSeq2Seq, TPTrans
from ..utils.datasets import AISDataset
from .metrics_traj import ade, fde

# Optional import for de-normalization from preprocessing
try:
    from ..preprocessing.preprocessing import de_normalize_track as _de_normalize_track
    _HAS_DENORM_FN = True
except Exception:
    _HAS_DENORM_FN = False

# Denmark default extent (lon_min, lon_max, lat_min, lat_max)
DEFAULT_DENMARK_EXTENT: Tuple[float, float, float, float] = (6.0, 16.0, 54.0, 58.0)

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


def to_lonlat(arr: np.ndarray, lat_i: int, lon_i: int) -> Tuple[np.ndarray, np.ndarray]:
    return arr[:, lon_i], arr[:, lat_i]


def build_denmark_axes(auto_extent: bool,
                       extent_source_points: Optional[np.ndarray],
                       sigma: float,
                       figsize: Tuple[float, float] = (10, 6)):
    if _HAS_CARTOPY:
        proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})
        try:
            ax.add_feature(cfeature.OCEAN, zorder=0)
            ax.add_feature(cfeature.LAND, facecolor="0.92", zorder=1)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=2)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=2)
        except Exception as e:
            print(f"[warn] cartopy features failed: {e}; using coastlines only")
            ax.coastlines(resolution="50m", linewidth=0.6)
        try:
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
            gl.top_labels = False; gl.right_labels = False
        except Exception as e:
            print(f"[warn] gridlines failed: {e}")

        if auto_extent and extent_source_points is not None and len(extent_source_points) > 0:
            pts = np.asarray(extent_source_points, dtype=float)
            lon = pts[:, 0]; lat = pts[:, 1]
            m_lon, s_lon = float(np.nanmean(lon)), float(np.nanstd(lon))
            m_lat, s_lat = float(np.nanmean(lat)), float(np.nanstd(lat))
            lon_min, lon_max = m_lon - sigma * s_lon, m_lon + sigma * s_lon
            lat_min, lat_max = m_lat - sigma * s_lat, m_lat + sigma * s_lat
            lon_min = max(lon_min, DEFAULT_DENMARK_EXTENT[0]); lon_max = min(lon_max, DEFAULT_DENMARK_EXTENT[1])
            lat_min = max(lat_min, DEFAULT_DENMARK_EXTENT[2]); lat_max = min(lat_max, DEFAULT_DENMARK_EXTENT[3])
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
        else:
            ax.set_extent(DEFAULT_DENMARK_EXTENT, crs=proj)
        return fig, ax, proj
    else:
        fig, ax = plt.subplots(figsize=figsize)
        if auto_extent and extent_source_points is not None and len(extent_source_points) > 0:
            pts = np.asarray(extent_source_points, dtype=float)
            lon = pts[:, 0]; lat = pts[:, 1]
            m_lon, s_lon = float(np.nanmean(lon)), float(np.nanstd(lon))
            m_lat, s_lat = float(np.nanmean(lat)), float(np.nanstd(lat))
            lon_min, lon_max = m_lon - sigma * s_lon, m_lon + sigma * s_lon
            lat_min, lat_max = m_lat - sigma * s_lat, m_lat + sigma * s_lat
            lon_min = max(lon_min, DEFAULT_DENMARK_EXTENT[0]); lon_max = min(lon_max, DEFAULT_DENMARK_EXTENT[1])
            lat_min = max(lat_min, DEFAULT_DENMARK_EXTENT[2]); lat_max = min(lat_max, DEFAULT_DENMARK_EXTENT[3])
            ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)
        else:
            ax.set_xlim(DEFAULT_DENMARK_EXTENT[0], DEFAULT_DENMARK_EXTENT[1])
            ax.set_ylim(DEFAULT_DENMARK_EXTENT[2], DEFAULT_DENMARK_EXTENT[3])
        return fig, ax, None


def to_iso(ts: float, fmt: str = "%Y-%m-%d %H:%M:%S UTC") -> str:
    return dt.datetime.fromtimestamp(float(ts), dt.timezone.utc).strftime(fmt)


def fname_ts(ts: float) -> str:
    return dt.datetime.fromtimestamp(float(ts), dt.timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


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


def parse_trip(fname: str) -> Tuple[int, int]:
    base = os.path.basename(fname).replace("_processed.pkl", "")
    mmsi_str, trip_id_str = base.split("_", 1)
    return int(mmsi_str), int(trip_id_str)


def load_trip(path: str, min_points: int = 30) -> np.ndarray:
    with open(path, "rb") as f:
        data = pickle.load(f)
    # Accept either dict with 'traj' or array directly
    trip = data["traj"] if isinstance(data, dict) and "traj" in data else np.asarray(data)
    trip = np.asarray(trip)
    if len(trip) < int(min_points):
        raise ValueError(f"too short: {len(trip)} points")
    ts = trip[:, 7]
    if not np.all(ts[:-1] <= ts[1:]):
        order = np.argsort(ts)
        trip = trip[order]
    return trip


def split_by_percent(trip: np.ndarray, pct: float) -> Tuple[np.ndarray, np.ndarray, int]:
    n = len(trip)
    cut = max(1, min(n - 2, int(round(n * pct / 100.0))))
    past = trip[:cut]
    future_true = trip[cut:]
    return past, future_true, cut


def evaluate_and_plot_trip(
    fpath: str,
    trip: np.ndarray,
    model,
    args,
    out_root: Path,
    basemap,
    proj,
    sample_idx: int,
):
    # Trip columns: [lat, lon, sog, cog, heading, rot, nav, timestamp, mmsi]
    mmsi, tid = parse_trip(fpath)

    n_total = len(trip)
    # Decide cut: either pred_cut percent, or past_len window
    if args.pred_cut is not None:
        past, fut_true, cut = split_by_percent(trip, args.pred_cut)
    else:
        cut = min(args.past_len, n_total - 2)
        past = trip[:cut]
        fut_true = trip[cut:]

    # Cap future length and model horizon
    fut_len = len(fut_true)
    horizon = getattr(model, 'horizon', fut_len)
    steps = fut_len
    if args.cap_future is not None:
        steps = min(steps, int(args.cap_future))
    steps = min(steps, int(horizon))
    if steps < 1:
        raise ValueError("no future steps to predict after cap/horizon")

    # Denormalize full trip for plotting, and extract segments in degrees
    full_dn = maybe_denorm(trip.copy(), lat_idx=0, lon_idx=1, name="full_trip",
                           lat_min=args.lat_min, lat_max=args.lat_max, lon_min=args.lon_min, lon_max=args.lon_max)
    lats_full = full_dn[:, 0]; lons_full = full_dn[:, 1]
    lats_past = lats_full[:cut]; lons_past = lons_full[:cut]
    lats_true = lats_full[cut:cut+steps]; lons_true = lons_full[cut:cut+steps]

    # Current point (degrees) — last past sample (immutable reference)
    past_last_lat = float(lats_past[-1]); past_last_lon = float(lons_past[-1])
    cur_lat = past_last_lat; cur_lon = past_last_lon

    # Helpers for normalization conversions
    def deg_to_norm(lat_deg: np.ndarray, lon_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        la_min, la_max = args.lat_min, args.lat_max
        lo_min, lo_max = args.lon_min, args.lon_max
        if None in (la_min, la_max, lo_min, lo_max):
            raise ValueError("deg_to_norm requires --lat_min/--lat_max/--lon_min/--lon_max bounds when --denorm is used.")
        lat_norm = (lat_deg - la_min) / float(la_max - la_min)
        lon_norm = (lon_deg - lo_min) / float(lo_max - lo_min)
        return lat_norm, lon_norm

    # Iterative rollout or single-shot
    # Slice to same feature dimension as training (lat, lon, sog, cog)
    seq_in = past[:, :4].astype(np.float32)

    # Normalize inputs if they look like degrees; otherwise assume already normalized
    def looks_norm_latlon(lat_col, lon_col):
        return (np.nanmin(lat_col) >= -0.05 and np.nanmax(lat_col) <= 1.2 and
                np.nanmin(lon_col) >= -0.05 and np.nanmax(lon_col) <= 1.2)

    if looks_norm_latlon(seq_in[:, 0], seq_in[:, 1]):
        seq_norm = seq_in
    else:
        la_min, la_max = args.lat_min, args.lat_max
        lo_min, lo_max = args.lon_min, args.lon_max
        if None in (la_min, la_max, lo_min, lo_max):
            raise ValueError("Inputs appear denormalized; provide --lat_min/--lat_max/--lon_min/--lon_max for normalization.")
        seq_norm = seq_in.copy()
        seq_norm[:, 0] = (seq_in[:, 0] - la_min) / float(la_max - la_min)
        seq_norm[:, 1] = (seq_in[:, 1] - lo_min) / float(lo_max - lo_min)
        try:
            from ..preprocessing.preprocessing import SPEED_MAX  # type: ignore
            speed_max = float(SPEED_MAX)
        except Exception:
            speed_max = 30.0
        seq_norm[:, 2] = np.clip(seq_in[:, 2] / float(speed_max), 0.0, 1.0)
        seq_norm[:, 3] = (seq_in[:, 3] % 360.0) / 360.0

    last_sog = seq_norm[-1, 2] if seq_norm.shape[1] > 2 else 0.0
    last_cog = seq_norm[-1, 3] if seq_norm.shape[1] > 3 else 0.0

    print(f"[dbg] order={args.y_order} lat_idx={args.lat_idx} lon_idx={args.lon_idx}")
    print(f"[debug] model horizon={getattr(model,'horizon',None)} past_len={args.past_len} input_seq_shape={seq_norm.shape}")
    pred_lat_list: list[float] = []
    pred_lon_list: list[float] = []

    remaining = steps
    while remaining > 0:
        Tin = min(args.past_len, len(seq_norm))
        X_in = seq_norm[-Tin:, :][None, ...]
        with torch.no_grad():
            xb = torch.from_numpy(X_in).to(next(model.parameters()).device)
            ypred_raw = model(xb)[0].cpu().numpy()  # [H,2]

        # Align to (lat,lon) order
        if args.y_order == 'lonlat':
            pred_lat_norm = ypred_raw[:, 1]
            pred_lon_norm = ypred_raw[:, 0]
        else:
            pred_lat_norm = ypred_raw[:, 0]
            pred_lon_norm = ypred_raw[:, 1]
        # Clamp absolutes to [0,1] before de-normalization (stability)
        pred_lat_norm = np.clip(pred_lat_norm, 0.0, 1.0)
        pred_lon_norm = np.clip(pred_lon_norm, 0.0, 1.0)

        # Number of steps to keep this iter
        keep = min(len(pred_lat_norm), remaining)
        pred_lat_norm = pred_lat_norm[:keep]
        pred_lon_norm = pred_lon_norm[:keep]

        # Convert to degrees; handle deltas or absolute
        pred_is_delta = args.pred_is_delta or (getattr(args, 'pred_mode', 'absolute') == 'delta')
        if pred_is_delta:
            # Scale deltas to degrees then anchor+cumsum
            dlat_deg, dlon_deg = maybe_denorm_deltas(pred_lat_norm, pred_lon_norm,
                                                     lat_min=args.lat_min, lat_max=args.lat_max,
                                                     lon_min=args.lon_min, lon_max=args.lon_max)
            seg_lat = cur_lat + np.cumsum(dlat_deg)
            seg_lon = cur_lon + np.cumsum(dlon_deg)
        else:
            # Absolute normalized → degrees, then anchor if needed
            pred_track = np.stack([pred_lat_norm, pred_lon_norm], axis=1)
            pred_track = maybe_denorm(pred_track, lat_idx=0, lon_idx=1, name="pred_iter",
                                      lat_min=args.lat_min, lat_max=args.lat_max, lon_min=args.lon_min, lon_max=args.lon_max)
            seg_lat = pred_track[:, 0]
            seg_lon = pred_track[:, 1]
            if args.anchor_pred and np.isfinite(seg_lat[0]) and np.isfinite(seg_lon[0]):
                # Always anchor first point to current past; then guardrail check
                dlat0 = cur_lat - float(seg_lat[0]); dlon0 = cur_lon - float(seg_lon[0])
                seg_lat = seg_lat + dlat0
                seg_lon = seg_lon + dlon0
                d0 = haversine_km(cur_lat, cur_lon, float(seg_lat[0]), float(seg_lon[0]))
                if d0 > 0.5:
                    print(f"[warn] large first_pred jump after anchoring: {d0:.2f} km")

        # Append to overall preds and update state
        pred_lat_list.extend(seg_lat.tolist())
        pred_lon_list.extend(seg_lon.tolist())
        cur_lat = pred_lat_list[-1]
        cur_lon = pred_lon_list[-1]

        # Update normalized sequence for next iteration using converted normalized absolutes
        add_lat_norm, add_lon_norm = deg_to_norm(np.asarray(seg_lat), np.asarray(seg_lon))
        add_feats = np.stack([
            add_lat_norm,
            add_lon_norm,
            np.full_like(add_lat_norm, last_sog, dtype=np.float32),
            np.full_like(add_lon_norm, last_cog, dtype=np.float32),
        ], axis=1)
        seq_norm = np.vstack([seq_norm, add_feats.astype(np.float32)])

        remaining -= keep

        if not args.iter_rollout:
            break

    pred_lat = np.asarray(pred_lat_list, dtype=float)
    pred_lon = np.asarray(pred_lon_list, dtype=float)
    # Enforce hard continuity at the very start of the predicted sequence (anchor to last past)
    if len(pred_lat) > 0 and np.isfinite(pred_lat[0]) and np.isfinite(pred_lon[0]):
        dlat0 = past_last_lat - float(pred_lat[0])
        dlon0 = past_last_lon - float(pred_lon[0])
        pred_lat = pred_lat + dlat0
        pred_lon = pred_lon + dlon0
        # guardrail
        try:
            d0 = haversine_km(past_last_lat, past_last_lon, float(pred_lat[0]), float(pred_lon[0]))
            if d0 > 0.5:
                print(f"[warn] first_pred still far after final anchoring: {d0:.2f} km")
        except Exception:
            pass

    # Diagnostics
    print(f"[trip] mmsi={mmsi} trip_id={tid} n_total={n_total} cut={cut} steps={steps}")
    print(f"[diag] cur=({past_last_lat:.5f},{past_last_lon:.5f}) pred0=({pred_lat[0]:.5f},{pred_lon[0]:.5f})")

    # Extent (auto or Europe)
    lats_src = np.concatenate([lats_past, lats_true, pred_lat])
    lons_src = np.concatenate([lons_past, lons_true, pred_lon])
    if args.auto_extent:
        # Include entire trip (gray) + segments for robust, clamped extent
        lats_all = np.concatenate([lats_full, lats_src])
        lons_all = np.concatenate([lons_full, lons_src])
        ext = robust_extent(lats_all, lons_all, sigma=args.extent_outlier_sigma)
    else:
        ext = tuple(args.map_extent) if args.map_extent is not None else DEFAULT_DENMARK_EXTENT

    # Basemap + plotting
    import matplotlib.pyplot as plt
    HAS_CARTOPY = False
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        HAS_CARTOPY = True
        proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=(9, 6), subplot_kw={'projection': proj})
        # Basemap features
        ax.add_feature(cfeature.OCEAN, zorder=0)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=2)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
        gl.top_labels = False; gl.right_labels = False
        ax.set_extent(ext, crs=proj)
        # Full trip context
        ax.plot(lons_full, lats_full, color="#999999", linewidth=1.0, alpha=0.3, transform=proj, label="full trip (context)", zorder=1)
        # Segments with layered zorder to keep blue visible
        ax.plot(lons_past, lats_past, '-', color="#1f77b4", linewidth=1.8, transform=proj, label="past (input)", zorder=3)
        ax.plot([past_last_lon], [past_last_lat], 'o', color='k', markersize=4, transform=proj, label='current pos', zorder=4)
        ax.plot(lons_true, lats_true, '-', color="#2ca02c", linewidth=1.8, transform=proj, label="true future", zorder=4)
        # Ensure red starts at current pos by prepending the current point
        pred_lon_plot = np.concatenate([[past_last_lon], pred_lon]) if len(pred_lon) else np.array([past_last_lon])
        pred_lat_plot = np.concatenate([[past_last_lat], pred_lat]) if len(pred_lat) else np.array([past_last_lat])
        ax.plot(pred_lon_plot, pred_lat_plot, '--', color="#d62728", linewidth=2.0, transform=proj, label="pred future", zorder=5)
    except Exception:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.plot(lons_full, lats_full, color="#999999", linewidth=1.0, alpha=0.3, label="full trip (context)", zorder=1)
        ax.plot(lons_past, lats_past, '-', color="#1f77b4", linewidth=1.8, label="past (input)", zorder=3)
        ax.plot([past_last_lon], [past_last_lat], 'o', color='k', markersize=4, label='current pos', zorder=4)
        ax.plot(lons_true, lats_true, '-', color="#2ca02c", linewidth=1.8, label="true future", zorder=4)
        pred_lon_plot = np.concatenate([[past_last_lon], pred_lon]) if len(pred_lon) else np.array([past_last_lon])
        pred_lat_plot = np.concatenate([[past_last_lat], pred_lat]) if len(pred_lat) else np.array([past_last_lat])
        ax.plot(pred_lon_plot, pred_lat_plot, '--', color="#d62728", linewidth=2.0, label="pred future", zorder=5)

    # Title and annotation
    t0 = float(np.nanmin(trip[:, 7])); t1 = float(np.nanmax(trip[:, 7]))
    t0_iso = to_iso(t0, fmt=args.timefmt); t1_iso = to_iso(t1, fmt=args.timefmt)
    if args.stamp_titles:
        ax.set_title(f"Trajectory sample {sample_idx} ({args.model}) — MMSI {mmsi} — {t0_iso} → {t1_iso}")
    if args.annotate_id:
        if HAS_CARTOPY:
            ax.text(past_last_lon, past_last_lat, f"MMSI {mmsi}\n{t0_iso} → {t1_iso}",
                    transform=proj, fontsize=8, ha="left", va="bottom",
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2), zorder=10)
        else:
            ax.text(past_last_lon, past_last_lat, f"MMSI {mmsi}\n{t0_iso} → {t1_iso}",
                    fontsize=8, ha="left", va="bottom",
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2), zorder=10)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.legend(loc="upper left", frameon=True)

    # Output paths
    out_dir = Path(out_root)
    if args.output_per_mmsi_subdir:
        out_dir = out_dir / str(mmsi)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.stamp_filename:
        out_name = f"traj_{args.model}_mmsi-{mmsi}_trip-{tid}_cut-{args.pred_cut or 'none'}_idx-{sample_idx}.png"
    else:
        out_name = f"traj_{args.model}_idx-{sample_idx}.png"
    fig_path = out_dir / out_name
    plt.savefig(fig_path, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"[ok] saved {fig_path}")

    # Metadata
    meta_row = {
        "sample_idx": int(sample_idx),
        "model": args.model,
        "mmsi": int(mmsi),
        "trip_id": int(tid),
        "mode": "single" if isinstance(args.mmsi, str) and args.mmsi != 'all' else ("batch_all" if isinstance(args.mmsi, str) and args.mmsi.lower()=="all" else "multi"),
        "pred_cut": float(args.pred_cut) if args.pred_cut is not None else None,
        "n_total": int(n_total),
        "n_past": int(len(lats_past)),
        "n_true_future": int(len(lats_true)),
        "n_pred": int(len(pred_lat)),
        "t_start_iso": t0_iso,
        "t_end_iso": t1_iso,
        "lat_min": float(np.nanmin(lats_src)),
        "lat_max": float(np.nanmax(lats_src)),
        "lon_min": float(np.nanmin(lons_src)),
        "lon_max": float(np.nanmax(lons_src)),
        "out_path": str(fig_path),
    }
    if args.save_meta:
        # Write to per-mmsi local csv if requested, plus global if specified
        if args.output_per_mmsi_subdir:
            local_meta = out_dir / "traj_eval_meta.csv"
            file_exists = local_meta.exists()
            with open(local_meta, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(meta_row.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(meta_row)
        # Global
        if args.meta_path:
            global_meta = Path(args.meta_path)
            global_meta.parent.mkdir(parents=True, exist_ok=True)
            file_exists = global_meta.exists()
            with open(global_meta, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(meta_row.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(meta_row)

    # Optional debug NPZ dump
    if getattr(args, 'debug_save_npz', False):
        dbg_dir = out_dir / 'debug_npz'
        dbg_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dbg_dir / f"debug_{mmsi}_{tid}_idx-{sample_idx}.npz",
            full_lat=lats_full, full_lon=lons_full,
            past_lat=lats_past, past_lon=lons_past,
            true_lat=lats_true, true_lon=lons_true,
            pred_lat=pred_lat, pred_lon=pred_lon,
            cur_lat=past_last_lat, cur_lon=past_last_lon,
            extent=np.asarray(ext),
        )


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
                  clamp: Tuple[float, float, float, float] = DEFAULT_DENMARK_EXTENT,
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
    # clamp to Denmark bounds
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
                    # clamp to Denmark
                    lon_min = max(DEFAULT_DENMARK_EXTENT[0], new_ext[0])
                    lon_max = min(DEFAULT_DENMARK_EXTENT[1], new_ext[1])
                    lat_min_e = max(DEFAULT_DENMARK_EXTENT[2], new_ext[2])
                    lat_max_e = min(DEFAULT_DENMARK_EXTENT[3], new_ext[3])
                    ext = (lon_min, lon_max, lat_min_e, lat_max_e)
        else:
            ext = map_extent if map_extent is not None else DEFAULT_DENMARK_EXTENT

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
    ap.add_argument("--ckpt", required=False, default=None)
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
                    help="Fix the map extent; default is Denmark clamp (6 16 54 58).")
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
    ap.add_argument("--pred_mode", choices=["absolute","delta"], default="absolute", help="Prediction mode: absolute outputs or per-step deltas.")
    # selection / modes
    ap.add_argument("--mmsi", type=str, default=None, help="Select MMSI: omit for default windows; 'all' for batch; or numeric ID")
    ap.add_argument("--trip_id", type=int, default=0, help="Trip index when --mmsi is numeric")
    ap.add_argument("--pred_cut", type=float, default=None, help="%% of trip to treat as past before predicting tail")
    ap.add_argument("--cap_future", type=int, default=None, help="Cap predicted horizon steps")
    ap.add_argument("--min_points", type=int, default=30, help="Skip too-short trips")
    ap.add_argument("--output_per_mmsi_subdir", action="store_true", help="Save outputs in per-MMSI subfolders under out_dir")
    ap.add_argument("--list_only", action="store_true", help="Dry-run; list selected files and exit")
    ap.add_argument("--log_skip_reasons", action="store_true", help="Print skip reason for each trip")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility of selection")
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
    ap.add_argument("--timefmt", type=str, default="%Y-%m-%d %H:%M:%S UTC",
                    help="Time format for titles/meta (use strftime tokens like %%Y-%%m-%%d %%%%H:%%%%M:%%%%S UTC)")
    ap.add_argument("--annotate_id", action="store_true", help="Draw MMSI + time span near current point.")
    ap.add_argument("--debug_save_npz", action="store_true", help="Dump past/true/pred/full arrays to out_dir/debug_npz for inspection.")
    # full trip
    ap.add_argument("--full_trip", action="store_true", help="Overlay full trip context for the sample's source file.")
    ap.set_defaults(full_trip=True)
    ap.add_argument("--iter_rollout", action="store_true", help="Iteratively roll out predictions to match tail length (or cap).")
    ap.add_argument("--mmsi_filter", type=int, default=None, help="When set, only overlay context if MMSI matches.")
    ap.add_argument("--max_hours", type=float, default=24.0, help="Max hours for context (currently advisory).")
    args = ap.parse_args()

    split = Path(args.split_dir)
    files = sorted(glob.glob(os.path.join(args.split_dir, "*.pkl")))

    def _to_int_or_keep(x):
        try:
            return int(x)
        except Exception:
            return x

    if args.mmsi is None:
        mode = "multi"
    elif isinstance(args.mmsi, str) and args.mmsi.lower() == "all":
        mode = "batch_all"
    else:
        mode = "single"
        args.mmsi = _to_int_or_keep(args.mmsi)

    print(f"[mode] {mode}")
    # List-only: dry run over files
    files = sorted(glob.glob(os.path.join(args.split_dir, "*.pkl")))
    if args.list_only:
        if mode == "multi":
            rng = np.random.default_rng(args.seed)
            n_select = min(len(files), args.max_plots) if args.max_plots else len(files)
            if len(files) > n_select:
                idx = rng.choice(len(files), size=n_select, replace=False)
                files_to_eval = [files[i] for i in sorted(idx)]
            else:
                files_to_eval = files[:n_select]
        elif mode == "batch_all":
            files_to_eval = files[:args.max_plots] if args.max_plots else files
        else:
            tid = int(args.trip_id or 0)
            files_to_eval = [os.path.join(args.split_dir, f"{int(args.mmsi)}_{tid}_processed.pkl")]
        print("\n[files selected for evaluation]\n")
        for f in files_to_eval:
            try:
                m, t = parse_trip(f)
                print(f" - {os.path.basename(f)}  (MMSI={m}, trip_id={t})")
            except Exception:
                print(f" - {os.path.basename(f)}")
        print("\n[done: list_only mode, no plots generated]\n")
        return

    if not args.list_only and not args.ckpt:
        raise SystemExit("--ckpt is required unless --list_only is set")

    def build_and_load_model(hfeat, hhorizon):
        m = build_model(args.model, hfeat, hhorizon)
        state = torch.load(args.ckpt, map_location="cpu")
        try:
            m.load_state_dict(state, strict=True)
        except Exception as e:
            print(f"[warn] strict load failed: {e}\n[info] retrying strict=False")
            m.load_state_dict(state, strict=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m.to(device).eval()
        return m, device

    if mode == "multi":
        rng = np.random.default_rng(args.seed)
        n_select = min(len(files), args.max_plots) if args.max_plots else len(files)
        if len(files) > n_select:
            idx = rng.choice(len(files), size=n_select, replace=False)
            files_to_eval = [files[i] for i in sorted(idx)]
        else:
            files_to_eval = files[:n_select]

    # File-based modes
    if mode != "multi":
        if not files:
            print("[warn] no .pkl files found in split_dir")
            return

        if mode == "batch_all":
            files_to_eval = files[:args.max_plots] if args.max_plots else files
        else:  # single
            tid = int(args.trip_id or 0)
            files_to_eval = [os.path.join(args.split_dir, f"{int(args.mmsi)}_{tid}_processed.pkl")]

    # Common evaluation for 'multi' (selected subset) and 'batch_all'/'single'
    if mode == "multi":
        pass  # files_to_eval prepared above
    elif mode in ("batch_all", "single"):
        pass
    else:
        files_to_eval = []

    if not files_to_eval:
        print("[warn] no .pkl files found in split_dir")
        return

    print(f"[mode] {mode}  found={len(files)}  selected={len(files_to_eval)}  max_plots={args.max_plots}")

    if args.list_only:
        print("\n[files selected for evaluation]\n")
        for f in files_to_eval:
            try:
                m, t = parse_trip(f)
                print(f" - {os.path.basename(f)}  (MMSI={m}, trip_id={t})")
            except Exception:
                print(f" - {os.path.basename(f)}")
        print("\n[done: list_only mode, no plots generated]\n")
        return

    # Build model for file-based inference (use generic horizon if not specified)
    feat_dim = 4; horizon = args.cap_future if args.cap_future else 128
    model, device = build_and_load_model(feat_dim, horizon)

    print(f"[info] Evaluating {len(files_to_eval)} trajectories from {args.split_dir}")
    ok = 0; skipped = 0
    for i, f in enumerate(files_to_eval):
        try:
            trip = load_trip(f, min_points=args.min_points)
            print(f"[info] {i+1}/{len(files_to_eval)} {os.path.basename(f)}: shape={trip.shape}")
            if trip.shape[1] < 4:
                raise ValueError(f"not enough features D={trip.shape[1]}; need at least 4 [lat,lon,sog,cog]")
            if np.isnan(trip).any():
                raise ValueError("contains NaNs")
            evaluate_and_plot_trip(f, trip, model, args, Path(args.out_dir), basemap=None, proj=None, sample_idx=i)
            ok += 1
            print(f"[ok] {os.path.basename(f)} plotted ({len(trip)} pts)")
        except Exception as e:
            skipped += 1
            if args.log_skip_reasons:
                print(f"[skip] {os.path.basename(f)} reason={e}")
    print(f"[summary] plotted={ok}, skipped={skipped}, total={len(files_to_eval)}")

if __name__ == "__main__":
    main()
