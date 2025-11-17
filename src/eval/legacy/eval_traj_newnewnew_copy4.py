# src/eval/eval_traj_newnewnew.py
from __future__ import annotations
import argparse, os, glob, pickle, csv, datetime as dt
from pathlib import Path
from typing import Optional, Tuple, Iterable, Sequence
import numpy as np
import torch
import matplotlib.pyplot as plt

# Model defs (TPTrans/GRU) — trained to output ABSOLUTE normalized [lat, lon]
from ..models import GRUSeq2Seq, TPTrans

# Try using your preprocessing scaler (preferred)
_HAS_DENORM_FN = False
try:
    from ..preprocessing.preprocessing import de_normalize_track as _de_normalize_track  # [lat,lon,sog,cog] -> degrees
    _HAS_DENORM_FN = True
except Exception:
    pass

# Default hard bounds (only used if de_normalize_track is unavailable)
DEFAULT_DENMARK_EXTENT: Tuple[float, float, float, float] = (6.0, 16.0, 54.0, 58.0)
_DEFAULT_LAT_MIN = _DEFAULT_LAT_MAX = _DEFAULT_LON_MIN = _DEFAULT_LON_MAX = None
try:
    from ..preprocessing.preprocessing import LAT_MIN as _LAT_MIN, LAT_MAX as _LAT_MAX, LON_MIN as _LON_MIN, LON_MAX as _LON_MAX  # type: ignore
    _DEFAULT_LAT_MIN, _DEFAULT_LAT_MAX = float(_LAT_MIN), float(_LAT_MAX)
    _DEFAULT_LON_MIN, _DEFAULT_LON_MAX = float(_LON_MIN), float(_LON_MAX)
except Exception:
    pass

# ---------- Helpers ----------

def parse_trip(fname: str) -> Tuple[int, int]:
    base = os.path.basename(fname).replace("_processed.pkl", "")
    mmsi_str, trip_id_str = base.split("_", 1)
    return int(mmsi_str), int(trip_id_str)

def to_iso(ts: float, fmt: str = "%Y-%m-%d %H:%M:%S UTC") -> str:
    return dt.datetime.fromtimestamp(float(ts), dt.timezone.utc).strftime(fmt)

def fname_ts(ts: float) -> str:
    return dt.datetime.fromtimestamp(float(ts), dt.timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")

def load_trip(path: str, min_points: int = 30) -> np.ndarray:
    with open(path, "rb") as f:
        data = pickle.load(f)
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

def robust_extent(lats: np.ndarray, lons: np.ndarray, pad: float = 0.75,
                  clamp: Tuple[float, float, float, float] = DEFAULT_DENMARK_EXTENT,
                  sigma: float = 3.0) -> Tuple[float, float, float, float]:
    lats = lats[np.isfinite(lats)]
    lons = lons[np.isfinite(lons)]
    if lats.size == 0 or lons.size == 0:
        return clamp
    def clip(arr):
        m = float(np.nanmean(arr)); s = float(np.nanstd(arr))
        if not np.isfinite(s) or s == 0.0: return arr
        return arr[(arr >= m - sigma*s) & (arr <= m + sigma*s)]
    lats_c = clip(lats); lons_c = clip(lons)
    if lats_c.size >= 2 and lons_c.size >= 2:
        lat_min, lat_max = float(np.min(lats_c)), float(np.max(lats_c))
        lon_min, lon_max = float(np.min(lons_c)), float(np.max(lons_c))
    else:
        lat_min, lat_max = float(np.min(lats)), float(np.max(lats))
        lon_min, lon_max = float(np.min(lons)), float(np.max(lons))
    if abs(lat_max-lat_min) < 0.2: lat_min -= 0.5; lat_max += 0.5
    if abs(lon_max-lon_min) < 0.2: lon_min -= 0.5; lon_max += 0.5
    lat_min -= pad; lat_max += pad; lon_min -= pad; lon_max += pad
    lon_min = max(clamp[0], lon_min); lon_max = min(clamp[1], lon_max)
    lat_min = max(clamp[2], lat_min); lat_max = min(clamp[3], lat_max)
    return (lon_min, lon_max, lat_min, lat_max)

def maybe_denorm_latlon(lat: np.ndarray, lon: np.ndarray,
                        lat_min: Optional[float], lat_max: Optional[float],
                        lon_min: Optional[float], lon_max: Optional[float]) -> Tuple[np.ndarray,np.ndarray]:
    """De-normalize [0..1] lat/lon to degrees using preprocessing scaler if available; otherwise linear scale by bounds."""
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    looks_norm = (
        np.nanmin(lat) >= -0.1 and np.nanmax(lat) <= 1.1 and
        np.nanmin(lon) >= -0.1 and np.nanmax(lon) <= 1.1
    )
    if looks_norm and _HAS_DENORM_FN:
        tmp = np.zeros((len(lat), 4), float)
        tmp[:,0] = lat; tmp[:,1] = lon
        tmp = _de_normalize_track(tmp)
        return tmp[:,0], tmp[:,1]
    if looks_norm:
        if None in (lat_min, lat_max, lon_min, lon_max):
            raise ValueError("Need --lat_min/--lat_max/--lon_min/--lon_max when de_normalize_track is unavailable.")
        lat_deg = lat*(lat_max-lat_min) + lat_min
        lon_deg = lon*(lon_max-lon_min) + lon_min
        return lat_deg, lon_deg
    return lat, lon

def build_model(kind: str, feat_dim: int, horizon: int, d_model=192, nhead=4, enc_layers=4, dec_layers=2):
    if kind == "gru":
        return GRUSeq2Seq(feat_dim, d_model=d_model, layers=2, horizon=horizon)
    return TPTrans(feat_dim=feat_dim, d_model=d_model, nhead=nhead,
                   enc_layers=enc_layers, dec_layers=dec_layers, horizon=horizon)

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1 = np.radians([lat1, lon1]); p2 = np.radians([lat2, lon2])
    dlat = p2[0]-p1[0]; dlon = p2[1]-p1[1]
    a = np.sin(dlat/2.0)**2 + np.cos(p1[0])*np.cos(p2[0])*np.sin(dlon/2.0)**2
    return float(2*R*np.arcsin(np.sqrt(a)))

# ---------- Core plotting ----------

def evaluate_and_plot_trip(
    fpath: str,
    trip: np.ndarray,
    model,
    args,
    out_root: Path,
    sample_idx: int,
):
    # Columns: [lat, lon, sog, cog, heading, rot, nav, timestamp, mmsi]
    mmsi, tid = parse_trip(fpath)
    n_total = len(trip)

    # Decide past / future by percentage (enforce blue+green == full trip)
    if args.pred_cut is None:
        raise ValueError("--pred_cut is required to define blue/green split as a percent of full trip.")
    past, fut_true, cut = split_by_percent(trip, args.pred_cut)

    # Total future steps = full tail unless user caps
    total_future = len(fut_true)
    if args.cap_future is not None:
        total_future = min(total_future, int(args.cap_future))
    if total_future < 1:
        raise ValueError("No future steps to predict after cap.")

    # Prepare degrees for plotting: full trip, past, full true future (green)
    # (We denorm columns 0=lat, 1=lon — inputs are either already degrees or [0..1])
    full_lat_deg, full_lon_deg = maybe_denorm_latlon(trip[:,0], trip[:,1],
                                                     args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    lats_past = full_lat_deg[:cut]; lons_past = full_lon_deg[:cut]
    lats_true_all = full_lat_deg[cut:]; lons_true_all = full_lon_deg[cut:]
    # The green segment we will display (may be capped)
    lats_true = lats_true_all[:total_future]; lons_true = lons_true_all[:total_future]

    # Current position = last past sample
    cur_lat = float(lats_past[-1]); cur_lon = float(lons_past[-1])

    # ---- Iterative rollout to match the full tail length ----
    # Build normalized input sequence [lat,lon,sog,cog] for the model
    seq_in = past[:, :4].astype(np.float32)

    # If seq_in lat/lon look like degrees, normalize with bounds (or preprocessing SPEED_MAX, if used)
    def looks_norm_latlon(lat_col, lon_col):
        return (np.nanmin(lat_col) >= -0.05 and np.nanmax(lat_col) <= 1.2 and
                np.nanmin(lon_col) >= -0.05 and np.nanmax(lon_col) <= 1.2)
    seq_norm = seq_in.copy()
    if not looks_norm_latlon(seq_in[:,0], seq_in[:,1]):
        if None in (args.lat_min, args.lat_max, args.lon_min, args.lon_max) and not _HAS_DENORM_FN:
            raise ValueError("Inputs appear to be in degrees; please provide --lat_min/--lat_max/--lon_min/--lon_max or ensure preprocessing provides de_normalize_track.")
        # Convert degrees -> normalized [0..1]
        lat_deg, lon_deg = seq_in[:,0], seq_in[:,1]
        if _HAS_DENORM_FN and (_DEFAULT_LAT_MIN is None or _DEFAULT_LON_MIN is None):
            # Fallback: infer from de_normalize_track inverse is unavailable; use CLI bounds
            pass
        seq_norm[:,0] = (lat_deg - args.lat_min) / float(args.lat_max - args.lat_min)
        seq_norm[:,1] = (lon_deg - args.lon_min) / float(args.lon_max - args.lon_min)
        try:
            from ..preprocessing.preprocessing import SPEED_MAX
            speed_max = float(SPEED_MAX)
        except Exception:
            speed_max = 30.0
        seq_norm[:,2] = np.clip(seq_in[:,2] / speed_max, 0.0, 1.0)
        seq_norm[:,3] = (seq_in[:,3] % 360.0) / 360.0

    # Roll out until we cover *total_future* (chunked by model.horizon)
    remaining = int(total_future)
    pred_lat_list: list[float] = []
    pred_lon_list: list[float] = []
    device = next(model.parameters()).device

    while remaining > 0:
        Tin = min(args.past_len, len(seq_norm))
        X_in = seq_norm[-Tin:, :][None, ...]          # [1, Tin, 4]
        with torch.no_grad():
            xb = torch.from_numpy(X_in).to(device)
            yraw = model(xb)[0].cpu().numpy()        # [H,2] ABS normalized (lat,lon)

        # how many new steps to keep in this chunk
        keep = min(yraw.shape[0], remaining)
        lat_norm = np.clip(yraw[:keep, 0], 0.0, 1.0)
        lon_norm = np.clip(yraw[:keep, 1], 0.0, 1.0)

        # Convert this chunk to degrees
        lat_deg, lon_deg = maybe_denorm_latlon(lat_norm, lon_norm,
                                               args.lat_min, args.lat_max, args.lon_min, args.lon_max)

        # Hard continuity: shift first predicted point to start at the current position
        if len(pred_lat_list) == 0 and keep > 0 and np.isfinite(lat_deg[0]) and np.isfinite(lon_deg[0]):
            dlat0 = cur_lat - float(lat_deg[0]); dlon0 = cur_lon - float(lon_deg[0])
            lat_deg = lat_deg + dlat0; lon_deg = lon_deg + dlon0
            # guardrail (just in case)
            try:
                d0 = haversine_km(cur_lat, cur_lon, float(lat_deg[0]), float(lon_deg[0]))
                if d0 > 0.5:
                    print(f"[warn] first_pred still far after anchoring: {d0:.2f} km")
            except Exception:
                pass

        pred_lat_list.extend(lat_deg.tolist())
        pred_lon_list.extend(lon_deg.tolist())

        # Update for next iteration (feed predicted chunk back as normalized features)
        # Convert lat/lon degrees -> normalized [0..1] to append
        lat_n = (lat_deg - args.lat_min) / float(args.lat_max - args.lat_min)
        lon_n = (lon_deg - args.lon_min) / float(args.lon_max - args.lon_min)
        last_sog = seq_norm[-1,2] if seq_norm.shape[1] > 2 else 0.0
        last_cog = seq_norm[-1,3] if seq_norm.shape[1] > 3 else 0.0
        add_feats = np.stack([lat_n, lon_n,
                              np.full_like(lat_n, last_sog, dtype=np.float32),
                              np.full_like(lon_n, last_cog, dtype=np.float32)], axis=1).astype(np.float32)
        seq_norm = np.vstack([seq_norm, add_feats])

        # Advance state
        cur_lat = float(pred_lat_list[-1]); cur_lon = float(pred_lon_list[-1])
        remaining -= keep

        if not args.iter_rollout:
            break

    pred_lat = np.asarray(pred_lat_list, float)
    pred_lon = np.asarray(pred_lon_list, float)

    # ----- Map extent -----
    if args.auto_extent:
        ext = robust_extent(full_lat_deg, full_lon_deg, sigma=args.extent_outlier_sigma)
    else:
        ext = DEFAULT_DENMARK_EXTENT

    # ----- Plot -----
    HAS_CARTOPY = False
    try:
        import cartopy.crs as ccrs, cartopy.feature as cfeature
        HAS_CARTOPY = True
        proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=(9, 6), subplot_kw={'projection': proj})
        ax.add_feature(cfeature.OCEAN, zorder=0)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=2)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
        try: gl.top_labels = False; gl.right_labels = False
        except Exception: pass
        ax.set_extent(ext, crs=proj)
        # Full trip context (light gray)
        ax.plot(full_lon_deg, full_lat_deg, color="#999999", linewidth=1.0, alpha=0.3, transform=proj, label="full trip (context)", zorder=1)
        # Past (blue), current (black), True future (green), Pred future (red)
        ax.plot(lons_past, lats_past, '-', color="#1f77b4", linewidth=1.8, transform=proj, label="past (input)", zorder=3)
        ax.plot([cur_lon], [cur_lat], 'o', color='k', markersize=4, transform=proj, label='current pos', zorder=4)
        ax.plot(lons_true, lats_true, '-', color="#2ca02c", linewidth=1.8, transform=proj, label="true future", zorder=4)
        # Ensure the red starts at the current point visually (prepend the current point)
        pred_lon_plot = np.concatenate([[cur_lon], pred_lon]) if len(pred_lon) else np.array([cur_lon])
        pred_lat_plot = np.concatenate([[cur_lat], pred_lat]) if len(pred_lat) else np.array([cur_lat])
        ax.plot(pred_lon_plot, pred_lat_plot, '--', color="#d62728", linewidth=2.0, transform=proj, label="pred future", zorder=5)
    except Exception:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.plot(full_lon_deg, full_lat_deg, color="#999999", linewidth=1.0, alpha=0.3, label="full trip (context)", zorder=1)
        ax.plot(lons_past, lats_past, '-', color="#1f77b4", linewidth=1.8, label="past (input)", zorder=3)
        ax.plot([cur_lon], [cur_lat], 'o', color='k', markersize=4, label='current pos', zorder=4)
        ax.plot(lons_true, lats_true, '-', color="#2ca02c", linewidth=1.8, label="true future", zorder=4)
        pred_lon_plot = np.concatenate([[cur_lon], pred_lon]) if len(pred_lon) else np.array([cur_lon])
        pred_lat_plot = np.concatenate([[cur_lat], pred_lat]) if len(pred_lat) else np.array([cur_lat])
        ax.plot(pred_lon_plot, pred_lat_plot, '--', color="#d62728", linewidth=2.0, label="pred future", zorder=5)

    # Title / labels
    t0 = float(np.nanmin(trip[:,7])); t1 = float(np.nanmax(trip[:,7]))
    t0_iso = to_iso(t0, fmt=args.timefmt); t1_iso = to_iso(t1, fmt=args.timefmt)
    ax.set_title(f"Trajectory ({args.model}) — MMSI {mmsi} — {t0_iso} → {t1_iso}")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.legend(loc="upper left", frameon=True)

    # Output path (per-MMSI subdir)
    out_dir = Path(args.out_dir) / str(mmsi) if args.output_per_mmsi_subdir else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"traj_{args.model}_mmsi-{mmsi}_trip-{tid}_cut-{args.pred_cut}_idx-{sample_idx}.png"
    fig_path = out_dir / out_name
    plt.savefig(fig_path, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"[ok] saved {fig_path}")

    # Optional debug NPZ dump
    if args.debug_save_npz:
        dbg_dir = out_dir / 'debug_npz'; dbg_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dbg_dir / f"debug_{mmsi}_{tid}_idx-{sample_idx}.npz",
            full_lat=full_lat_deg, full_lon=full_lon_deg,
            past_lat=lats_past, past_lon=lons_past,
            true_lat=lats_true, true_lon=lons_true,
            pred_lat=pred_lat, pred_lon=pred_lon,
            cur_lat=cur_lat, cur_lon=cur_lon,
            extent=np.asarray(ext),
        )

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Evaluate trajectory forecasts with consistent de/normalization and full-trip plotting.")
    ap.add_argument("--split_dir", required=True, help="Directory with *_processed.pkl trips (e.g., data/map_reduced/val)")
    ap.add_argument("--ckpt", required=True, help="Path to model checkpoint (.pt)")
    ap.add_argument("--model", choices=["gru","tptrans"], default="tptrans")
    ap.add_argument("--horizon", type=int, default=12, help="Model's training horizon (chunk size for rollout) — must match training config.")
    ap.add_argument("--past_len", type=int, default=64, help="Past window length fed to the model at each rollout step (<= training window).")
    ap.add_argument("--pred_cut", type=float, required=True, help="Percent of the full trip used as past (blue). Remainder is true future (green).")
    ap.add_argument("--cap_future", type=int, default=None, help="Optional cap on future steps (otherwise use the full tail).")
    ap.add_argument("--out_dir", default="data/figures", help="Output directory root.")
    ap.add_argument("--mmsi", type=str, default="all",
                    help="MMSI selection: 'all', a single ID (e.g., 219000111), or a comma-separated list (e.g., 219000111,219000222).")
    ap.add_argument("--trip_id", type=int, default=None, help="When MMSI is a single numeric ID, choose the specific trip id (default=0).")
    ap.add_argument("--max_plots", type=int, default=None, help="Limit number of plots in 'all' mode (optional).")
    ap.add_argument("--iter_rollout", action="store_true", default=True, help="Iteratively roll out to match full tail length.")
    ap.add_argument("--output_per_mmsi_subdir", action="store_true", default=True, help="Save each figure under out_dir/<MMSI>/")
    ap.add_argument("--auto_extent", action="store_true", help="Zoom to track (clamped to Denmark). If not set, map is locked to Denmark.")
    ap.add_argument("--extent_outlier_sigma", type=float, default=3.0, help="Sigma for robust auto-zoom.")
    ap.add_argument("--denorm", action="store_true", help="No-op retained for compatibility; de/normalization is now consistent and automatic.")
    ap.add_argument("--lat_min", type=float, default=_DEFAULT_LAT_MIN, help="Lat min bound (only used if de_normalize_track is unavailable).")
    ap.add_argument("--lat_max", type=float, default=_DEFAULT_LAT_MAX, help="Lat max bound (only used if de_normalize_track is unavailable).")
    ap.add_argument("--lon_min", type=float, default=_DEFAULT_LON_MIN, help="Lon min bound (only used if de_normalize_track is unavailable).")
    ap.add_argument("--lon_max", type=float, default=_DEFAULT_LON_MAX, help="Lon max bound (only used if de_normalize_track is unavailable).")
    ap.add_argument("--timefmt", type=str, default="%Y-%m-%d %H:%M:%S UTC")
    ap.add_argument("--debug_save_npz", action="store_true")
    ap.add_argument("--min_points", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Build model with the *training* horizon (needed for matching checkpoint head)
    feat_dim = 4
    model = build_model(args.model, feat_dim, args.horizon)
    state = torch.load(args.ckpt, map_location="cpu")
    try:
        model.load_state_dict(state, strict=True)
    except Exception as e:
        print(f"[warn] strict load failed: {e} — retrying strict=False")
        model.load_state_dict(state, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # Collect files according to selection
    files = sorted(glob.glob(os.path.join(args.split_dir, "*.pkl")))
    if not files:
        raise SystemExit(f"No trips found in {args.split_dir}")

    rng = np.random.default_rng(args.seed)

    def select_files_all():
        if args.max_plots is None:
            return files
        if len(files) <= args.max_plots:
            return files
        idx = rng.choice(len(files), size=args.max_plots, replace=False)
        return [files[i] for i in sorted(idx)]

    selected: list[str] = []
    if args.mmsi.lower() == "all":
        selected = select_files_all()
    else:
        # single or comma-separated
        id_strs = [s.strip() for s in args.mmsi.split(",") if s.strip()]
        id_set = set(int(s) for s in id_strs)
        # If a single MMSI + trip_id specified, use exactly that
        if len(id_set) == 1 and args.trip_id is not None:
            only = f"{list(id_set)[0]}_{int(args.trip_id)}_processed.pkl"
            f = os.path.join(args.split_dir, only)
            selected = [f]
        else:
            # Collect all trips belonging to any of the MMSIs
            for f in files:
                try:
                    m, _t = parse_trip(f)
                    if m in id_set:
                        selected.append(f)
                except Exception:
                    pass
            if args.max_plots is not None and len(selected) > args.max_plots:
                idx = rng.choice(len(selected), size=args.max_plots, replace=False)
                selected = [selected[i] for i in sorted(idx)]

    print(f"[select] total={len(files)} selected={len(selected)} mode={'all' if args.mmsi=='all' else args.mmsi}")
    ok = 0; skipped = 0
    for i, f in enumerate(selected):
        try:
            trip = load_trip(f, min_points=args.min_points)
            if trip.shape[1] < 4:
                raise ValueError(f"Trip has D={trip.shape[1]}; need at least 4 [lat,lon,sog,cog].")
            evaluate_and_plot_trip(f, trip, model, args, Path(args.out_dir), sample_idx=i)
            ok += 1
        except Exception as e:
            skipped += 1
            print(f"[skip] {os.path.basename(f)}: {e}")
    print(f"[summary] plotted={ok} skipped={skipped} total_selected={len(selected)}")

if __name__ == "__main__":
    main()
