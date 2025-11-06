# src/eval/eval_traj.py
from __future__ import annotations
import argparse, os, glob, pickle, csv, datetime as dt, traceback
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt


from ..models import GRUSeq2Seq, TPTrans

# Prefer preprocessing scaler if available
_HAS_DENORM_FN = False
try:
    from ..preprocessing.preprocessing import de_normalize_track as _de_normalize_track
    _HAS_DENORM_FN = True
except Exception:
    pass

# Denmark extent by default
DEFAULT_DK_EXTENT: Tuple[float, float, float, float] = (6.0, 16.0, 54.0, 58.0)

# Bounds (used if de_normalize_track is unavailable)
_DEFAULT_LAT_MIN = _DEFAULT_LAT_MAX = _DEFAULT_LON_MIN = _DEFAULT_LON_MAX = None
try:
    from ..preprocessing.preprocessing import LAT_MIN as _LAT_MIN, LAT_MAX as _LAT_MAX, LON_MIN as _LON_MIN, LON_MAX as _LON_MAX
    _DEFAULT_LAT_MIN, _DEFAULT_LAT_MAX = float(_LAT_MIN), float(_LAT_MAX)
    _DEFAULT_LON_MIN, _DEFAULT_LON_MAX = float(_LON_MIN), float(_LON_MAX)
except Exception:
    pass


def parse_trip(fname: str) -> Tuple[int, int]:
    base = os.path.basename(fname).replace("_processed.pkl", "")
    mmsi_str, trip_id_str = base.split("_", 1)
    return int(mmsi_str), int(trip_id_str)

def to_iso(ts: float, fmt: str = "%Y-%m-%d %H:%M:%S UTC") -> str:
    return dt.datetime.fromtimestamp(float(ts), dt.timezone.utc).strftime(fmt)

def load_trip(path: str, min_points: int = 30) -> np.ndarray:
    with open(path, "rb") as f:
        data = pickle.load(f)
    trip = data["traj"] if isinstance(data, dict) and "traj" in data else np.asarray(data)
    trip = np.asarray(trip)
    if len(trip) < int(min_points):
        raise ValueError(f"too short: {len(trip)} points")
    ts = trip[:, 7]
    if not np.all(ts[:-1] <= ts[1:]):
        trip = trip[np.argsort(ts)]
    return trip

def split_by_percent(trip: np.ndarray, pct: float) -> Tuple[np.ndarray, np.ndarray, int]:
    n = len(trip)
    cut = max(1, min(n - 2, int(round(n * pct / 100.0))))
    past = trip[:cut]
    future_true = trip[cut:]
    return past, future_true, cut

def robust_extent(lats: np.ndarray, lons: np.ndarray, pad: float = 0.75,
                  clamp: Tuple[float, float, float, float] = DEFAULT_DK_EXTENT,
                  sigma: float = 3.0) -> Tuple[float, float, float, float]:
    lats = lats[np.isfinite(lats)]; lons = lons[np.isfinite(lons)]
    if lats.size == 0 or lons.size == 0: return clamp
    def clip(arr):
        m = float(np.nanmean(arr)); s = float(np.nanstd(arr))
        if not np.isfinite(s) or s == 0.0: return arr
        return arr[(arr >= m - sigma*s) & (arr <= m + sigma*s)]
    lats_c, lons_c = clip(lats), clip(lons)
    if lats_c.size >= 2 and lons_c.size >= 2:
        lat_min, lat_max = float(np.min(lats_c)), float(np.max(lats_c))
        lon_min, lon_max = float(np.min(lons_c)), float(np.max(lons_c))
    else:
        lat_min, lat_max = float(np.min(lats)), float(np.max(lats))
        lon_min, lon_max = float(np.min(lons)), float(np.max(lons))
    if abs(lat_max - lat_min) < 0.2: lat_min -= 0.5; lat_max += 0.5
    if abs(lon_max - lon_min) < 0.2: lon_min -= 0.5; lon_max += 0.5
    lat_min -= pad; lat_max += pad; lon_min -= pad; lon_max += pad
    lon_min = max(clamp[0], lon_min); lon_max = min(clamp[1], lon_max)
    lat_min = max(clamp[2], lat_min); lat_max = min(clamp[3], lat_max)
    return (lon_min, lon_max, lat_min, lat_max)

def maybe_denorm_latlon(lat: np.ndarray, lon: np.ndarray,
                        lat_min: Optional[float], lat_max: Optional[float],
                        lon_min: Optional[float], lon_max: Optional[float]) -> Tuple[np.ndarray,np.ndarray]:
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    looks_norm = (np.nanmin(lat) >= -0.1 and np.nanmax(lat) <= 1.1 and
                  np.nanmin(lon) >= -0.1 and np.nanmax(lon) <= 1.1)
    if looks_norm and _HAS_DENORM_FN:
        tmp = np.zeros((len(lat), 4), float); tmp[:,0] = lat; tmp[:,1] = lon
        tmp = _de_normalize_track(tmp); return tmp[:,0], tmp[:,1]
    if looks_norm:
        if None in (lat_min, lat_max, lon_min, lon_max):
            raise ValueError("Provide --lat_min/--lat_max/--lon_min/--lon_max or implement de_normalize_track.")
        lat_deg = lat*(lat_max-lat_min) + lat_min
        lon_deg = lon*(lon_max-lon_min) + lon_min
        return lat_deg, lon_deg
    return lat, lon

def build_model(kind: str, feat_dim: int, horizon: int, d_model=192, nhead=4, enc_layers=4, dec_layers=2):
    if kind == "gru":
        return GRUSeq2Seq(feat_dim, d_model=d_model, layers=2, horizon=horizon)
    return TPTrans(feat_dim=feat_dim, d_model=d_model, nhead=nhead, enc_layers=enc_layers, dec_layers=dec_layers, horizon=horizon)

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1 = np.radians([lat1, lon1]); p2 = np.radians([lat2, lon2])
    dlat = p2[0]-p1[0]; dlon = p2[1]-p1[1]
    a = np.sin(dlat/2.0)**2 + np.cos(p1[0])*np.cos(p2[0])*np.sin(dlon/2.0)**2
    return float(2*R*np.arcsin(np.sqrt(a)))

def cumdist(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    if len(lat) == 0: return np.array([0.0])
    cd = [0.0]
    for i in range(1, len(lat)):
        cd.append(cd[-1] + haversine_km(lat[i-1], lon[i-1], lat[i], lon[i]))
    return np.asarray(cd, float)

def evaluate_and_plot_trip(
    fpath: str,
    trip: np.ndarray,
    model,
    args,
    sample_idx: int,
):
    """One trip end-to-end."""
    # Trip metadata
    mmsi, tid = parse_trip(fpath)
    n_total = len(trip)

    # --- Split by percent ---
    past, future_true_all, cut = split_by_percent(trip, args.pred_cut)
    n_past = cut
    n_fut_raw = len(future_true_all)
    if n_past < 2 or n_fut_raw < 2:
        raise ValueError(f"too short after split (total={n_total}, past={n_past}, future={n_fut_raw})")

    # Optional future cap
    N_future = n_fut_raw if args.cap_future is None else min(n_fut_raw, int(args.cap_future))
    if N_future < 1:
        raise ValueError("No future steps to predict after cap.")

    # --- Denorm full lat/lon for plotting ---
    full_lat_deg, full_lon_deg = maybe_denorm_latlon(trip[:,0], trip[:,1],
                                                     args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    lats_past = full_lat_deg[:cut]; lons_past = full_lon_deg[:cut]
    cur_lat = float(lats_past[-1]); cur_lon = float(lons_past[-1])




    #lats_true_all = full_lat_deg[cut:]; lons_true_all = full_lon_deg[cut:]
    #lats_true = lats_true_all[:N_future]; lons_true = lons_true_all[:N_future]


    # full tail for plotting
    lats_true_all = full_lat_deg[cut:]
    lons_true_all = full_lon_deg[cut:]

    # target portion for eval/prediction length (cap if requested)
    lats_true_eval = lats_true_all[:N_future]
    lons_true_eval = lons_true_all[:N_future]





    # --- Iterative rollout to cover N_future ---
    seq_in = past[:, :4].astype(np.float32)  # [lat,lon,sog,cog]
    def looks_norm(x): return (np.nanmin(x) >= -0.05 and np.nanmax(x) <= 1.2)
    seq_norm = seq_in.copy()
    if not (looks_norm(seq_in[:,0]) and looks_norm(seq_in[:,1])):
        if None in (args.lat_min, args.lat_max, args.lon_min, args.lon_max) and not _HAS_DENORM_FN:
            raise ValueError("Inputs appear degrees; provide bounds or implement de_normalize_track.")
        # degrees -> [0..1]
        seq_norm[:,0] = (seq_in[:,0] - args.lat_min) / float(args.lat_max - args.lat_min)
        seq_norm[:,1] = (seq_in[:,1] - args.lon_min) / float(args.lon_max - args.lon_min)
        try:
            from ..preprocessing.preprocessing import SPEED_MAX
            speed_max = float(SPEED_MAX)
        except Exception:
            speed_max = 30.0
        seq_norm[:,2] = np.clip(seq_in[:,2] / speed_max, 0.0, 1.0)
        seq_norm[:,3] = (seq_in[:,3] % 360.0) / 360.0

    remaining = int(N_future)
    pred_lat_list, pred_lon_list = [], []
    device = next(model.parameters()).device

    while remaining > 0:
        Tin = min(args.past_len, len(seq_norm))
        X_in = seq_norm[-Tin:, :][None, ...]
        with torch.no_grad():
            yraw = model(torch.from_numpy(X_in).to(device))[0].cpu().numpy()  # [H,2] normalized abs

        keep = min(yraw.shape[0], remaining)
        lat_n = np.clip(yraw[:keep, 0], 0.0, 1.0)
        lon_n = np.clip(yraw[:keep, 1], 0.0, 1.0)
        lat_deg, lon_deg = maybe_denorm_latlon(lat_n, lon_n, args.lat_min, args.lat_max, args.lon_min, args.lon_max)

        # Anchor first chunk to current pos for continuity
        if len(pred_lat_list) == 0 and keep > 0 and np.isfinite(lat_deg[0]) and np.isfinite(lon_deg[0]):
            dlat0 = cur_lat - float(lat_deg[0]); dlon0 = cur_lon - float(lon_deg[0])
            lat_deg = lat_deg + dlat0; lon_deg = lon_deg + dlon0

        pred_lat_list.extend(lat_deg.tolist())
        pred_lon_list.extend(lon_deg.tolist())

        # feedback normalized
        lat_n2 = (lat_deg - args.lat_min) / float(args.lat_max - args.lat_min)
        lon_n2 = (lon_deg - args.lon_min) / float(args.lon_max - args.lon_min)
        last_sog = seq_norm[-1,2] if seq_norm.shape[1] > 2 else 0.0
        last_cog = seq_norm[-1,3] if seq_norm.shape[1] > 3 else 0.0
        add_feats = np.stack([lat_n2, lon_n2,
                              np.full_like(lat_n2, last_sog, dtype=np.float32),
                              np.full_like(lon_n2, last_cog, dtype=np.float32)], axis=1).astype(np.float32)
        seq_norm = np.vstack([seq_norm, add_feats])

        remaining -= keep
        if not args.iter_rollout:
            break

    pred_lat = np.asarray(pred_lat_list, float)
    pred_lon = np.asarray(pred_lon_list, float)

    # --- Enforce equal number of points ---
    N_true = len(lats_true_eval)
    if len(pred_lat) > N_true:
        pred_lat, pred_lon = pred_lat[:N_true], pred_lon[:N_true]
    elif len(pred_lat) < N_true:
        if len(pred_lat) == 0:
            pred_lat = np.full(N_true, cur_lat); pred_lon = np.full(N_true, cur_lon)
        else:
            pred_lat = np.concatenate([pred_lat, np.full(N_true-len(pred_lat), pred_lat[-1])])
            pred_lon = np.concatenate([pred_lon, np.full(N_true-len(pred_lon), pred_lon[-1])])

    # --- OPTIONAL: distance-match red to green (same km), default OFF ---
    if args.match_distance:
        green_cd = cumdist(lats_true_eval, lons_true_eval)
        green_total = float(green_cd[-1]) if len(green_cd) else 0.0

        pred_cd = cumdist(np.r_[cur_lat, pred_lat], np.r_[cur_lon, pred_lon])  # includes connector
        pred_cd = pred_cd[1:] if len(pred_cd) > 0 else pred_cd

        if len(pred_cd):
            pred_total = float(pred_cd[-1])

            if pred_total > green_total + 1e-6:
                # TRIM (prediction longer than green) — optionally with interpolation
                cut_idx = int(np.searchsorted(pred_cd, green_total, side="right") - 1)
                cut_idx = max(0, min(cut_idx, len(pred_cd)-1))
                # interpolate one step so the last pred point is nearer green_total
                if cut_idx < len(pred_cd) - 1:
                    frac = (green_total - pred_cd[cut_idx]) / (pred_cd[cut_idx+1] - pred_cd[cut_idx] + 1e-9)
                    lat_last = pred_lat[cut_idx] + frac * (pred_lat[cut_idx+1] - pred_lat[cut_idx])
                    lon_last = pred_lon[cut_idx] + frac * (pred_lon[cut_idx+1] - pred_lon[cut_idx])
                    pred_lat = np.concatenate([pred_lat[:cut_idx+1], [lat_last]])
                    pred_lon = np.concatenate([pred_lon[:cut_idx+1], [lon_last]])
                else:
                    pred_lat = pred_lat[:cut_idx+1]
                    pred_lon = pred_lon[:cut_idx+1]
            else:
                # PRED SHORTER than green -> DO NOT trim (avoid collapsing to 1 point)
                pass

        if len(pred_lat) == 0:
            pred_lat = np.array([cur_lat], float); pred_lon = np.array([cur_lon], float)

    # --- FINAL ALIGNMENT (points) ---
    N_true = len(lats_true_eval)
    N_pred = len(pred_lat)
    n_eff = min(N_true, N_pred)
    if n_eff < 2:
        raise ValueError(f"too short after alignment (true={N_true}, pred={N_pred})")

    # aligned arrays for metrics & CSV(pred)
    lats_true_eval = lats_true_eval[:n_eff]
    lons_true_eval = lons_true_eval[:n_eff]
    pred_lat       = pred_lat[:n_eff]
    pred_lon       = pred_lon[:n_eff]


    # --- Metrics ---
    def ade_km(tlat, tlon, plat, plon) -> float:
        n = min(len(tlat), len(plat))
        if n < 1: return float("nan")
        d = [haversine_km(tlat[i], tlon[i], plat[i], plon[i]) for i in range(n)]
        return float(np.mean(d))

    def fde_km(tlat, tlon, plat, plon) -> float:
        n = min(len(tlat), len(plat))
        if n < 1: return float("nan")
        return haversine_km(tlat[n-1], tlon[n-1], plat[n-1], plon[n-1])

    def mae_km(tlat, tlon, plat, plon) -> float:
        n = min(len(tlat), len(plat))
        if n < 1: return float("nan")
        d = [haversine_km(tlat[i], tlon[i], plat[i], plon[i]) for i in range(n)]
        return float(np.median(d))

    ade = ade_km(lats_true_eval, lons_true_eval, pred_lat, pred_lon)
    fde = fde_km(lats_true_eval, lons_true_eval, pred_lat, pred_lon)
    mae = mae_km(lats_true_eval, lons_true_eval, pred_lat, pred_lon)
    #print(f"[debug] ade={ade:.3f}km  fde={fde:.3f}km  mae={mae:.3f}km")

    # --- Extent ---
    ext = robust_extent(full_lat_deg, full_lon_deg, sigma=args.extent_outlier_sigma) if args.auto_extent else DEFAULT_DK_EXTENT

    # --- Plot ---
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
        ax.plot(full_lon_deg, full_lat_deg, color="#999999", linewidth=1.0, alpha=0.3, transform=proj, label="full trip (context)", zorder=1)
        ax.plot(lons_past, lats_past, '-', color="#1f77b4", linewidth=1.8, transform=proj, label="past (input)", zorder=3)
        ax.plot([cur_lon], [cur_lat], 'o', color='k', markersize=5, transform=proj, label='current pos', zorder=5)

        # green: full tail for plotting
        if len(lats_true_all) >= 2:
            ax.plot(lons_true_all, lats_true_all, '-', color="#2ca02c", linewidth=1.8,
                    transform=proj if HAS_CARTOPY else None, label="true future", zorder=4)
        else:
            ax.plot(lons_true_all, lats_true_all, 'o', color="#2ca02c",
                    transform=proj if HAS_CARTOPY else None, zorder=4, label="true future (pt)")

        # red: aligned prediction
        if len(pred_lat) >= 2:
            ax.plot(pred_lon, pred_lat, '--', color="#d62728", linewidth=2.0,
                    transform=proj if HAS_CARTOPY else None, label="pred future", zorder=5)
        else:
            ax.plot(pred_lon, pred_lat, 'x', color="#d62728",
                    transform=proj if HAS_CARTOPY else None, zorder=5, label="pred future (pt)")
            
        if len(pred_lon) > 0:
            ax.plot([cur_lon, pred_lon[0]], [cur_lat, pred_lat[0]], '--', color="#d62728", linewidth=2.0, transform=proj, alpha=0.7, zorder=5)
    
    except Exception:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.plot(full_lon_deg, full_lat_deg, color="#999999", linewidth=1.0, alpha=0.3, label="full trip (context)", zorder=1)
        ax.plot(lons_past, lats_past, '-', color="#1f77b4", linewidth=1.8, label="past (input)", zorder=3)
        ax.plot([cur_lon], [cur_lat], 'o', color='k', markersize=5, label='current pos', zorder=5)
        if len(lats_true) >= 2:
            ax.plot(lons_true, lats_true, '-', color="#2ca02c", linewidth=1.8, label="true future", zorder=4)
        else:
            ax.plot(lons_true, lats_true, 'o', color="#2ca02c", zorder=4, label="true future (pt)")
        if len(pred_lat) >= 2:
            ax.plot(pred_lon, pred_lat, '--', color="#d62728", linewidth=2.0, label="pred future", zorder=5)
        else:
            ax.plot(pred_lon, pred_lat, 'x', color="#d62728", zorder=5, label="pred future (pt)")
        if len(pred_lon) > 0:
            ax.plot([cur_lon, pred_lon[0]], [cur_lat, pred_lat[0]], '--', color="#d62728", linewidth=2.0, alpha=0.7, zorder=5)

    t0 = float(np.nanmin(trip[:,7])); t1 = float(np.nanmax(trip[:,7]))
    future_len_full = len(lats_true_all)

    ax.set_title(f"Trajectory ({args.model}) — MMSI {mmsi} — {to_iso(t0, args.timefmt)} → {to_iso(t1, args.timefmt)}\n"
                 f"cut={args.pred_cut}%  future={future_len_full}  ADE={ade:.3f}km  FDE={fde:.3f}km  MAE~={mae:.3f}km")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.legend(loc="upper left", frameon=True)

    # --- Outputs (per-MMSI folder) ---
    out_dir = Path(args.out_dir) / str(mmsi) if args.output_per_mmsi_subdir else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_path = out_dir / f"traj_{args.model}_mmsi-{mmsi}_trip-{tid}_cut-{args.pred_cut}_idx-{sample_idx}.png"
    plt.savefig(fig_path, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"[ok] saved {fig_path}")

    # Trip CSV (slice-safe timestamps)
    trip_csv = out_dir / f"trip_{mmsi}_{tid}_cut-{args.pred_cut}_idx-{sample_idx}.csv"
    with open(trip_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx","segment","timestamp","lat","lon"])
    
        ts_past     = trip[:cut, 7]                              # full past
        ts_true_all = trip[cut:, 7]                              # full future (plot)
        ts_true_eval= trip[cut:cut + len(lats_true_eval), 7]     # eval-aligned future
    
        # past
        for i, (ts, la, lo) in enumerate(zip(ts_past, lats_past, lons_past)):
            w.writerow([i, "past", to_iso(ts), la, lo])
    
        base_idx = len(lats_past)
    
        # true future (FULL)
        for j, (ts, la, lo) in enumerate(zip(ts_true_all, lats_true_all, lons_true_all)):
            w.writerow([base_idx + j, "true_future", to_iso(ts), la, lo])
    
        # pred future (aligned to first n_eff timestamps of the future)
        for j, (ts, la, lo) in enumerate(zip(ts_true_eval, pred_lat, pred_lon)):
            w.writerow([base_idx + j, "pred_future", to_iso(ts), la, lo])


    # Per-MMSI metrics CSV (append)
    metrics_csv = out_dir / f"metrics_{mmsi}.csv"
    new_file = not metrics_csv.exists()
    with open(metrics_csv, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["mmsi","trip_id","pred_cut_pct","n_total","n_past","n_future","ade_km","fde_km","mae_km"])
        w.writerow([mmsi, tid, args.pred_cut, n_total, n_past, N_true, f"{ade:.6f}", f"{fde:.6f}", f"{mae:.6f}"])
    # >>> ADD THIS RETURN <<<
    return {
        "mmsi": mmsi,
        "trip_id": tid,
        "pred_cut_pct": float(args.pred_cut),
        "n_total": int(n_total),
        "n_past": int(n_past),
        "n_future": int(len(lats_true_eval)),   # eval-aligned future length
        "ade_km": float(ade),
        "fde_km": float(fde),
        "mae_km": float(mae),
    }

def main():
    ap = argparse.ArgumentParser(
        "Evaluate a trained trajectory model on map-reduced trips.\n"
        "- Blue = past (first pred_cut% of points)\n"
        "- Green = full true future (from cut to the end)\n"
        "- Red = predicted future (aligned to the first N_future timestamps; "
        "optionally trimmed by distance if --match_distance)\n"
        "Outputs: per-MMSI figures, per-trip CSVs, and per-MMSI metrics CSV."
    )

    # IO & selection
    ap.add_argument("--split_dir", required=True,
        help="Directory with map-reduced *.pkl trips (e.g., data/map_reduced/val).")
    ap.add_argument("--ckpt", required=True,
        help="Path to model checkpoint (e.g., data/checkpoints/traj_tptrans.pt).")
    ap.add_argument("--out_dir", default="data/figures",
        help="Root output folder. Per-MMSI subfolders will be created here.")
    ap.add_argument("--mmsi", type=str, default="all",
        help="Which MMSI(s) to evaluate. Use 'all' or a comma list, e.g. '210046000,210174000'.")
    ap.add_argument("--trip_id", type=int, default=None,
        help="If a single MMSI is given, evaluate only this exact trip index (e.g. 0).")
    ap.add_argument("--max_plots", type=int, default=None,
        help="If set, randomly sample at most this many files (per run).")

    # Model
    ap.add_argument("--model", choices=["gru","tptrans"], default="tptrans",
        help="Model architecture to load (must match the checkpoint).")
    ap.add_argument("--horizon", type=int, default=12,
        help="Forecast chunk per forward pass. Use the same value you trained with (12 for TPTrans here).")
    ap.add_argument("--past_len", type=int, default=64,
        help="Past context length used at each rollout step. Must be <= training window (64 here).")

    # Evaluation slicing
    ap.add_argument("--pred_cut", type=float, required=True,
        help="Percent of points used as past. Example: 75 => 75%% past (blue), 25%% future (green).")
    ap.add_argument("--cap_future", type=int, default=None,
        help="Optional cap on number of future points used for evaluation. "
             "If unset, uses the entire tail (from cut to end) for plotting; "
             "metrics/prediction use the first N_future points.")
    ap.add_argument("--min_points", type=int, default=30,
        help="Skip trips with fewer than this many points BEFORE splitting (quality control).")

    # Plotting & map
    ap.add_argument("--auto_extent", action="store_true",
        help="Auto-zoom to the track (clamped to Denmark bounds). If not set, the map is locked to Denmark.")
    ap.add_argument("--extent_outlier_sigma", type=float, default=3.0,
        help="When --auto_extent, clip lat/lon outliers using mean±sigma*std before computing extents.")
    ap.add_argument("--output_per_mmsi_subdir", action="store_true", default=True,
        help="Write outputs into per-MMSI subfolders (enabled by default).")

    # Denorm / bounds
    ap.add_argument("--denorm", action="store_true",
        help="(Legacy switch) If your inputs were normalized and the code can’t auto-denorm, "
             "you must pass the bounds below.")
    ap.add_argument("--lat_min", type=float, default=_DEFAULT_LAT_MIN,
        help="Latitude min bound for [0..1] -> degrees denormalization (if needed).")
    ap.add_argument("--lat_max", type=float, default=_DEFAULT_LAT_MAX,
        help="Latitude max bound for [0..1] -> degrees denormalization (if needed).")
    ap.add_argument("--lon_min", type=float, default=_DEFAULT_LON_MIN,
        help="Longitude min bound for [0..1] -> degrees denormalization (if needed).")
    ap.add_argument("--lon_max", type=float, default=_DEFAULT_LON_MAX,
        help="Longitude max bound for [0..1] -> degrees denormalization (if needed).")

    # Behavior toggles
    ap.add_argument("--iter_rollout", action="store_true", default=True,
        help="Iteratively roll out predictions until the evaluation future window is covered (default ON).")
    ap.add_argument("--match_distance", action="store_true",
        help="Trim the predicted polyline only if it's longer (in km) than the green true tail. "
             "Keeps equal timestamps; never collapses short preds.")
    ap.add_argument("--debug_save_npz", action="store_true",
        help="Save internal arrays for debugging (per-trip npz).")
    ap.add_argument("--timefmt", type=str, default="%Y-%m-%d %H:%M:%S UTC",
        help="Timestamp format for titles/CSVs (UTC).")
    ap.add_argument("--seed", type=int, default=0,
        help="Random seed for sampling when --max_plots is used.")
    ap.add_argument("--verbose", action="store_true",
        help="Print first three full tracebacks for easier debugging.")

    args = ap.parse_args()

    # Model
    model = build_model(args.model, feat_dim=4, horizon=args.horizon)
    state = torch.load(args.ckpt, map_location="cpu")
    try:
        model.load_state_dict(state, strict=True)
    except Exception as e:
        print(f"[warn] strict load failed: {e} — retrying strict=False")
        model.load_state_dict(state, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # Files
    files = sorted(glob.glob(os.path.join(args.split_dir, "*.pkl")))
    if not files: raise SystemExit(f"No trips found in {args.split_dir}")

    # Select
    rng = np.random.default_rng(args.seed)
    def choose_all(fs):
        if args.max_plots is None or len(fs) <= args.max_plots: return fs
        idx = rng.choice(len(fs), size=args.max_plots, replace=False)
        return [fs[i] for i in sorted(idx)]

    if args.mmsi.lower() == "all":
        selected = choose_all(files)
    else:
        ids = [s.strip() for s in args.mmsi.split(",") if s.strip()]
        id_set = set(int(s) for s in ids)
        if len(id_set) == 1 and args.trip_id is not None:
            selected = [os.path.join(args.split_dir, f"{list(id_set)[0]}_{int(args.trip_id)}_processed.pkl")]
        else:
            selected = [f for f in files if parse_trip(f)[0] in id_set]
            selected = choose_all(selected)

    print(f"[select] total={len(files)} selected={len(selected)} mode={'all' if args.mmsi=='all' else args.mmsi}")

    ok = skipped = 0
    all_rows = []  # collect per-trip rows for global summary

    for i, f in enumerate(selected):
        try:
            trip = load_trip(f, min_points=args.min_points)
            if trip.shape[1] < 4:
                raise ValueError(f"Trip has D={trip.shape[1]}; need at least 4 [lat,lon,sog,cog].")
            row = evaluate_and_plot_trip(f, trip, model, args, sample_idx=i)
            all_rows.append(row)  # <<— collect
            ok += 1
        except Exception as e:
            skipped += 1
            if args.verbose and skipped <= 3:
                print(f"[skip] {os.path.basename(f)}: {e}\n{traceback.format_exc()}")
            else:
                print(f"[skip] {os.path.basename(f)}: {e}")
    print(f"[summary] plotted={ok} skipped={skipped} total_selected={len(selected)}")
    
    # ---------- Global summary: per-trip, per-MMSI, and overall ----------
    try:
        if len(all_rows) > 0:
            out_root = Path(args.out_dir)

            # 2.1 Write raw per-trip metrics
            trips_csv = out_root / "summary_trips.csv"
            with open(trips_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["mmsi","trip_id","pred_cut_pct","n_total","n_past","n_future","ade_km","fde_km","mae_km"])
                for r in all_rows:
                    w.writerow([r["mmsi"], r["trip_id"], r["pred_cut_pct"], r["n_total"], r["n_past"],
                                r["n_future"], f'{r["ade_km"]:.6f}', f'{r["fde_km"]:.6f}', f'{r["mae_km"]:.6f}'])

            # helpers
            import numpy as _np
            from collections import defaultdict as _dd

            def _stats(vals):
                x = _np.asarray(vals, float)
                x = x[_np.isfinite(x)]
                if x.size == 0:
                    return _np.nan, _np.nan, _np.nan, _np.nan
                return (float(x.mean()),
                        float(_np.median(x)),
                        float(_np.percentile(x, 90)),
                        float(x.max()))

            # 2.2 Aggregate per MMSI
            by = _dd(list)
            for r in all_rows:
                by[r["mmsi"]].append(r)

            bymmsi_csv = out_root / "summary_by_mmsi.csv"
            with open(bymmsi_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "mmsi","n_trips",
                    "ade_mean","ade_median","ade_p90","ade_max",
                    "fde_mean","fde_median","fde_p90","fde_max",
                    "mae_mean","mae_median","mae_p90","mae_max"
                ])
                for mmsi, rows in sorted(by.items()):
                    ade = [r["ade_km"] for r in rows]
                    fde = [r["fde_km"] for r in rows]
                    mae = [r["mae_km"] for r in rows]
                    a_mean, a_med, a_p90, a_max = _stats(ade)
                    f_mean, f_med, f_p90, f_max = _stats(fde)
                    m_mean, m_med, m_p90, m_max = _stats(mae)
                    w.writerow([
                        mmsi, len(rows),
                        f"{a_mean:.6f}", f"{a_med:.6f}", f"{a_p90:.6f}", f"{a_max:.6f}",
                        f"{f_mean:.6f}", f"{f_med:.6f}", f"{f_p90:.6f}", f"{f_max:.6f}",
                        f"{m_mean:.6f}", f"{m_med:.6f}", f"{m_p90:.6f}", f"{m_max:.6f}",
                    ])

            # 2.3 Overall summary (single row)
            ade_all = [r["ade_km"] for r in all_rows]
            fde_all = [r["fde_km"] for r in all_rows]
            mae_all = [r["mae_km"] for r in all_rows]
            a_mean, a_med, a_p90, a_max = _stats(ade_all)
            f_mean, f_med, f_p90, f_max = _stats(fde_all)
            m_mean, m_med, m_p90, m_max = _stats(mae_all)

            overall_csv = out_root / "summary_overall.csv"
            with open(overall_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "n_trips",
                    "ade_mean","ade_median","ade_p90","ade_max",
                    "fde_mean","fde_median","fde_p90","fde_max",
                    "mae_mean","mae_median","mae_p90","mae_max"
                ])
                w.writerow([
                    len(all_rows),
                    f"{a_mean:.6f}", f"{a_med:.6f}", f"{a_p90:.6f}", f"{a_max:.6f}",
                    f"{f_mean:.6f}", f"{f_med:.6f}", f"{f_p90:.6f}", f"{f_max:.6f}",
                    f"{m_mean:.6f}", f"{m_med:.6f}", f"{m_p90:.6f}", f"{m_max:.6f}",
                ])

            print(f"[summary files] {trips_csv} | {bymmsi_csv} | {overall_csv}")
    except Exception as e:
        print(f"[warn] failed to write global summary: {e}")

if __name__ == "__main__":
    main()
