# src/eval/eval_traj_newnewnew.py
from __future__ import annotations
import argparse, os, glob, pickle, csv, datetime as dt
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
import traceback
#import cut


from ..models import GRUSeq2Seq, TPTrans

# Prefer your preprocessing scaler
_HAS_DENORM_FN = False
try:
    from ..preprocessing.preprocessing import de_normalize_track as _de_normalize_track
    _HAS_DENORM_FN = True
except Exception:
    pass

DEFAULT_DENMARK_EXTENT: Tuple[float, float, float, float] = (6.0, 16.0, 54.0, 58.0)

_DEFAULT_LAT_MIN = _DEFAULT_LAT_MAX = _DEFAULT_LON_MIN = _DEFAULT_LON_MAX = None
try:
    from ..preprocessing.preprocessing import LAT_MIN as _LAT_MIN, LAT_MAX as _LAT_MAX, LON_MIN as _LON_MIN, LON_MAX as _LON_MAX
    _DEFAULT_LAT_MIN, _DEFAULT_LAT_MAX = float(_LAT_MIN), float(_LAT_MAX)
    _DEFAULT_LON_MIN, _DEFAULT_LON_MAX = float(_LON_MIN), float(_LON_MAX)
except Exception:
    pass


# ----------------- utils -----------------
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
    return trip[:cut], trip[cut:], cut

def robust_extent(lats: np.ndarray, lons: np.ndarray, pad: float = 0.75,
                  clamp: Tuple[float, float, float, float] = DEFAULT_DENMARK_EXTENT,
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

def maybe_denorm_latlon(lat, lon, lat_min, lat_max, lon_min, lon_max):
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

def ade_km(true_lat, true_lon, pred_lat, pred_lon) -> float:
    n = len(true_lat); 
    if n == 0: return float("nan")
    d = [haversine_km(true_lat[i], true_lon[i], pred_lat[i], pred_lon[i]) for i in range(n)]
    return float(np.mean(d))

def fde_km(true_lat, true_lon, pred_lat, pred_lon) -> float:
    if len(true_lat) == 0: return float("nan")
    return haversine_km(true_lat[-1], true_lon[-1], pred_lat[-1], pred_lon[-1])

def mae_km(true_lat, true_lon, pred_lat, pred_lon) -> float:
    n = len(true_lat); 
    if n == 0: return float("nan")
    d = [haversine_km(true_lat[i], true_lon[i], pred_lat[i], pred_lon[i]) for i in range(n)]
    return float(np.median(d))  # median abs error in km


# ----------------- core -----------------
def evaluate_and_plot_trip(fpath: str, trip: np.ndarray, model, args, out_root: Path, sample_idx: int):
    # [lat, lon, sog, cog, heading, rot, nav, timestamp, mmsi]
    mmsi, tid = parse_trip(fpath)
    
    # sizes *before* any capping/rollout
    n_total = len(trip)
    n_past = cut
    n_future_raw = len(fut_true_all)

    # hard guard: need at least 2 past points and 2 future points after the split
    if n_past < 2 or n_future_raw < 2:
        raise ValueError(f"too short after split (total={n_total}, past={n_past}, future={n_future_raw})")

    total_future = len(fut_true_all)
    if args.cap_future is not None:
        total_future = min(total_future, int(args.cap_future))
    if total_future < 1:
        raise ValueError("No future steps to predict after cap.")

    # Denorm ALL for plotting
    full_lat_deg, full_lon_deg = maybe_denorm_latlon(trip[:,0], trip[:,1], args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    lats_past = full_lat_deg[:cut]; lons_past = full_lon_deg[:cut]

    # immutable current position for plotting
    cur0_lat = float(lats_past[-1]); cur0_lon = float(lons_past[-1])

    # "green" (true future) can be capped
    lats_true_all = full_lat_deg[cut:]; lons_true_all = full_lon_deg[cut:]
    lats_true = lats_true_all[:total_future]; lons_true = lons_true_all[:total_future]

    # ---------- iterative rollout ----------
    seq_in = past[:, :4].astype(np.float32)  # [lat,lon,sog,cog]
    # normalize to [0..1] if in degrees
    def looks_norm(x): return np.nanmin(x) >= -0.05 and np.nanmax(x) <= 1.2
    seq_norm = seq_in.copy()
    if not (looks_norm(seq_in[:,0]) and looks_norm(seq_in[:,1])):
        if None in (args.lat_min, args.lat_max, args.lon_min, args.lon_max) and not _HAS_DENORM_FN:
            raise ValueError("Inputs look like degrees; provide bounds or de_normalize_track.")
        lat_deg, lon_deg = seq_in[:,0], seq_in[:,1]
        seq_norm[:,0] = (lat_deg - args.lat_min) / float(args.lat_max - args.lat_min)
        seq_norm[:,1] = (lon_deg - args.lon_min) / float(args.lon_max - args.lon_min)
        try:
            from ..preprocessing.preprocessing import SPEED_MAX
            speed_max = float(SPEED_MAX)
        except Exception:
            speed_max = 30.0
        seq_norm[:,2] = np.clip(seq_in[:,2] / speed_max, 0.0, 1.0)
        seq_norm[:,3] = (seq_in[:,3] % 360.0) / 360.0

    remaining = int(total_future)
    pred_lat_list, pred_lon_list = [], []
    device = next(model.parameters()).device

    while remaining > 0:
        Tin = min(args.past_len, len(seq_norm))
        X_in = seq_norm[-Tin:, :][None, ...]
        with torch.no_grad():
            yraw = model(torch.from_numpy(X_in).to(device))[0].cpu().numpy()   # [H,2] ABS normalized

        keep = min(yraw.shape[0], remaining)
        lat_norm = np.clip(yraw[:keep, 0], 0.0, 1.0)
        lon_norm = np.clip(yraw[:keep, 1], 0.0, 1.0)
        lat_deg, lon_deg = maybe_denorm_latlon(lat_norm, lon_norm, args.lat_min, args.lat_max, args.lon_min, args.lon_max)

        # anchor FIRST chunk to cur0 (visual continuity)
        if len(pred_lat_list) == 0 and keep > 0 and np.isfinite(lat_deg[0]) and np.isfinite(lon_deg[0]):
            dlat0 = cur0_lat - float(lat_deg[0]); dlon0 = cur0_lon - float(lon_deg[0])
            lat_deg = lat_deg + dlat0; lon_deg = lon_deg + dlon0

        pred_lat_list.extend(lat_deg.tolist()); pred_lon_list.extend(lon_deg.tolist())

        # feedback (normalize to [0..1] for next rollout)
        lat_n = (lat_deg - args.lat_min) / float(args.lat_max - args.lat_min)
        lon_n = (lon_deg - args.lon_min) / float(args.lon_max - args.lon_min)
        last_sog = seq_norm[-1,2] if seq_norm.shape[1] > 2 else 0.0
        last_cog = seq_norm[-1,3] if seq_norm.shape[1] > 3 else 0.0
        add_feats = np.stack([lat_n, lon_n,
                              np.full_like(lat_n, last_sog, dtype=np.float32),
                              np.full_like(lon_n, last_cog, dtype=np.float32)], axis=1).astype(np.float32)
        seq_norm = np.vstack([seq_norm, add_feats])

        remaining -= keep
        if not args.iter_rollout: break

    pred_lat = np.asarray(pred_lat_list, float)
    pred_lon = np.asarray(pred_lon_list, float)

    # ---------- enforce equal lengths ----------
    N_true = len(lats_true)
    if len(pred_lat) > N_true:
        pred_lat, pred_lon = pred_lat[:N_true], pred_lon[:N_true]
    elif len(pred_lat) < N_true:
        # pad with last value (keeps metric defined)
        if len(pred_lat) == 0:
            pred_lat = np.full(N_true, cur0_lat); pred_lon = np.full(N_true, cur0_lon)
        else:
            pred_lat = np.concatenate([pred_lat, np.full(N_true-len(pred_lat), pred_lat[-1])])
            pred_lon = np.concatenate([pred_lon, np.full(N_true-len(pred_lon), pred_lon[-1])])


    # ---------- OPTIONAL: enforce equal *distance* (km), not just point count ----------
    def cumdist(lat, lon):
        if len(lat) == 0: return np.array([0.0])
        cd = [0.0]
        for i in range(1, len(lat)):
            cd.append(cd[-1] + haversine_km(lat[i-1], lon[i-1], lat[i], lon[i]))
        return np.asarray(cd, float)

    if True:  # distance-matching ON by default; flip to False if you only want point-count matching
        # distance along green (true)
        green_cd = cumdist(lats_true, lons_true)
        green_total = float(green_cd[-1]) if len(green_cd) else 0.0

        # distance along predicted (starting at the black dot to be fair)
        pred_cd = cumdist(np.r_[cur0_lat, pred_lat], np.r_[cur0_lon, pred_lon])
        # pred_cd includes the connector step, so normalize to start from 0 at first pred point:
        pred_cd = pred_cd[1:] if len(pred_cd) > 0 else pred_cd

        # trim/pad pred so its last point reaches green_total (within the step granularity)
        # find last index where pred_cd <= green_total
        if len(pred_cd):
            last_idx = int(np.searchsorted(pred_cd, green_total, side="right") - 1)
            last_idx = max(0, min(last_idx, len(pred_cd)-1))
            pred_lat = pred_lat[:last_idx+1]
            pred_lon = pred_lon[:last_idx+1]

            # if we still fall short in km vs green_total, extend by duplicating last point
            if haversine_km(lats_true[0], lons_true[0], pred_lat[-1], pred_lon[-1]) < 1e-9 and len(pred_lat) == 1:
                # degenerate one-point future; leave as is
                pass
            # optional: if the last predicted km is still under green_total by a tiny epsilon, keep as is.
            # Fine enough for visual matching without complicated interpolation.



    # ---------- metrics ----------
    ade = ade_km(lats_true, lons_true, pred_lat, pred_lon)
    fde = fde_km(lats_true, lons_true, pred_lat, pred_lon)
    mae = mae_km(lats_true, lons_true, pred_lat, pred_lon)

    # ---------- plotting ----------
    ext = robust_extent(full_lat_deg, full_lon_deg, sigma=args.extent_outlier_sigma) if args.auto_extent else DEFAULT_DENMARK_EXTENT

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
        ax.plot([cur0_lon], [cur0_lat], 'o', color='k', markersize=5, transform=proj, label='current pos', zorder=5)

        # --- True future (green)
        #ax.plot(lons_true, lats_true, '-', color="#2ca02c", linewidth=1.8, transform=proj if HAS_CARTOPY else None, label="true future", zorder=4)
        
        # --- Pred future (red)
        #ax.plot(pred_lon, pred_lat, '--', color="#d62728", linewidth=2.0, transform=proj if HAS_CARTOPY else None, label="pred future", zorder=5)
        
        # OPTIONAL: small dashed connector (cosmetic)
        #if len(pred_lon) > 0:
        #    ax.plot([cur0_lon, pred_lon[0]], [cur0_lat, pred_lat[0]], '--', color="#d62728", linewidth=2.0, transform=proj if HAS_CARTOPY else None, alpha=0.7, zorder=5)
        
        # --- True future (green)
        if len(lats_true) >= 2:
            ax.plot(lons_true, lats_true, '-', color="#2ca02c", linewidth=1.8,
                    transform=proj if HAS_CARTOPY else None, label="true future", zorder=4)
        elif len(lats_true) == 1:
            ax.plot([lons_true[0]], [lats_true[0]], 'o', color="#2ca02c",
                    transform=proj if HAS_CARTOPY else None, zorder=4, label="true future (pt)")

        # --- Pred future (red)
        if len(pred_lat) >= 2:
            ax.plot(pred_lon, pred_lat, '--', color="#d62728", linewidth=2.0,
                    transform=proj if HAS_CARTOPY else None, label="pred future", zorder=5)
        elif len(pred_lat) == 1:
            ax.plot([pred_lon[0]], [pred_lat[0]], 'x', color="#d62728",
                    transform=proj if HAS_CARTOPY else None, zorder=5, label="pred future (pt)")
        
        # OPTIONAL: small dashed connector (cosmetic)
        if len(pred_lon) > 0:
            ax.plot([cur0_lon, pred_lon[0]], [cur0_lat, pred_lat[0]], '--',
                    color="#d62728", linewidth=2.0, transform=proj if HAS_CARTOPY else None, alpha=0.7, zorder=5)


    except Exception:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.plot(full_lon_deg, full_lat_deg, color="#999999", linewidth=1.0, alpha=0.3, label="full trip (context)", zorder=1)
        ax.plot(lons_past, lats_past, '-', color="#1f77b4", linewidth=1.8, label="past (input)", zorder=3)
        ax.plot([cur0_lon], [cur0_lat], 'o', color='k', markersize=5, label='current pos', zorder=5)
        ax.plot(lons_true, lats_true, '-', color="#2ca02c", linewidth=1.8, label="true future", zorder=4)
        ax.plot(np.r_[cur0_lon, pred_lon], np.r_[cur0_lat, pred_lat], '--', color="#d62728", linewidth=2.0, label="pred future", zorder=5)

    t0 = float(np.nanmin(trip[:,7])); t1 = float(np.nanmax(trip[:,7]))
    ax.set_title(f"Trajectory ({args.model}) — MMSI {mmsi} — {to_iso(t0, args.timefmt)} → {to_iso(t1, args.timefmt)}\n"
                 f"cut={args.pred_cut}%  future={N_true}  ADE={ade:.3f}km  FDE={fde:.3f}km  MAE~={mae:.3f}km")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.legend(loc="upper left", frameon=True)

    # ---------- outputs ----------
    out_dir = Path(args.out_dir) / str(mmsi) if args.output_per_mmsi_subdir else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # figure
    fig_path = out_dir / f"traj_{args.model}_mmsi-{mmsi}_trip-{tid}_cut-{args.pred_cut}_idx-{sample_idx}.png"
    plt.savefig(fig_path, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"[ok] saved {fig_path}")

    # trip CSV (for plotting data)
    trip_csv = out_dir / f"trip_{mmsi}_{tid}_cut-{args.pred_cut}_idx-{sample_idx}.csv"
    with open(trip_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx","segment","timestamp","lat","lon"])

        ts_past = trip[:cut, 7]                   # (cut,)
        ts_true = trip[cut:cut+N_true, 7]         # (N_true,)

        # past
        for i, (ts, la, lo) in enumerate(zip(ts_past, lats_past, lons_past)):
            w.writerow([i, "past", to_iso(ts), la, lo])

        # true future
        base_idx = len(lats_past)
        for j, (ts, la, lo) in enumerate(zip(ts_true, lats_true, lons_true)):
            w.writerow([base_idx + j, "true_future", to_iso(ts), la, lo])

        # pred future (same length as green)
        for j, (ts, la, lo) in enumerate(zip(ts_true, pred_lat, pred_lon)):
            w.writerow([base_idx + j, "pred_future", to_iso(ts), la, lo])


    # per-MMSI metrics CSV (append or create)
    metrics_csv = out_dir / f"metrics_{mmsi}.csv"
    new_file = not metrics_csv.exists()
    with open(metrics_csv, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["mmsi","trip_id","pred_cut_pct","n_past","n_future","ade_km","fde_km","mae_km"])
        w.writerow([mmsi, tid, args.pred_cut, cut, N_true, f"{ade:.6f}", f"{fde:.6f}", f"{mae:.6f}"])

    # optional NPZ
    if args.debug_save_npz:
        dbg_dir = out_dir / 'debug_npz'; dbg_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dbg_dir / f"debug_{mmsi}_{tid}_idx-{sample_idx}.npz",
            full_lat=full_lat_deg, full_lon=full_lon_deg,
            past_lat=lats_past, past_lon=lons_past,
            true_lat=lats_true, true_lon=lons_true,
            pred_lat=pred_lat, pred_lon=pred_lon,
            cur_lat=cur0_lat, cur_lon=cur0_lon,
            extent=np.asarray(ext),
        )


def main():
    ap = argparse.ArgumentParser("Consistent full-trip eval with DK map, equal-length pred/true, per-MMSI folders + metrics.")
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", choices=["gru","tptrans"], default="tptrans")
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--past_len", type=int, default=64)
    ap.add_argument("--pred_cut", type=float, required=True)
    ap.add_argument("--cap_future", type=int, default=None)
    ap.add_argument("--out_dir", default="data/figures")
    ap.add_argument("--mmsi", type=str, default="all")
    ap.add_argument("--trip_id", type=int, default=None)
    ap.add_argument("--max_plots", type=int, default=None)
    ap.add_argument("--iter_rollout", action="store_true", default=True)
    ap.add_argument("--output_per_mmsi_subdir", action="store_true", default=True)
    ap.add_argument("--auto_extent", action="store_true")
    ap.add_argument("--extent_outlier_sigma", type=float, default=3.0)
    ap.add_argument("--denorm", action="store_true")  # retained for CLI compatibility
    ap.add_argument("--lat_min", type=float, default=_DEFAULT_LAT_MIN)
    ap.add_argument("--lat_max", type=float, default=_DEFAULT_LAT_MAX)
    ap.add_argument("--lon_min", type=float, default=_DEFAULT_LON_MIN)
    ap.add_argument("--lon_max", type=float, default=_DEFAULT_LON_MAX)
    ap.add_argument("--timefmt", type=str, default="%Y-%m-%d %H:%M:%S UTC")
    ap.add_argument("--debug_save_npz", action="store_true")
    ap.add_argument("--min_points", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # model
    model = build_model(args.model, feat_dim=4, horizon=args.horizon)
    state = torch.load(args.ckpt, map_location="cpu")
    try:
        model.load_state_dict(state, strict=True)
    except Exception as e:
        skipped += 1
        if args.verbose and skipped <= 3:
            print(f"[skip] {os.path.basename(f)}: {e}\n{traceback.format_exc()}")
        else:
            print(f"[skip] {os.path.basename(f)}: {e}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    files = sorted(glob.glob(os.path.join(args.split_dir, "*.pkl")))
    if not files: raise SystemExit(f"No trips found in {args.split_dir}")

    rng = np.random.default_rng(args.seed)

    def choose_all(fs):
        if args.max_plots is None or len(fs) <= args.max_plots: return fs
        idx = rng.choice(len(fs), size=args.max_plots, replace=False)
        return [fs[i] for i in sorted(idx)]

    # selection
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
    for i, f in enumerate(selected):
        try:
            trip = load_trip(f, min_points=args.min_points)
            if trip.shape[1] < 4: raise ValueError(f"Trip has D={trip.shape[1]}; need at least 4 [lat,lon,sog,cog].")
            evaluate_and_plot_trip(f, trip, model, args, Path(args.out_dir), sample_idx=i)
            ok += 1
        except Exception as e:
            skipped += 1; print(f"[skip] {os.path.basename(f)}: {e}")
    print(f"[summary] plotted={ok} skipped={skipped} total_selected={len(selected)}")

if __name__ == "__main__":
    main()
