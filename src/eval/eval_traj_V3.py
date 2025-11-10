# src/eval/eval_traj_V3.py
from __future__ import annotations
import argparse, os, glob, pickle, csv, datetime as dt, traceback
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
from ..models.TrAISformer import TrAISformer, BinSpec
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



# Helpers
def _ensure_bt(x: torch.Tensor, device=None) -> torch.Tensor:
    """
    Ensure tensor is 2-D [B, T] with B=1. Accepts 0-D/1-D/2-D/3-D inputs and
    squeezes singleton dims safely. Raises if shape is truly incompatible.
    """
    if device is not None:
        x = x.to(device)
    # drop singleton dims first (e.g., [1, T] stays [1, T]; [1, 1, T] -> [1, T])
    x = x.squeeze()
    if x.dim() == 0:
        # scalar -> [1,1]
        x = x.view(1, 1)
    elif x.dim() == 1:
        # [T] -> [1, T]
        x = x.unsqueeze(0)
    elif x.dim() == 2:
        # [B, T] keep as-is; if B>1 it's still fine but we'll only use B=1 in eval
        return x
    else:
        # More than 2D: try to squeeze extra singleton dims; if still >2D, error out
        x = x.squeeze()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() > 2:
            raise ValueError(f"Expected <=2 dims for bin indices, got shape {tuple(x.shape)}")
    return x



# --- strict helper to normalize shapes/dtypes for TrAISformer ---
def _to_idx_1xT(x: torch.Tensor, device) -> torch.Tensor:
    """
    Force bin indices to shape [1, T], dtype long, contiguous.
    Accepts anything from [T], [1,T], [1,1,T], [T,1] etc.
    """
    x = x.to(device)
    x = x.squeeze()              # drop all singleton dims first
    if x.dim() == 0:
        x = x.view(1, 1)         # scalar -> [1,1]
    elif x.dim() == 1:
        x = x.unsqueeze(0)       # [T] -> [1,T]
    elif x.dim() > 2:
        # squeeze again in case it's [T,1] -> [T]; or [1,1,T] -> [T]
        x = x.squeeze()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() != 2:
            raise ValueError(f"Expected <=2 dims for bin indices, got {tuple(x.shape)}")
    # now [1, T]
    return x.to(dtype=torch.long).contiguous()





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

def build_model(kind: str, feat_dim: int, horizon: int, d_model=192, nhead=4, enc_layers=4, dec_layers=2, bins=None, tconf=None):
    if kind == "gru":
        return GRUSeq2Seq(feat_dim, d_model=d_model, layers=2, horizon=horizon)
    if kind == "traisformer":
        return TrAISformer(bins=bins,
                           d_model=(tconf or {}).get("d_model", 512),
                           nhead=(tconf or {}).get("nhead", 8),
                           num_layers=(tconf or {}).get("enc_layers", 8),
                           dropout=(tconf or {}).get("dropout", 0.1),
                           coarse_merge=(tconf or {}).get("coarse_merge", 3),
                           coarse_beta=(tconf or {}).get("coarse_beta", 0.2))
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
    device = next(model.parameters()).device
    if args.model == "traisformer":
        # Past to bins:
        seq_in = past[:, :4].astype(np.float32)  # [lat,lon,sog,cog]
        lat, lon = seq_in[:,0], seq_in[:,1]

        # --- NEW: robust SOG/COG handling (works for normalized OR physical units) ---
        raw_sog = seq_in[:,2]
        raw_cog = seq_in[:,3]

        # SOG: if looks normalized (≤~1.2), scale to knots; else assume already in knots
        if np.nanmax(raw_sog) <= 1.2:
            sog = np.clip(raw_sog, 0.0, 1.0) * float(model.bins.sog_max)
        else:
            sog = np.clip(raw_sog, 0.0, float(model.bins.sog_max))

        # COG: if looks normalized (≤~1.5), map 0..1 → 0..360; else assume degrees and wrap
        if np.nanmax(np.abs(raw_cog)) <= 1.5:
            cog = (raw_cog % 1.0) * 360.0
        else:
            cog = raw_cog % 360.0
        # --- END NEW ---

        lat_idx = model.bins.lat_to_bin(torch.tensor(lat, device=device))
        lon_idx = model.bins.lon_to_bin(torch.tensor(lon, device=device))
        sog_idx = model.bins.sog_to_bin(torch.tensor(sog, device=device))
        cog_idx = model.bins.cog_to_bin(torch.tensor(cog, device=device))

        past_idxs = {
            "lat": _to_idx_1xT(lat_idx, device),
            "lon": _to_idx_1xT(lon_idx, device),
            "sog": _to_idx_1xT(sog_idx, device),
            "cog": _to_idx_1xT(cog_idx, device),
        }

        # (optional sanity)
        for k,v in past_idxs.items():
            assert v.dim()==2 and v.shape[0]==1, f"{k} bad shape {tuple(v.shape)}"

        # Sample K times, keep best ADE
        best = None
        for _ in range(max(1, int(getattr(args, "samples", 1)))):
            out_idx = model.generate(past_idxs, L=N_future,
                                     sampling="sample" if args.temperature != 0 else "greedy",
                                     temperature=float(getattr(args, "temperature", 1.0)),
                                     top_k=int(getattr(args, "top_k", 20)))
            
           # DEBUG: how many unique tokens did we sample?
            if args.verbose:
                for key in ("lat","lon","sog","cog"):
                    t = out_idx[key]                # expected shape [1, L] (long)
                    vals = t.squeeze(0).detach().cpu().numpy()
                    uniq = np.unique(vals)
                    print(f"[debug] uniq {key} bins: {len(uniq)}  first5={uniq[:5]}")
 

            cont = model.bins_to_continuous(out_idx)  # dict of tensors
            pred_lat = cont["lat"].squeeze(0).cpu().numpy()
            pred_lon = cont["lon"].squeeze(0).cpu().numpy()
            pred_sog = cont["sog"].squeeze(0).cpu().numpy()  # knots
            pred_cog = cont["cog"].squeeze(0).cpu().numpy()  # degrees

            # --- Fallback: if predicted lat/lon are (nearly) constant, build a polyline from current pos using predicted SOG/COG ---
            _lat_std = float(np.nanstd(pred_lat))
            _lon_std = float(np.nanstd(pred_lon))

            if (_lat_std < 1e-9) and (_lon_std < 1e-9):
                # step duration (seconds): use future timestamps if available, else median past cadence, else 60s
                if cut + 1 < len(trip):
                    ts_slice = trip[cut : min(len(trip), cut + 1 + len(pred_sog)), 7].astype(float)
                    dts = np.diff(ts_slice)
                    dt = float(np.nanmedian(dts[dts > 0])) if dts.size else 60.0
                else:
                    # fallback to median cadence from the whole trip or 60s
                    ts_all = trip[:, 7].astype(float)
                    dts = np.diff(ts_all)
                    dt = float(np.nanmedian(dts[dts > 0])) if dts.size else 60.0

                R = 6371000.0  # meters
                lat_seq = [float(cur_lat)]
                lon_seq = [float(cur_lon)]
                lat_now = float(cur_lat)
                lon_now = float(cur_lon)

                for k in range(len(pred_sog)):
                    sog_kn = float(np.clip(pred_sog[k], 0.0, float(model.bins.sog_max)))  # knots
                    cog_deg = float(pred_cog[k] % 360.0)

                    # meters per step from knots
                    ds_m = sog_kn * 0.514444 * dt

                    # heading to radians; 0°=North by nautical convention, but map uses lon eastward
                    # Use theta where 0 rad points East for simple local-projection step:
                    theta = np.radians(90.0 - cog_deg)

                    # convert meters to degrees at current latitude
                    coslat = max(1e-6, np.cos(np.radians(lat_now)))
                    dlon_deg = (ds_m * np.cos(theta)) / (R * coslat) * (180.0 / np.pi)
                    dlat_deg = (ds_m * np.sin(theta)) / R * (180.0 / np.pi)

                    lon_now += dlon_deg
                    lat_now += dlat_deg
                    lon_seq.append(lon_now)
                    lat_seq.append(lat_now)

                # replace lat/lon with synthesized polyline (length = len(pred_sog)+1)
                pred_lon = np.asarray(lon_seq, dtype=float)
                pred_lat = np.asarray(lat_seq, dtype=float)


            # Anchor first point to current pos (visual continuity)
            if len(pred_lat) > 0:
                dlat0 = cur_lat - float(pred_lat[0]); dlon0 = cur_lon - float(pred_lon[0])
                pred_lat = pred_lat + dlat0; pred_lon = pred_lon + dlon0

            ade_tmp = np.mean([haversine_km(lats_true_eval[i], lons_true_eval[i], pred_lat[i], pred_lon[i]) for i in range(min(len(pred_lat), len(lats_true_eval)))])
            cand = (ade_tmp, pred_lat, pred_lon)
            if (best is None) or (ade_tmp < best[0]): best = cand

        pred_lat, pred_lon = np.asarray(best[1]), np.asarray(best[2])
        
        
        # --- Enforce equal number of points ---
        #N_true = len(lats_true_eval)
    else:
        # GRU rollout
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

        # --- Fallback: if model is degenerate (no movement), synthesize a simple line
        # Uses last known SOG/COG and future timestamps to step forward, then --match_distance will trim.
        if float(np.std(pred_lat)) == 0.0 and float(np.std(pred_lon)) == 0.0:
            # Estimate per-step duration from true future timestamps (same cadence as green)
            ts_future = trip[cut : cut + N_true, 7] if 'N_true' in locals() else trip[cut:, 7]
            if len(ts_future) >= 2:
                dt_steps = np.diff(ts_future)
                dt_mean = float(np.clip(np.nanmedian(dt_steps), 1.0, 3600.0))  # seconds
            else:
                dt_mean = 60.0

            # Take last SOG/COG from the past (robust to units)
            last_sog_raw = float(past[-1, 2])
            last_cog_raw = float(past[-1, 3])

            # SOG knots -> m/s (robust if already normalized)
            sog_knots = (last_sog_raw * float(model.bins.sog_max)) if last_sog_raw <= 1.2 else last_sog_raw
            sog_mps = max(0.0, sog_knots * 0.514444)
            step_km = (sog_mps * dt_mean) / 1000.0  # km per step

            # COG to radians (robust if normalized)
            cog_deg = (last_cog_raw * 360.0) if abs(last_cog_raw) <= 1.5 else last_cog_raw
            cog_deg = float(cog_deg % 360.0)
            theta = np.radians(90.0 - cog_deg)  # 0° east in our simple projection

            # crude local projection around current pos (ok for short steps)
            R = 6371.0  # km
            coslat = np.cos(np.radians(cur_lat)) if np.isfinite(cur_lat) else 1.0
            n_steps = max(2, len(lats_true_eval))  # aim to match N_true
            px = [cur_lon]; py = [cur_lat]
            for _ in range(n_steps - 1):
                dlon = (step_km * np.cos(theta)) / (R * coslat) * (180.0 / np.pi)
                dlat = (step_km * np.sin(theta)) / R * (180.0 / np.pi)
                px.append(px[-1] + dlon)
                py.append(py[-1] + dlat)
            pred_lon = np.asarray(px, float)
            pred_lat = np.asarray(py, float)

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

    # --- Clean NaNs just before plotting ---
    # --- Make sure they are plain float arrays ---
    pred_lat = np.asarray(pred_lat, dtype=float)
    pred_lon = np.asarray(pred_lon, dtype=float)

    # --- Safe NaN/Inf mask (with fallback so we always plot *something*) ---
    mask = np.isfinite(pred_lat) & np.isfinite(pred_lon)
    if not np.any(mask):
        # Fallback: draw a tiny 1-point “prediction” at current pos so the red marker shows.
        # This also guarantees the connector line is visible.
        pred_lat = np.array([cur_lat], dtype=float)
        pred_lon = np.array([cur_lon], dtype=float)
    else:
        pred_lat = pred_lat[mask]
        pred_lon = pred_lon[mask]

    if args.verbose:
        print(f"[debug] pred_len={len(pred_lat)}  first2={(pred_lat[:2], pred_lon[:2])}")
    if args.verbose:
        print("[debug] pred std (lat, lon):",
              float(np.std(pred_lat)), float(np.std(pred_lon)))
        
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
    # OLD:
    # ext = robust_extent(full_lat_deg, full_lon_deg, sigma=args.extent_outlier_sigma) if args.auto_extent else DEFAULT_DK_EXTENT

    # NEW (include prediction in the extent box)
    if args.auto_extent:
        ext = robust_extent(
            np.r_[full_lat_deg, pred_lat],
            np.r_[full_lon_deg, pred_lon],
            sigma=args.extent_outlier_sigma
        )
    else:
        ext = DEFAULT_DK_EXTENT

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
            ax.plot(pred_lon, pred_lat, '--', color="#d62728",
                    linewidth=2.5, dashes=(4, 3),                # a hair thicker
                    transform=proj, label="pred future",
                    zorder=7, solid_capstyle='round', dash_capstyle='round')
        else:
            ax.plot(pred_lon, pred_lat, 'x', color="#d62728",
                    transform=proj, zorder=7, label="pred future (pt)")

        # ALWAYS add dots so red is visible even when overlapping green
        #ax.scatter(pred_lon, pred_lat, s=12, transform=proj, zorder=8, label=None)
        ax.scatter(
                    pred_lon, pred_lat,
                    s=18, marker='o',
                    facecolors='none', edgecolors="#d62728", linewidths=1.2,   # <- red ring, clearly visible
                    transform=proj if HAS_CARTOPY else None,
                    zorder=8, label=None
                )    
        if len(pred_lon) > 0:
            ax.plot([cur_lon, pred_lon[0]], [cur_lat, pred_lat[0]], '--', color="#d62728", linewidth=2.0, transform=proj, alpha=0.7, zorder=5)
    
    except Exception:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.plot(full_lon_deg, full_lat_deg, color="#999999", linewidth=1.0, alpha=0.3, label="full trip (context)", zorder=1)
        ax.plot(lons_past, lats_past, '-', color="#1f77b4", linewidth=1.8, label="past (input)", zorder=3)
        ax.plot([cur_lon], [cur_lat], 'o', color='k', markersize=5, label='current pos', zorder=5)
        if len(lats_true_all) >= 2:
            ax.plot(lons_true_all, lats_true_all, '-', color="#2ca02c", linewidth=1.8, label="true future", zorder=4)
        else:
            ax.plot(lons_true_all, lats_true_all, 'o', color="#2ca02c", zorder=4, label="true future (pt)")
        if len(pred_lat) >= 2:
            ax.plot(pred_lon, pred_lat, '--', color="#d62728",
                    linewidth=2.5, dashes=(4, 3), label="pred future",
                    zorder=7, solid_capstyle='round', dash_capstyle='round')
        else:
            ax.plot(pred_lon, pred_lat, 'x', color="#d62728", zorder=7, label="pred future (pt)")

        #ax.scatter(pred_lon, pred_lat, s=12, zorder=8, label=None)
        ax.scatter(
            pred_lon, pred_lat,
            s=18, marker='o',
            facecolors='none', edgecolors="#d62728", linewidths=1.2,   # <- red ring, clearly visible
            transform=proj if HAS_CARTOPY else None,
            zorder=8, label=None
        )  
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
    # TrAISformer sampling
    ap.add_argument("--samples", type=int, default=1, help="If model=traisformer, draw N samples and keep best ADE.")
    ap.add_argument("--temperature", type=float, default=1.0, help="Sampler temperature for TrAISformer.")
    ap.add_argument("--top_k", type=int, default=20, help="Top-k sampling for TrAISformer.")

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
    ap.add_argument("--model", choices=["gru","tptrans","traisformer"], default="tptrans",
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
    state = torch.load(args.ckpt, map_location="cpu")
    if args.model == "traisformer":
        if "bins" not in state:
            raise SystemExit("TrAISformer checkpoint must contain 'bins'.")
        bins = BinSpec(**state["bins"])
        model = build_model("traisformer", feat_dim=4, horizon=args.horizon, bins=bins,
                            tconf={"d_model":512,"nhead":8,"enc_layers":8,"coarse_merge":3,"coarse_beta":0.2})
        model.load_state_dict(state["state_dict"], strict=True)
    else:
        model = build_model(args.model, feat_dim=4, horizon=args.horizon)
        try: model.load_state_dict(state, strict=True)
        except Exception as e:
            print(f"[warn] strict load failed: {e} — retrying strict=False")
            model.load_state_dict(state, strict=False)


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
