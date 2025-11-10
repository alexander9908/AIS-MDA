# src/eval/eval_traj_V3.py
from __future__ import annotations
import argparse, os, glob, pickle, csv, datetime as dt, traceback, random
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import torch
import matplotlib.pyplot as plt

# Models
from ..models import GRUSeq2Seq, TPTrans
from ..models.traisformer1 import TrAISformer, BinSpec


from src.eval.build_water_mask import make_water_mask, snap_to_water_path
import numpy as np
from collections import deque

# ---------- Optional de-normalizer from preprocessing ----------
_HAS_DENORM_FN = False
try:
    from ..preprocessing.preprocessing import de_normalize_track as _de_normalize_track
    _HAS_DENORM_FN = True
except Exception:
    pass

# ---------- Map defaults ----------
DEFAULT_DK_EXTENT: Tuple[float, float, float, float] = (6.0, 16.0, 54.0, 58.0)

# Bounds (fallbacks if we need to denormalize but no preprocessing helper)
_DEFAULT_LAT_MIN = _DEFAULT_LAT_MAX = _DEFAULT_LON_MIN = _DEFAULT_LON_MAX = None
try:
    from ..preprocessing.preprocessing import LAT_MIN as _LAT_MIN, LAT_MAX as _LAT_MAX, \
        LON_MIN as _LON_MIN, LON_MAX as _LON_MAX
    _DEFAULT_LAT_MIN, _DEFAULT_LAT_MAX = float(_LAT_MIN), float(_LAT_MAX)
    _DEFAULT_LON_MIN, _DEFAULT_LON_MAX = float(_LON_MIN), float(_LON_MAX)
except Exception:
    pass

# ---------- Small helpers ----------
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

# ---------- Model factory ----------
def build_model(kind: str, feat_dim: int, horizon: int, d_model=192, nhead=4,
                enc_layers=4, dec_layers=2, bins: Optional[BinSpec]=None, tconf: Optional[dict]=None):
    if kind == "gru":
        return GRUSeq2Seq(feat_dim, d_model=d_model, layers=2, horizon=horizon)
    if kind == "traisformer":
        return TrAISformer(
            bins=bins,
            d_model=(tconf or {}).get("d_model", 512),
            nhead=(tconf or {}).get("nhead", 8),
            num_layers=(tconf or {}).get("enc_layers", 8),
            dropout=(tconf or {}).get("dropout", 0.1),
            coarse_merge=(tconf or {}).get("coarse_merge", 3),
            coarse_beta=(tconf or {}).get("coarse_beta", 0.2),
        )
    return TPTrans(feat_dim=feat_dim, d_model=d_model, nhead=nhead,
                   enc_layers=enc_layers, dec_layers=dec_layers, horizon=horizon)

# ---------- Strict tensor shape helper for TrAISformer ----------
def _to_idx_1xT(x: torch.Tensor, device) -> torch.Tensor:
    x = x.to(device).squeeze()
    if x.dim() == 0:
        x = x.view(1, 1)
    elif x.dim() == 1:
        x = x.unsqueeze(0)
    elif x.dim() > 2:
        x = x.squeeze()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() != 2:
            raise ValueError(f"Expected <=2 dims for bin indices, got {tuple(x.shape)}")
    return x.to(dtype=torch.long).contiguous()

# ---------- Core per-trip evaluation ----------
def evaluate_and_plot_trip(
    fpath: str,
    trip: np.ndarray,
    model,
    args,
    sample_idx: int,
) -> Dict[str, Any]:

    mmsi, tid = parse_trip(fpath)
    n_total = len(trip)

    past, future_true_all, cut = split_by_percent(trip, args.pred_cut)
    n_past = cut
    n_fut_raw = len(future_true_all)
    if n_past < 2 or n_fut_raw < 2:
        raise ValueError(f"too short after split (total={n_total}, past={n_past}, future={n_fut_raw})")

    # evaluation length
    N_future = n_fut_raw if args.cap_future is None else min(n_fut_raw, int(args.cap_future))
    if N_future < 1:
        raise ValueError("No future steps to predict after cap.")

    # lat/lon in degrees for plotting
    full_lat_deg, full_lon_deg = maybe_denorm_latlon(trip[:,0], trip[:,1],
                                                     args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    lats_past = full_lat_deg[:cut]; lons_past = full_lon_deg[:cut]
    cur_lat = float(lats_past[-1]); cur_lon = float(lons_past[-1])

    # green (full tail for plot) and green-eval (first N_future for metrics/trim)
    lats_true_all = full_lat_deg[cut:]
    lons_true_all = full_lon_deg[cut:]
    lats_true_eval = lats_true_all[:N_future]
    lons_true_eval = lons_true_all[:N_future]

    device = next(model.parameters()).device

    # ---------- MODEL BRANCHES ----------
    if args.model == "traisformer":
        seq_in = past[:, :4].astype(np.float32)  # [lat,lon,sog,cog]

        # AFTER — convert to degrees if the inputs look normalized
        lat_deg, lon_deg = maybe_denorm_latlon(
            seq_in[:,0], seq_in[:,1],
            args.lat_min, args.lat_max, args.lon_min, args.lon_max
        )

        # then bin degrees, not normalized values
        lat_idx = model.bins.lat_to_bin(torch.tensor(lat_deg, device=device))
        lon_idx = model.bins.lon_to_bin(torch.tensor(lon_deg, device=device))

        raw_sog, raw_cog = seq_in[:,2], seq_in[:,3]

        # robust SOG/COG handling (works for normalized or physical units)
        sog = (np.clip(raw_sog, 0.0, 1.0) * float(model.bins.sog_max)) if np.nanmax(raw_sog) <= 1.2 else np.clip(raw_sog, 0.0, float(model.bins.sog_max))
        cog = (raw_cog % 1.0) * 360.0 if np.nanmax(np.abs(raw_cog)) <= 1.5 else (raw_cog % 360.0)

        lat_idx = model.bins.lat_to_bin(torch.tensor(lat_deg, device=device))
        lon_idx = model.bins.lon_to_bin(torch.tensor(lon_deg, device=device))
        sog_idx = model.bins.sog_to_bin(torch.tensor(sog, device=device))
        cog_idx = model.bins.cog_to_bin(torch.tensor(cog, device=device))

        past_idxs = {
            "lat": _to_idx_1xT(lat_idx, device),
            "lon": _to_idx_1xT(lon_idx, device),
            "sog": _to_idx_1xT(sog_idx, device),
            "cog": _to_idx_1xT(cog_idx, device),
        }
        assert torch.unique(past_idxs["lat"]).numel() > 1, "lat bins look constant; check (de)normalization"
        assert torch.unique(past_idxs["lon"]).numel() > 1, "lon bins look constant; check (de)normalization"

        # sample K candidates, keep best ADE
        best = None
        K = max(1, int(getattr(args, "samples", 1)))
        for _ in range(K):
            out_idx = model.generate(
                past_idxs,
                L=N_future,
                sampling="sample" if args.temperature != 0 else "greedy",
                temperature=float(getattr(args, "temperature", 1.0)),
                top_k=int(getattr(args, "top_k", 20)),
            )
            cont = model.bins_to_continuous(out_idx)
            pred_lat = cont["lat"].squeeze(0).cpu().numpy()
            pred_lon = cont["lon"].squeeze(0).cpu().numpy()
            pred_sog = cont["sog"].squeeze(0).cpu().numpy()
            pred_cog = cont["cog"].squeeze(0).cpu().numpy()

            # fallback: if lat/lon collapsed, synthesize steps from SOG/COG
            #if float(np.nanstd(pred_lat)) < 1e-9 and float(np.nanstd(pred_lon)) < 1e-9:
            #    # step duration: use future cadence if available, else median, else 60s
            #    if cut + 1 < len(trip):
            #        ts_slice = trip[cut : min(len(trip), cut + 1 + len(pred_sog)), 7].astype(float)
            #        dts = np.diff(ts_slice)
            #        dt = float(np.nanmedian(dts[dts > 0])) if dts.size else 60.0
            #    else:
            #        ts_all = trip[:, 7].astype(float); dts = np.diff(ts_all)
            #        dt = float(np.nanmedian(dts[dts > 0])) if dts.size else 60.0
#
            #    R = 6371000.0
            #    lat_seq = [cur_lat]; lon_seq = [cur_lon]
            #    lat_now, lon_now = float(cur_lat), float(cur_lon)
            #    for k in range(len(pred_sog)):
            #        sog_kn = float(np.clip(pred_sog[k], 0.0, float(model.bins.sog_max)))
            #        cog_deg = float(pred_cog[k] % 360.0)
            #        ds_m = sog_kn * 0.514444 * dt
            #        theta = np.radians(90.0 - cog_deg)
            #        coslat = max(1e-6, np.cos(np.radians(lat_now)))
            #        dlon_deg = (ds_m * np.cos(theta)) / (R * coslat) * (180.0 / np.pi)
            #        dlat_deg = (ds_m * np.sin(theta)) / R * (180.0 / np.pi)
            #        lon_now += dlon_deg; lat_now += dlat_deg
            #        lon_seq.append(lon_now); lat_seq.append(lat_now)
            #    pred_lon = np.asarray(lon_seq, float)
            #    pred_lat = np.asarray(lat_seq, float)

            # anchor to current position
            if len(pred_lat) > 0:
                dlat0 = cur_lat - float(pred_lat[0]); dlon0 = cur_lon - float(pred_lon[0])
                pred_lat = pred_lat + dlat0; pred_lon = pred_lon + dlon0

            ade_tmp = np.mean([haversine_km(lats_true_eval[i], lons_true_eval[i],
                                            pred_lat[i], pred_lon[i])
                               for i in range(min(len(pred_lat), len(lats_true_eval)))])
            if (best is None) or (ade_tmp < best[0]):
                best = (ade_tmp, pred_lat, pred_lon)

        pred_lat, pred_lon = np.asarray(best[1]), np.asarray(best[2])

    else:
        # TPTrans / GRU iterative rollout on normalized features
        seq_in = past[:, :4].astype(np.float32)  # [lat,lon,sog,cog]
        def looks_norm(x): return (np.nanmin(x) >= -0.05 and np.nanmax(x) <= 1.2)
        seq_norm = seq_in.copy()
        if not (looks_norm(seq_in[:,0]) and looks_norm(seq_in[:,1])):
            if None in (args.lat_min, args.lat_max, args.lon_min, args.lon_max) and not _HAS_DENORM_FN:
                raise ValueError("Inputs appear degrees; provide bounds or implement de_normalize_track.")
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
        while remaining > 0:
            Tin = min(args.past_len, len(seq_norm))
            X_in = seq_norm[-Tin:, :][None, ...]
            with torch.no_grad():
                yraw = model(torch.from_numpy(X_in).to(device))[0].cpu().numpy()  # [H,2] normalized absolute
            keep = min(yraw.shape[0], remaining)
            lat_n = np.clip(yraw[:keep, 0], 0.0, 1.0)
            lon_n = np.clip(yraw[:keep, 1], 0.0, 1.0)
            lat_deg, lon_deg = maybe_denorm_latlon(lat_n, lon_n, args.lat_min, args.lat_max, args.lon_min, args.lon_max)
            if len(pred_lat_list) == 0 and keep > 0 and np.isfinite(lat_deg[0]) and np.isfinite(lon_deg[0]):
                dlat0 = cur_lat - float(lat_deg[0]); dlon0 = cur_lon - float(lon_deg[0])
                lat_deg = lat_deg + dlat0; lon_deg = lon_deg + dlon0
            pred_lat_list.extend(lat_deg.tolist()); pred_lon_list.extend(lon_deg.tolist())

            # feedback
            lat_n2 = (lat_deg - args.lat_min) / float(args.lat_max - args.lat_min)
            lon_n2 = (lon_deg - args.lon_min) / float(args.lon_max - args.lon_min)
            last_sog = seq_norm[-1,2] if seq_norm.shape[1] > 2 else 0.0
            last_cog = seq_norm[-1,3] if seq_norm.shape[1] > 3 else 0.0
            add_feats = np.stack([lat_n2, lon_n2,
                                  np.full_like(lat_n2, last_sog, dtype=np.float32),
                                  np.full_like(lon_n2, last_cog, dtype=np.float32)], axis=1).astype(np.float32)
            seq_norm = np.vstack([seq_norm, add_feats])
            remaining -= keep
            #if not args.iter_rollout: break
        pred_lat = np.asarray(pred_lat_list, float)
        pred_lon = np.asarray(pred_lon_list, float)



        # === NEW: snap TPTrans predictions to water if a mask is available ===
        #wm = getattr(args, "_wm", None)
        #bins = getattr(args, "_bins", None)
        #if getattr(args, "snap_to_water", True) and wm is not None and bins is not None \
        #   and wm.mean() < 0.99 and len(pred_lat) > 0:
        #    # helper: lat/lon -> grid indices
        #    def to_idx(lat_deg, lon_deg):
        #        i = np.floor((lat_deg - bins.lat_min) / (bins.lat_max - bins.lat_min + 1e-12) * bins.n_lat)
        #        j = np.floor((lon_deg - bins.lon_min) / (bins.lon_max - bins.lon_min + 1e-12) * bins.n_lon)
        #        i = np.clip(i.astype(int), 0, bins.n_lat - 1)
        #        j = np.clip(j.astype(int), 0, bins.n_lon - 1)
        #        return i, j
#
        #    # helper: grid indices -> bin midpoints (degrees)
        #    def idx_to_mid(i, j):
        #        lat = bins.lat_min + (i + 0.5) * (bins.lat_max - bins.lat_min) / bins.n_lat
        #        lon = bins.lon_min + (j + 0.5) * (bins.lon_max - bins.lon_min) / bins.n_lon
        #        return lat, lon
#
        #    # convert predicted path to grid, snap, convert back
        #    ii, jj = to_idx(pred_lat, pred_lon)
        #    ii2, jj2 = snap_to_water_path(ii.copy(), jj.copy(), wm)
        #    pred_lat, pred_lon = idx_to_mid(ii2, jj2)        


        #wm = getattr(args, "_wm", None)
        #bins = getattr(args, "_bins", None)
        #if wm is not None and bins is not None and wm.mean() < 0.99 and len(pred_lat):
        #    def to_idx(lat_deg, lon_deg):
        #        i = np.floor((lat_deg - bins.lat_min) / (bins.lat_max - bins.lat_min + 1e-12) * bins.n_lat).astype(int)
        #        j = np.floor((lon_deg - bins.lon_min) / (bins.lon_max - bins.lon_min + 1e-12) * bins.n_lon).astype(int)
        #        i = np.clip(i, 0, bins.n_lat - 1); j = np.clip(j, 0, bins.n_lon - 1)
        #        return i, j
        #    def idx_to_mid(i, j):
        #        lat = bins.lat_min + (i + 0.5) * (bins.lat_max - bins.lat_min) / bins.n_lat
        #        lon = bins.lon_min + (j + 0.5) * (bins.lon_max - bins.lon_min) / bins.n_lon
        #        return lat, lon
        #    ii, jj = to_idx(pred_lat, pred_lon)
        #    ii2, jj2 = snap_to_water_path(ii.copy(), jj.copy(), wm)
        #    pred_lat, pred_lon = idx_to_mid(ii2, jj2)
        
        wm  = getattr(args, "_wm", None)
        bins = getattr(args, "_bins", None)
        if getattr(args, "snap_to_water", True) and wm is not None and bins is not None and len(pred_lat):
            def to_idx(lat_deg, lon_deg):
                i = np.floor((lat_deg - bins.lat_min) / (bins.lat_max - bins.lat_min + 1e-12) * bins.n_lat).astype(int)
                j = np.floor((lon_deg - bins.lon_min) / (bins.lon_max - bins.lon_min + 1e-12) * bins.n_lon).astype(int)
                i = np.clip(i, 0, bins.n_lat - 1); j = np.clip(j, 0, bins.n_lon - 1)
                return i, j
            def idx_to_mid(i, j):
                lat = bins.lat_min + (i + 0.5) * (bins.lat_max - bins.lat_min) / bins.n_lat
                lon = bins.lon_min + (j + 0.5) * (bins.lon_max - bins.lon_min) / bins.n_lon
                return lat, lon
            ii, jj   = to_idx(pred_lat, pred_lon)
            ii2, jj2 = snap_to_water_path(ii.copy(), jj.copy(), wm)
            pred_lat, pred_lon = idx_to_mid(ii2, jj2)


    # ---------- Align lengths ----------
    N_true = len(lats_true_eval)
    if len(pred_lat) > N_true:
        pred_lat, pred_lon = pred_lat[:N_true], pred_lon[:N_true]
    elif len(pred_lat) < N_true:
        if len(pred_lat) == 0:
            pred_lat = np.full(N_true, cur_lat); pred_lon = np.full(N_true, cur_lon)
        else:
            pred_lat = np.concatenate([pred_lat, np.full(N_true-len(pred_lat), pred_lat[-1])])
            pred_lon = np.concatenate([pred_lon, np.full(N_true-len(pred_lon), pred_lon[-1])])

    # ---------- Optional distance trim to green ----------
    if args.match_distance:
        green_cd = cumdist(lats_true_eval, lons_true_eval)
        green_total = float(green_cd[-1]) if len(green_cd) else 0.0
        pred_cd = cumdist(np.r_[cur_lat, pred_lat], np.r_[cur_lon, pred_lon])[1:]
        if len(pred_cd):
            pred_total = float(pred_cd[-1])
            if pred_total > green_total + 1e-6:
                cut_idx = int(np.searchsorted(pred_cd, green_total, side="right") - 1)
                cut_idx = max(0, min(cut_idx, len(pred_cd)-1))
                if cut_idx < len(pred_cd) - 1:
                    frac = (green_total - pred_cd[cut_idx]) / (pred_cd[cut_idx+1] - pred_cd[cut_idx] + 1e-9)
                    lat_last = pred_lat[cut_idx] + frac * (pred_lat[cut_idx+1] - pred_lat[cut_idx])
                    lon_last = pred_lon[cut_idx] + frac * (pred_lon[cut_idx+1] - pred_lon[cut_idx])
                    pred_lat = np.concatenate([pred_lat[:cut_idx+1], [lat_last]])
                    pred_lon = np.concatenate([pred_lon[:cut_idx+1], [lon_last]])
                else:
                    pred_lat = pred_lat[:cut_idx+1]
                    pred_lon = pred_lon[:cut_idx+1]
        if len(pred_lat) == 0:
            pred_lat = np.array([cur_lat], float); pred_lon = np.array([cur_lon], float)

    # ---------- Final clean / metrics ----------
    pred_lat = np.asarray(pred_lat, float); pred_lon = np.asarray(pred_lon, float)
    mask = np.isfinite(pred_lat) & np.isfinite(pred_lon)
    if not np.any(mask):
        pred_lat = np.array([cur_lat], float); pred_lon = np.array([cur_lon], float)
    else:
        pred_lat = pred_lat[mask]; pred_lon = pred_lon[mask]

    n_eff = min(len(lats_true_eval), len(pred_lat))
    if n_eff < 1: n_eff = 1
    lats_true_eval = lats_true_eval[:n_eff]; lons_true_eval = lons_true_eval[:n_eff]
    pred_lat = pred_lat[:n_eff]; pred_lon = pred_lon[:n_eff]

    def ade_km(tlat, tlon, plat, plon) -> float:
        d = [haversine_km(tlat[i], tlon[i], plat[i], plon[i]) for i in range(min(len(tlat), len(plat)))]
        return float(np.mean(d)) if d else float("nan")
    def fde_km(tlat, tlon, plat, plon) -> float:
        n = min(len(tlat), len(plat))
        return haversine_km(tlat[n-1], tlon[n-1], plat[n-1], plon[n-1]) if n >= 1 else float("nan")
    def mae_km(tlat, tlon, plat, plon) -> float:
        d = [haversine_km(tlat[i], tlon[i], plat[i], plon[i]) for i in range(min(len(tlat), len(plat)))]
        return float(np.median(d)) if d else float("nan")

    ade, fde, mae = ade_km(lats_true_eval, lons_true_eval, pred_lat, pred_lon), \
                    fde_km(lats_true_eval, lons_true_eval, pred_lat, pred_lon), \
                    mae_km(lats_true_eval, lons_true_eval, pred_lat, pred_lon)

    # ---------- Map extent (include prediction) ----------
    if args.auto_extent:
        ext = robust_extent(np.r_[full_lat_deg, pred_lat], np.r_[full_lon_deg, pred_lon], sigma=args.extent_outlier_sigma)
    else:
        ext = DEFAULT_DK_EXTENT

    # ---------- Plot ----------
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
        try:
            gl.top_labels = False
            gl.right_labels = False
        except Exception:
            pass
        # ensure gridlines stay under the track
        for artist in getattr(gl, 'xline_artists', []) + getattr(gl, 'yline_artists', []):
            artist.set_zorder(0)
        ax.set_extent(ext, crs=proj)
        ax.plot(full_lon_deg, full_lat_deg, color="#999999", linewidth=1.0, alpha=0.3, transform=proj, label="full trip (context)", zorder=1)
        ax.plot(lons_past, lats_past, '-', color="#1f77b4", linewidth=1.8, transform=proj, label="past (input)", zorder=3)
        ax.plot([cur_lon], [cur_lat], 'o', color='k', markersize=5, transform=proj, label='current pos', zorder=6)
        if len(lats_true_all) >= 2:
            ax.plot(lons_true_all, lats_true_all, '-', color="#2ca02c", linewidth=1.8, transform=proj, label="true future", zorder=4)
        else:
            ax.plot(lons_true_all, lats_true_all, 'o', color="#2ca02c", transform=proj, zorder=4, label="true future (pt)")
        
        # *** SOLID red prediction ***
        if len(pred_lat) >= 2:
            ax.plot(
                pred_lon, pred_lat, '-',
                linewidth=3.0, color='#d33', alpha=0.95,
                transform=proj if HAS_CARTOPY else None,
                zorder=9,                   # above land/borders/grid
                solid_capstyle='round',
                clip_on=False,              # don't let coastlines/ticks cut the line
                label='pred future'
            )
            # ax.plot(pred_lon, pred_lat, '-', color="#d62728", linewidth=2.3,
            #         transform=proj, zorder=7, label="pred future")
        else:
            ax.plot(pred_lon, pred_lat, 'o', color="#d62728", transform=proj, zorder=7, label="pred future (pt)")
        # connector from current to first pred point (helps visibility)
        if len(pred_lon) > 0:
            ax.plot([cur_lon, pred_lon[0]], [cur_lat, pred_lat[0]], '-', color="#d62728",
                    linewidth=2.0, alpha=0.6, transform=proj, zorder=7)
    except Exception:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.plot(full_lon_deg, full_lat_deg, color="#999999", linewidth=1.0, alpha=0.3, label="full trip (context)", zorder=1)
        ax.plot(lons_past, lats_past, '-', color="#1f77b4", linewidth=1.8, label="past (input)", zorder=3)
        ax.plot([cur_lon], [cur_lat], 'o', color='k', markersize=5, label='current pos', zorder=6)
        if len(lats_true_all) >= 2:
            ax.plot(lons_true_all, lats_true_all, '-', color="#2ca02c", linewidth=1.8, label="true future", zorder=4)
        else:
            ax.plot(lons_true_all, lats_true_all, 'o', color="#2ca02c", zorder=4, label="true future (pt)")
        if len(pred_lat) >= 2:
            ax.plot(pred_lon, pred_lat, '-', color="#d62728", linewidth=2.3, label="pred future", zorder=7)
        else:
            ax.plot(pred_lon, pred_lat, 'o', color="#d62728", zorder=7, label="pred future (pt)")
        if len(pred_lon) > 0:
            ax.plot([cur_lon, pred_lon[0]], [cur_lat, pred_lat[0]], '-', color="#d62728", linewidth=2.0, alpha=0.6, zorder=7)

    t0 = float(np.nanmin(trip[:,7])); t1 = float(np.nanmax(trip[:,7]))
    future_len_full = len(lats_true_all)
    ax.set_title(f"Trajectory ({args.model}) — MMSI {mmsi} — {to_iso(t0, args.timefmt)} → {to_iso(t1, args.timefmt)}\n"
                 f"cut={args.pred_cut}%  future={future_len_full}  ADE={ade:.3f}km  FDE={fde:.3f}km  MAE~={mae:.3f}km")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.legend(loc="upper left", frameon=True)

    out_dir = Path(args.out_dir) / str(mmsi) if args.output_per_mmsi_subdir else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_path = out_dir / f"traj_{args.model}_mmsi-{mmsi}_trip-{tid}_cut-{args.pred_cut}_idx-{sample_idx}.png"
    plt.savefig(fig_path, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"[ok] saved {fig_path}")

    # per-trip CSV
    trip_csv = out_dir / f"trip_{mmsi}_{tid}_cut-{args.pred_cut}_idx-{sample_idx}.csv"
    with open(trip_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["idx","segment","timestamp","lat","lon"])
        ts_past     = trip[:cut, 7]
        ts_true_all = trip[cut:, 7]
        ts_true_eval= trip[cut:cut + len(lats_true_eval), 7]
        for i, (ts, la, lo) in enumerate(zip(ts_past, lats_past, lons_past)):
            w.writerow([i, "past", to_iso(ts), la, lo])
        base_idx = len(lats_past)
        for j, (ts, la, lo) in enumerate(zip(ts_true_all, lats_true_all, lons_true_all)):
            w.writerow([base_idx + j, "true_future", to_iso(ts), la, lo])
        for j, (ts, la, lo) in enumerate(zip(ts_true_eval, pred_lat, pred_lon)):
            w.writerow([base_idx + j, "pred_future", to_iso(ts), la, lo])

    # per-MMSI metrics CSV
    metrics_csv = out_dir / f"metrics_{mmsi}.csv"
    new_file = not metrics_csv.exists()
    with open(metrics_csv, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["mmsi","trip_id","pred_cut_pct","n_total","n_past","n_future","ade_km","fde_km","mae_km"])
        w.writerow([mmsi, tid, args.pred_cut, n_total, n_past, len(lats_true_eval),
                    f"{ade:.6f}", f"{fde:.6f}", f"{mae:.6f}"])

    return {
        "mmsi": mmsi, "trip_id": tid, "pred_cut_pct": float(args.pred_cut),
        "n_total": int(n_total), "n_past": int(n_past), "n_future": int(len(lats_true_eval)),
        "ade_km": float(ade), "fde_km": float(fde), "mae_km": float(mae),
    }

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(
        "Evaluate a trained trajectory model on map-reduced trips.\n"
        "Blue = past, Green = full true tail, Red = predicted future.\n"
        "Works with TPTrans, GRU (optional), and TrAISformer."
    )
    # IO & selection
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", default="data/figures")
    ap.add_argument("--mmsi", type=str, default="all", help="all or comma list")
    ap.add_argument("--trip_id", type=int, default=None)
    ap.add_argument("--max_plots", type=int, default=None)
    ap.add_argument("--min_points", type=int, default=30)

    # Model
    ap.add_argument("--model", choices=["tptrans","gru","traisformer"], default="tptrans")
    ap.add_argument("--horizon", type=int, default=12)    # TPTrans uses this; TrAISformer ignores
    ap.add_argument("--past_len", type=int, default=64)

    # TrAISformer bin spec (defaults match your training config)
    ap.add_argument("--lat_min", type=float, default=_DEFAULT_LAT_MIN if _DEFAULT_LAT_MIN is not None else 54.0)
    ap.add_argument("--lat_max", type=float, default=_DEFAULT_LAT_MAX if _DEFAULT_LAT_MAX is not None else 58.0)
    ap.add_argument("--lon_min", type=float, default=_DEFAULT_LON_MIN if _DEFAULT_LON_MIN is not None else 6.0)
    ap.add_argument("--lon_max", type=float, default=_DEFAULT_LON_MAX if _DEFAULT_LON_MAX is not None else 16.0)
    ap.add_argument("--sog_max", type=float, default=50.0)
    ap.add_argument("--n_lat", type=int, default=256)
    ap.add_argument("--n_lon", type=int, default=512)
    ap.add_argument("--n_sog", type=int, default=50)
    ap.add_argument("--n_cog", type=int, default=72)
    ap.add_argument("--t_d_model", type=int, default=512)
    ap.add_argument("--t_nhead", type=int, default=8)
    ap.add_argument("--t_enc_layers", type=int, default=8)
    ap.add_argument("--t_dropout", type=float, default=0.1)
    ap.add_argument("--t_coarse_merge", type=int, default=3)
    ap.add_argument("--t_coarse_beta", type=float, default=0.2)

    # Eval slicing
    ap.add_argument("--pred_cut", type=float, required=True)
    ap.add_argument("--cap_future", type=int, default=None)

    # Plot + behavior
    ap.add_argument("--auto_extent", action="store_true")
    ap.add_argument("--extent_outlier_sigma", type=float, default=3.0)
    ap.add_argument("--output_per_mmsi_subdir", action="store_true", default=True)
    ap.add_argument("--iter_rollout", action="store_true", default=True)
    ap.add_argument("--match_distance", action="store_true")
    ap.add_argument("--samples", type=int, default=8)       # TrAISformer only
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--snap_to_water", action="store_true", default=True,
                    help="Snap TPTrans/GRU predictions onto nearest water cell")
    ap.add_argument("--timefmt", type=str, default="%Y-%m-%d %H:%M:%S UTC")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # ----- Build model & load checkpoint -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    # --- LOAD CHECKPOINT AND BINS CONFIG (single source of truth) ---
    ckpt = torch.load(args.ckpt, map_location=device)

    # Prefer "model" then "state_dict" in the checkpoint
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))

    # Strip DistributedDataParallel prefixes if they exist
    cleaned = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v

    # ----- Build BinSpec / transformer config from ckpt if present -----
    if "bins" in ckpt:
        bins = BinSpec(**ckpt["bins"])
    else:
        bins = BinSpec(
            lat_min=args.lat_min, lat_max=args.lat_max,
            lon_min=args.lon_min, lon_max=args.lon_max,
            sog_max=float(args.sog_max),
            n_lat=int(args.n_lat), n_lon=int(args.n_lon),
            n_sog=int(args.n_sog), n_cog=int(args.n_cog),
        )

    if "tconf" in ckpt:
        tconf = ckpt["tconf"]
    else:
        tconf = dict(
            d_model=args.t_d_model, nhead=args.t_nhead, enc_layers=args.t_enc_layers,
            dropout=args.t_dropout, coarse_merge=args.t_coarse_merge, coarse_beta=args.t_coarse_beta,
        )

    # ----- Build model once, then restore weights once -----
    feat_dim = 4
    model = build_model(args.model, feat_dim, args.horizon, bins=bins, tconf=tconf).to(device)


    # Exposes a coastline water_mask; TrAISformer and TPTrans/GRU.
    wm = make_water_mask(
        bins.lat_min, bins.lat_max, bins.lon_min, bins.lon_max, bins.n_lat, bins.n_lon
    )   # True=water, False=land
    try:
        print(f"water mask: built {wm.shape}  water_fraction={wm.mean():.3f}")
    except Exception:
        pass

    # Make bins + coast mask available to the evaluator
    args._bins = bins
    args._wm = wm

    if hasattr(model, "water_mask") and getattr(model, "water_mask") is not None:
        try:
            wm = model.water_mask.detach().cpu().numpy()
            print("water mask: shape", wm.shape, "  water_fraction=", wm.mean())
            # Light sanity check for the TrAISformer mask
            assert wm.mean() < 0.99, "Water mask looks like 'all water' — cartopy/shapely not working."
        except Exception:
            # If anything odd happens, just continue without the diagnostic
            print("[warn] couldn't inspect water_mask; continuing without mask diagnostics.")
            wm = None



    # Load weights strictly (helpful while debugging mismatches)
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch.\nMissing: {missing}\nUnexpected: {unexpected}"
        )

    model.eval()



    # ----- File selection -----
    files = sorted(glob.glob(os.path.join(args.split_dir, "*_processed.pkl")))
    if args.mmsi != "all":
        wanted = {int(x) for x in args.mmsi.split(",")}
        files = [f for f in files if parse_trip(f)[0] in wanted]
    if args.trip_id is not None:
        files = [f for f in files if parse_trip(f)[1] == int(args.trip_id)]
    total = len(files)
    if args.max_plots and total > args.max_plots:
        rng = np.random.default_rng(args.seed)
        files = list(rng.choice(files, size=args.max_plots, replace=False))
    print(f"[select] total={total} selected={len(files)} mode={args.mmsi}")

    # ----- Evaluate -----
    per_trip_rows: List[Dict[str, Any]] = []
    n_skipped = 0
    err_shown = 0
    for i, fpath in enumerate(files):
        try:
            trip = load_trip(fpath, min_points=args.min_points)
            row = evaluate_and_plot_trip(fpath, trip, model, args, i)
            per_trip_rows.append(row)
        except Exception as e:
            n_skipped += 1
            if err_shown < 3 or args.verbose:
                print(f"[skip] {os.path.basename(fpath)}: {e}")
                if args.verbose:
                    traceback.print_exc()
                err_shown += 1

    # ----- Summaries -----
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # per-trip table
    sum_trips = out_root / "summary_trips.csv"
    with open(sum_trips, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_trip_rows[0].keys()) if per_trip_rows else
                           ["mmsi","trip_id","pred_cut_pct","n_total","n_past","n_future","ade_km","fde_km","mae_km"])
        w.writeheader()
        for r in per_trip_rows: w.writerow(r)

    # group by MMSI
    by_mmsi: Dict[int, List[Dict[str, Any]]] = {}
    for r in per_trip_rows:
        by_mmsi.setdefault(int(r["mmsi"]), []).append(r)
    sum_mmsi = out_root / "summary_by_mmsi.csv"
    with open(sum_mmsi, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mmsi","n_trips","ade_km_mean","fde_km_mean","mae_km_mean"])
        for m, rows in sorted(by_mmsi.items()):
            ade = np.mean([float(x["ade_km"]) for x in rows]) if rows else float("nan")
            fde = np.mean([float(x["fde_km"]) for x in rows]) if rows else float("nan")
            mae = np.mean([float(x["mae_km"]) for x in rows]) if rows else float("nan")
            w.writerow([m, len(rows), f"{ade:.6f}", f"{fde:.6f}", f"{mae:.6f}"])

    # overall
    sum_all = out_root / "summary_overall.csv"
    if per_trip_rows:
        ade = np.mean([float(x["ade_km"]) for x in per_trip_rows])
        fde = np.mean([float(x["fde_km"]) for x in per_trip_rows])
        mae = np.mean([float(x["mae_km"]) for x in per_trip_rows])
    else:
        ade = fde = mae = float("nan")
    with open(sum_all, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_trips","ade_km_mean","fde_km_mean","mae_km_mean","skipped"])
        w.writerow([len(per_trip_rows), f"{ade:.6f}", f"{fde:.6f}", f"{mae:.6f}", n_skipped])

    print(f"[summary] plotted={len(per_trip_rows)} skipped={n_skipped} total_selected={len(files)}")
    print(f"[summary files] {sum_trips} | {sum_mmsi} | {sum_all}")

if __name__ == "__main__":
    main()
