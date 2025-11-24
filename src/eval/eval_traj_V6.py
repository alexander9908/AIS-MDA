# src/eval/eval_traj_V6.py
from __future__ import annotations
import argparse, os, glob, pickle, csv, datetime as dt
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import torch
import matplotlib.pyplot as plt

# ---------------- Models ----------------
# Adjust these imports to your tree if paths differ.
from src.models.traisformer1 import TrAISformer, BinSpec
from src.models.tptrans import TPTrans

# ---------------- Water mask (background only) ----------------
from src.eval.build_water_mask_V2 import make_water_mask

# Water guidance used for TPTrans rollout (project predictions to water)
from src.utils.water_guidance import is_water, project_to_water

# ---------------- Style ----------------
plt.rcParams.update({
    "figure.figsize": (7.5, 6.0),
    "axes.edgecolor": "#2a2a2a",
    "axes.labelcolor": "#2a2a2a",
    "xtick.color": "#2a2a2a",
    "ytick.color": "#2a2a2a",
    "font.size": 11,
})

# Optional: contextily basemap support
try:
    import contextily as ctx
    from xyzservices import providers as xz
    _HAS_CTX = True
except Exception:
    _HAS_CTX = False

# ---------------- Helpers ----------------
def parse_trip(fname: str) -> Tuple[int, int]:
    base = os.path.basename(fname).replace("_processed.pkl", "")
    mmsi_str, trip_id_str = base.split("_", 1)
    return int(mmsi_str), int(trip_id_str)

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

def robust_extent(lats: np.ndarray, lons: np.ndarray, pad: float = 0.4) -> Tuple[float,float,float,float]:
    lats = lats[np.isfinite(lats)]; lons = lons[np.isfinite(lons)]
    if lats.size == 0 or lons.size == 0:
        return (10.0, 13.0, 55.0, 58.0)
    lat_min, lat_max = float(np.min(lats)), float(np.max(lats))
    lon_min, lon_max = float(np.min(lons)), float(np.max(lons))
    if abs(lat_max - lat_min) < 0.2: lat_min -= 0.5; lat_max += 0.5
    if abs(lon_max - lon_min) < 0.2: lon_min -= 0.5; lon_max += 0.5
    lat_min -= pad; lat_max += pad; lon_min -= pad; lon_max += pad
    return (lon_min, lon_max, lat_min, lat_max)

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

def looks_norm(x: np.ndarray) -> bool:
    return (np.nanmin(x) >= -0.05 and np.nanmax(x) <= 1.05)

def to_idx_1xT(x: torch.Tensor, device) -> torch.Tensor:
    x = torch.as_tensor(x, device=device)  # <- key change
    x = x.squeeze()
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


# ---------------- Model factory ----------------
def clean_state_dict(sd):
    if "state_dict" in sd:
        sd = sd["state_dict"]
    if "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    new = {}
    for k, v in sd.items():
        nk = k[7:] if k.startswith("module.") else k
        new[nk] = v
    return new

def load_bins_from_ckpt(sd) -> BinSpec:
    if isinstance(sd, dict):
        if "bins" in sd and isinstance(sd["bins"], BinSpec):
            return sd["bins"]
        if "bins" in sd and isinstance(sd["bins"], dict):
            return BinSpec(**sd["bins"])
        flat = ["lat_min","lat_max","lon_min","lon_max","sog_max","n_lat","n_lon","n_sog","n_cog"]
        if all(k in sd for k in flat):
            return BinSpec(**{k: sd[k] for k in flat})
        for key in ("meta","cfg","config"):
            if key in sd and isinstance(sd[key], dict):
                inner = sd[key]
                if "bins" in inner and isinstance(inner["bins"], dict):
                    return BinSpec(**inner["bins"])
    raise KeyError("Could not find BinSpec in checkpoint.")

def build_model(kind: str, ckpt: str, feat_dim: int, horizon: int):
    sd_top = torch.load(ckpt, map_location="cpu")

    if kind.lower() == "tptrans":
        model = TPTrans(feat_dim=feat_dim, d_model=512, nhead = 4, enc_layers = 4, dec_layers = 2, horizon = horizon) # nhead=4, enc_layers = 4, dec_layers = 2, horizon = horizon) #  d_model=192 , nhead=4, enc_layers=4, dec_layers=2, horizon=horizon)
        model.load_state_dict(clean_state_dict(sd_top), strict=False)
        return model

    if kind.lower() == "traisformer":
        bins = load_bins_from_ckpt(sd_top)
        d_model = sd_top.get("d_model", 512)
        nhead = sd_top.get("nhead", 8)
        num_layers = sd_top.get("num_layers", 8)
        dropout = sd_top.get("dropout", 0.1)
        coarse_merge = sd_top.get("coarse_merge", 3)
        coarse_beta = sd_top.get("coarse_beta", 0.2)
        model = TrAISformer(
            bins=bins,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            coarse_merge=coarse_merge,
            coarse_beta=coarse_beta,
            use_water_mask=True,
        )
        model.load_state_dict(clean_state_dict(sd_top), strict=False)
        return model

    raise ValueError(f"unknown model kind: {kind}")

# ---------------- Core per-trip evaluation ----------------
def evaluate_and_plot_trip(
    fpath: str,
    trip: np.ndarray,
    model,
    args,
    sample_idx: int,
) -> Dict[str, Any]:

    mmsi, tid = parse_trip(fpath)

    # ---- split exactly at pred_cut ----
    past, future_true_all, cut = split_by_percent(trip, args.pred_cut)
    if len(past) < 2 or len(future_true_all) < 2:
        raise ValueError("too short after split")

    N_future = len(future_true_all) if args.cap_future is None else min(len(future_true_all), int(args.cap_future))

    # De/normalize to degrees for plotting
    past_lat_raw, past_lon_raw = past[:,0], past[:,1]
    full_lat_raw, full_lon_raw = trip[:,0], trip[:,1]

    if args.model.lower() == "traisformer":
        # Auto-de-normalize using model.bins (no CLI bounds needed)
        bm = model.bins
        if looks_norm(full_lat_raw) and looks_norm(full_lon_raw):
            full_lat_deg = full_lat_raw*(bm.lat_max-bm.lat_min) + bm.lat_min
            full_lon_deg = full_lon_raw*(bm.lon_max-bm.lon_min) + bm.lon_min
        else:
            full_lat_deg, full_lon_deg = full_lat_raw, full_lon_raw
    else:
        # TPTrans plots use either given bounds or already-degree inputs
        if looks_norm(full_lat_raw) and looks_norm(full_lon_raw):
            if None in (args.lat_min, args.lat_max, args.lon_min, args.lon_max):
                raise ValueError("TPTrans: normalized inputs; provide --lat_min/--lat_max/--lon_min/--lon_max.")
            full_lat_deg = full_lat_raw*(args.lat_max-args.lat_min) + args.lat_min
            full_lon_deg = full_lon_raw*(args.lon_max-args.lon_min) + args.lon_min
        else:
            full_lat_deg, full_lon_deg = full_lat_raw, full_lon_raw

    lats_past = full_lat_deg[:cut]; lons_past = full_lon_deg[:cut]
    cur_lat = float(lats_past[-1]); cur_lon = float(lons_past[-1])
    lats_true_eval = full_lat_deg[cut:cut+N_future]
    lons_true_eval = full_lon_deg[cut:cut+N_future]
    lats_true_all = full_lat_deg[cut:]; lons_true_all = full_lon_deg[cut:]

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = model.to(device).eval()

    # ---- Predict tail ----
    if args.model.lower() == "traisformer":
        seq_in = past[:, :4].astype(np.float32)

        # Degrees for binning
        if looks_norm(seq_in[:,0]) and looks_norm(seq_in[:,1]):
            bm = model.bins
            lat_deg = seq_in[:,0]*(bm.lat_max-bm.lat_min) + bm.lat_min
            lon_deg = seq_in[:,1]*(bm.lon_max-bm.lon_min) + bm.lon_min
        else:
            lat_deg, lon_deg = seq_in[:,0], seq_in[:,1]

        lat_idx = model.bins.lat_to_bin(torch.tensor(lat_deg, device=device))
        lon_idx = model.bins.lon_to_bin(torch.tensor(lon_deg, device=device))

        raw_sog, raw_cog = seq_in[:,2], seq_in[:,3]
        sog = (np.clip(raw_sog, 0.0, 1.0) * float(model.bins.sog_max)) if np.nanmax(raw_sog) <= 1.2 else np.clip(raw_sog, 0.0, float(model.bins.sog_max))
        cog = (raw_cog % 1.0) * 360.0 if np.nanmax(np.abs(raw_cog)) <= 1.5 else (raw_cog % 360.0)
        sog_idx = model.bins.sog_to_bin(torch.tensor(sog, device=device))
        cog_idx = model.bins.cog_to_bin(torch.tensor(cog, device=device))

        past_idxs = {
            "lat": to_idx_1xT(lat_idx, device),
            "lon": to_idx_1xT(lon_idx, device),
            "sog": to_idx_1xT(sog_idx, device),
            "cog": to_idx_1xT(cog_idx, device),
        }

        best = None
        K = max(1, int(getattr(args, "samples", 1)))
        with torch.no_grad():
            for _ in range(K):
                out_idx = model.generate(
                    past_idxs,
                    L=N_future,
                    sampling="sample" if args.temperature > 0 else "greedy",
                    temperature=float(args.temperature),
                    top_k=int(args.top_k),
                )
                cont = model.bins_to_continuous(out_idx)
                pred_lat = cont["lat"].squeeze(0).cpu().numpy()
                pred_lon = cont["lon"].squeeze(0).cpu().numpy()


                # displacement anchoring: first point = cut; keep step deltas
                if len(pred_lat) > 0:
                    raw_lat = pred_lat.copy(); raw_lon = pred_lon.copy()
                    pred_lat[0] = cur_lat;     pred_lon[0] = cur_lon
                    for t in range(1, len(pred_lat)):
                        pred_lat[t] = pred_lat[t-1] + (raw_lat[t] - raw_lat[t-1])
                        pred_lon[t] = pred_lon[t-1] + (raw_lon[t] - raw_lon[t-1])


                ade_tmp = np.mean([haversine_km(lats_true_eval[i], lons_true_eval[i],
                                                pred_lat[i], pred_lon[i])
                                   for i in range(min(len(pred_lat), len(lats_true_eval)))])
                if (best is None) or (ade_tmp < best[0]):
                    best = (ade_tmp, pred_lat, pred_lon)
        pred_lat, pred_lon = np.asarray(best[1]), np.asarray(best[2])

    else:  # TPTrans
        seq_in = past[:, :4].astype(np.float32)

        # Normalize inputs (model expects normalized abs lat/lon)
        if looks_norm(seq_in[:,0]) and looks_norm(seq_in[:,1]):
            seq_norm = seq_in.copy()
        else:
            if None in (args.lat_min, args.lat_max, args.lon_min, args.lon_max):
                raise ValueError("TPTrans requires bounds to normalize.")
            seq_norm = seq_in.copy()
            seq_norm[:,0] = (seq_in[:,0] - args.lat_min) / float(args.lat_max - args.lat_min)
            seq_norm[:,1] = (seq_in[:,1] - args.lon_min) / float(args.lon_max - args.lon_min)
        speed_max = float(getattr(args, "speed_max", 30.0))
        if np.nanmax(seq_norm[:,2]) > 1.5:  # not normalized
            seq_norm[:,2] = np.clip(seq_in[:,2] / speed_max, 0.0, 1.0)
        if np.nanmax(np.abs(seq_norm[:,3])) > 1.5:  # not normalized
            seq_norm[:,3] = (seq_in[:,3] % 360.0) / 360.0

        remaining = int(N_future)
        pred_lat_list, pred_lon_list = [], []

        with torch.no_grad():
            while remaining > 0:
                Tin = min(args.past_len, len(seq_norm))
                X_in = seq_norm[-Tin:, :][None, ...]  # [1,T,4]
                yraw = model(torch.from_numpy(X_in).to(device))[0].cpu().numpy()  # [H,2] normalized absolute
                keep = min(yraw.shape[0], remaining)

                lat_n = np.clip(yraw[:keep, 0], 0.0, 1.0)
                lon_n = np.clip(yraw[:keep, 1], 0.0, 1.0)
                # de-normalize to degrees for evaluation
                lat_deg = lat_n*(args.lat_max-args.lat_min) + args.lat_min
                lon_deg = lon_n*(args.lon_max-args.lon_min) + args.lon_min

                # Keep running "previous placed" point
                prev_lat, prev_lon = (cur_lat, cur_lon) if len(pred_lat_list) == 0 else (pred_lat_list[-1], pred_lon_list[-1])

                # ---- TPTrans displacement-anchored, water-aware rollout with min-step guard ----
                # constants
                MIN_STEP_KM = 0.25     # ~250 m; try 0.15–0.30
                HEAD_BLEND0 = 0.70     # first 2 steps: % of track bearing (rest from model)
                HEAD_BLEND  = 0.50     # subsequent steps

                # helper: bearing between last two past (degrees)
                def track_bearing_deg(lat1, lon1, lat2, lon2):
                    la1, lo1, la2, lo2 = map(np.radians, [lat1, lon1, lat2, lon2])
                    y = np.sin(lo2-lo1) * np.cos(la2)
                    x = np.cos(la1)*np.sin(la2) - np.sin(la1)*np.cos(la2)*np.cos(lo2-lo1)
                    brg = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
                    return brg

                # starting state
                prev_lat, prev_lon = (cur_lat, cur_lon) if len(pred_lat_list) == 0 else (pred_lat_list[-1], pred_lon_list[-1])

                # track bearing from last two past points (fallback to COG if unavailable)
                if len(lats_past) >= 2:
                    brg_track = track_bearing_deg(lats_past[-2], lons_past[-2], lats_past[-1], lons_past[-1])
                else:
                    brg_track = None

                for k in range(keep):
                    # ---------- 1) Displacement anchoring (preserve model's step delta) ----------
                    if k == 0 and len(pred_lat_list) == 0:
                        # first predicted point = cut point
                        cand_lat, cand_lon = prev_lat, prev_lon
                    else:
                        dlat = float(lat_deg[k] - lat_deg[k-1])
                        dlon = float(lon_deg[k] - lon_deg[k-1])
                        # optional initial heading blend for first two steps
                        if (k <= 1) and (brg_track is not None):
                            # rotate (dlat,dlon) a bit toward track bearing keeping its norm
                            # convert a small forward step along brg_track and blend with model delta
                            # (very light touch; avoids tiny back-and-forth at cut)
                            step_km = 111.0 * np.hypot(dlat, dlon)  # rough deg->km (ok for small steps)
                            step_km = max(step_km, MIN_STEP_KM)     # ensure some forward progress
                            w = HEAD_BLEND0 if k == 0 else HEAD_BLEND
                            # forward unit vector from bearing
                            th = np.radians(brg_track)
                            f_dlat = (step_km / 111.0) * np.cos(th)
                            f_dlon = (step_km / (111.0 * np.cos(np.radians(prev_lat)))) * np.sin(th + 1e-12)
                            dlat = (1 - w) * dlat + w * f_dlat
                            dlon = (1 - w) * dlon + w * f_dlon

                        cand_lat = prev_lat + dlat
                        cand_lon = prev_lon + dlon

                    # ---------- 2) Water projection (strict) ----------
                    if is_water(cand_lat, cand_lon):
                        fix_lat, fix_lon = cand_lat, cand_lon
                    else:
                        fix_lat, fix_lon = project_to_water(prev_lat, prev_lon, cand_lat, cand_lon)

                    # ---------- 3) Min-step guard (avoid tiny “nub” after cut) ----------
                    # if step is too short after projection, push MIN_STEP_KM along track and re-project
                    d_km = 111.0 * np.hypot(fix_lat - prev_lat,
                                            (fix_lon - prev_lon) * np.cos(np.radians(prev_lat)))
                    if d_km < MIN_STEP_KM:
                        if brg_track is not None:
                            th = np.radians(brg_track)
                            nlat = prev_lat + (MIN_STEP_KM / 111.0) * np.cos(th)
                            nlon = prev_lon + (MIN_STEP_KM / (111.0 * np.cos(np.radians(prev_lat)))) * np.sin(th + 1e-12)
                            if is_water(nlat, nlon):
                                fix_lat, fix_lon = nlat, nlon
                            else:
                                fix_lat, fix_lon = project_to_water(prev_lat, prev_lon, nlat, nlon)
                        # else: keep as is

                    pred_lat_list.append(fix_lat)
                    pred_lon_list.append(fix_lon)
                    prev_lat, prev_lon = fix_lat, fix_lon




                # FEEDBACK uses the FIXED coords
                lat_seg = np.asarray(pred_lat_list[-keep:], float)
                lon_seg = np.asarray(pred_lon_list[-keep:], float)
                lat_n2 = (lat_seg - args.lat_min) / float(args.lat_max - args.lat_min)
                lon_n2 = (lon_seg - args.lon_min) / float(args.lon_max - args.lon_min)
                last_sog = seq_norm[-1,2] if seq_norm.shape[1] > 2 else 0.0
                last_cog = seq_norm[-1,3] if seq_norm.shape[1] > 3 else 0.0
                add_feats = np.stack([lat_n2, lon_n2,
                                      np.full_like(lat_n2, last_sog, dtype=np.float32),
                                      np.full_like(lon_n2, last_cog, dtype=np.float32)], axis=1).astype(np.float32)
                seq_norm = np.vstack([seq_norm, add_feats])
                remaining -= keep

        pred_lat = np.asarray(pred_lat_list, float)
        pred_lon = np.asarray(pred_lon_list, float)

    # Optionally trim predicted length to match true evaluation distance
    if args.match_distance and len(pred_lat) > 1 and len(lats_true_eval) > 1:
        dt_true = cumdist(lats_true_eval, lons_true_eval)[-1]
        cd = cumdist(pred_lat, pred_lon)
        keep = int(np.searchsorted(cd, dt_true, side="right"))
        keep = max(1, min(keep, len(pred_lat)))
        pred_lat, pred_lon = pred_lat[:keep], pred_lon[:keep]

    # ---- Metrics ----
    n_comp = min(len(pred_lat), len(lats_true_eval))
    ade = float(np.mean([haversine_km(lats_true_eval[i], lons_true_eval[i],
                                      pred_lat[i], pred_lon[i]) for i in range(n_comp)])) if n_comp > 0 else np.nan
    fde = float(haversine_km(lats_true_eval[n_comp-1], lons_true_eval[n_comp-1],
                             pred_lat[n_comp-1], pred_lon[n_comp-1])) if n_comp > 0 else np.nan

    # ---- Plot ----
    outdir_mmsi = Path(args.out_dir) / f"{mmsi}"
    outdir_mmsi.mkdir(parents=True, exist_ok=True)
    fname_png = outdir_mmsi / f"traj_{args.model}_mmsi-{mmsi}_trip-{tid}_cut-{args.pred_cut}_idx-{sample_idx}.png"

    ext = robust_extent(
        np.concatenate([lats_past, lats_true_all, pred_lat]),
        np.concatenate([lons_past, lons_true_all, pred_lon]),
        pad=0.35 if args.auto_extent else 0.2
    )
    lon_min, lon_max, lat_min, lat_max = ext

    fig, ax = plt.subplots()
    # Basemap background (with robust fallback)
    drew_background = False
    if getattr(args, "style", "classic") != "classic" and _HAS_CTX:
        try:
            style = str(args.style).strip().lower()
            providers = {
                "satellite": xz.Esri.WorldImagery,
                "terrain":   xz.Stamen.Terrain,
                "toner":     xz.Stamen.TonerLite,
                "watercolor": xz.Stamen.Watercolor,
                "positron":  xz.CartoDB.Positron,
                "osm":       xz.OpenStreetMap.Mapnik,
            }
            src = providers.get(style, xz.Esri.WorldImagery)
            # Set extent first so contextily knows the window
            ax.set_xlim(lon_min, lon_max)
            ax.set_ylim(lat_min, lat_max)
            ctx.add_basemap(ax, source=src, crs="EPSG:4326", attribution_size=4)
            drew_background = True
        except Exception:
            drew_background = False

    if not drew_background:
        # Fallback: land/water shading with soft colors for a more "map-like" look
        try:
            wm = make_water_mask(lat_min, lat_max, lon_min, lon_max, n_lat=256, n_lon=512)
            lat_edges = np.linspace(lat_min, lat_max, 256+1)
            lon_edges = np.linspace(lon_min, lon_max, 512+1)
            # land=True -> 1, water=False -> 0
            land = (~wm).astype(float)
            # Colors: slightly darker blue water and darker gray land
            water_rgba = (0.50, 0.68, 0.88, 0.90)  # darker blue
            land_rgba  = (0.78, 0.78, 0.78, 0.90)  # darker gray
            # Build an RGBA image
            img = np.zeros((land.shape[0], land.shape[1], 4), dtype=float)
            img[...] = water_rgba
            img[land > 0.5] = land_rgba
            ax.pcolormesh(lon_edges, lat_edges, land, shading="auto", alpha=0.0)  # set up grid
            ax.imshow(img, extent=(lon_min, lon_max, lat_min, lat_max), origin="lower", aspect="auto")
        except Exception:
            # Minimal fallback: light grey
            ax.set_facecolor((0.96, 0.96, 0.96))

    # Classic solid lines as before
    ax.plot(lons_past, lats_past, lw=1.6, color="#2a77ff", alpha=0.9, label="past")
    ax.plot(lons_true_all, lats_true_all, lw=2.2, color="#2aaa2a", alpha=0.95, label="true (future)")
    if len(pred_lon) >= 2:
        ax.plot(pred_lon, pred_lat, lw=2.2, color="#d33", alpha=0.95, label=f"pred ({args.model})")
    else:
        ax.scatter(pred_lon, pred_lat, s=20, color="#d33", label=f"pred ({args.model})")

    ax.scatter([lons_past[-1]], [lats_past[-1]], s=36, color="#d33", edgecolor="k", zorder=5)  # cut
    if len(pred_lon) > 0:
        ax.scatter([pred_lon[0]], [pred_lat[0]], s=22, color="#d33", zorder=6, alpha=0.9)

    ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.set_title(f"MMSI {mmsi} · Trip {tid} · Cut {args.pred_cut:.1f}% · ADE {ade:.2f} km · FDE {fde:.2f} km")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(fname_png, dpi=int(getattr(args, "dpi", 180)))
    plt.close(fig)

    return {
        "mmsi": mmsi, "trip": tid, "cut_idx": cut,
        "ade_km": ade, "fde_km": fde,
        "n_future_true": len(lats_true_eval),
        "n_future_pred": len(pred_lat),
        "png": str(fname_png),
    }

# ---------------- Main ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split_dir", required=True, help="folder with *_processed.pkl")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model", choices=["tptrans","traisformer"], required=True)
    p.add_argument("--horizon", type=int, default=12, help="decoder horizon for tptrans")
    p.add_argument("--past_len", type=int, default=64)
    p.add_argument("--pred_cut", type=float, default=80.0, help="percent split: past/future")
    p.add_argument("--mmsi", type=str, default="", help="comma sep MMSIs to include (optional)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--auto_extent", action="store_true")
    p.add_argument("--match_distance", action="store_true")
    p.add_argument("--samples", type=int, default=1, help="TrAISformer: #samples keep best ADE")
    p.add_argument("--temperature", type=float, default=1.0, help="TrAISformer sampling temperature")
    p.add_argument("--top_k", type=int, default=60, help="TrAISformer top-k sampling")
    p.add_argument("--cap_future", type=int, default=None, help="cap evaluation tail length")
    p.add_argument("--lat_min", type=float, default=None)
    p.add_argument("--lat_max", type=float, default=None)
    p.add_argument("--lon_min", type=float, default=None)
    p.add_argument("--lon_max", type=float, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--speed_max", type=float, default=30.0, help="for TPTrans normalization if speeds present")
    p.add_argument("--style", choices=["classic","satellite","terrain","toner","watercolor","positron","osm"],
                   default="classic", help="Basemap style (satellite requires contextily/xyzservices).")
    p.add_argument("--dpi", type=int, default=180, help="Output figure DPI")
    # TrAISformer sampler tuning (optional)
    p.add_argument("--lambda_cont", type=float, default=None, help="Continuity weight (lower = less sticky)")
    p.add_argument("--alpha_dir", type=float, default=None, help="Direction push weight (higher = moves more along heading)")
    p.add_argument("--beta_turn", type=float, default=None, help="Turn penalty (lower = more willing to turn/move)")
    p.add_argument("--step_scale", type=float, default=None, help="Step scale factor (higher = larger steps)")
    p.add_argument("--sog_floor", type=float, default=None, help="Speed floor in knots for direction prior")
    p.add_argument("--no_first_neigh", action="store_true", help="Disable first-step 8-neighborhood constraint")
    p.add_argument("--allow_same_cell", action="store_true", help="Allow staying in the same cell (not recommended)")

    args = p.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.split_dir, "*_processed.pkl")))
    if args.mmsi.strip():
        allow = set(int(x) for x in args.mmsi.split(","))
        files = [f for f in files if parse_trip(f)[0] in allow]

    feat_dim = 4
    model = build_model(args.model, args.ckpt, feat_dim=feat_dim, horizon=args.horizon)

    # If TrAISformer, apply optional sampler overrides
    if args.model.lower() == "traisformer" and hasattr(model, "set_sampler"):
        model.set_sampler(
            lambda_cont=args.lambda_cont,
            alpha_dir=args.alpha_dir,
            beta_turn=args.beta_turn,
            step_scale=args.step_scale,
            sog_floor=args.sog_floor,
            first_step_neigh=(not args.no_first_neigh),
            forbid_same_cell=(not args.allow_same_cell),
        )

    metrics: List[Dict[str,Any]] = []
    plotted = 0; skipped = 0
    for idx, f in enumerate(files):
        try:
            trip = load_trip(f)
            res = evaluate_and_plot_trip(f, trip, model, args, sample_idx=idx)
            metrics.append(res)
            plotted += 1
            print(f"[ok] saved {res['png']}")
        except Exception as e:
            skipped += 1
            print(f"[skip] {os.path.basename(f)}: {e}")

    if metrics:
        sum_trips = Path(args.out_dir) / "summary_trips.csv"
        with open(sum_trips, "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=list(metrics[0].keys()))
            w.writeheader(); w.writerows(metrics)

        by = {}
        for m in metrics:
            key = m["mmsi"]
            by.setdefault(key, []).append(m["ade_km"])
        rows = [{"mmsi": k, "n_trips": len(v), "ade_km_mean": float(np.mean(v)), "ade_km_median": float(np.median(v))}
                for k, v in by.items()]
        sum_mmsi = Path(args.out_dir) / "summary_by_mmsi.csv"
        with open(sum_mmsi, "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=["mmsi","n_trips","ade_km_mean","ade_km_median"])
            w.writeheader(); w.writerows(rows)

        all_ades = [m["ade_km"] for m in metrics]
        overall = {
            "n_trips": len(metrics),
            "ade_km_mean": float(np.mean(all_ades)),
            "ade_km_median": float(np.median(all_ades)),
            "fde_km_mean": float(np.mean([m["fde_km"] for m in metrics])),
        }
        sum_overall = Path(args.out_dir) / "summary_overall.csv"
        with open(sum_overall, "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=list(overall.keys()))
            w.writeheader(); w.writerow(overall)
        print(f"[summary] plotted={plotted} skipped={skipped} total_selected={len(files)}")
        print(f"[summary files] {sum_trips} | {sum_mmsi} | {sum_overall}")
    else:
        print(f"[summary] plotted=0 skipped={skipped} total_selected={len(files)}")

if __name__ == "__main__":
    main()
