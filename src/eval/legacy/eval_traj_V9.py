# src/eval/eval_traj_V9.py
from __future__ import annotations
import matplotlib
matplotlib.use('Agg') # Fix for running without a display/GUI
import argparse, os, glob, pickle, csv, datetime as dt
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import torch
import matplotlib.pyplot as plt

# ---------------- Models ----------------
from src.models.traisformer_old import TrAISformer, BinSpec
# Use V2 as requested
try:
    from src.models.tptrans_gru import TPTrans
except ImportError:
    print("[Warning] Could not import TPTrans from src.models.tptrans_V2, falling back to src.models.tptrans")
    from src.models.legacy.tptrans import TPTrans

# ---------------- Preprocessing ----------------
from src.preprocessing.preprocessing import de_normalize_track, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, SPEED_MAX

# ---------------- Water mask (background only) ----------------
from src.utils.build_water_mask import make_water_mask

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
    "axes.facecolor": "#F8F9FA", # Light grey background
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
    parts = base.split("_")
    if len(parts) >= 2:
        return int(parts[0]), int(parts[1])
    return 0, 0

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
        return (5.0, 17.0, 54.0, 59.0)
    lat_min, lat_max = float(np.min(lats)), float(np.max(lats))
    lon_min, lon_max = float(np.min(lons)), float(np.max(lons))
    
    # Ensure minimum extent to avoid ultra-zoom on short lines
    min_span = 0.05 
    if (lat_max - lat_min) < min_span:
        mid = (lat_max + lat_min) / 2
        lat_min = mid - min_span/2
        lat_max = mid + min_span/2
    if (lon_max - lon_min) < min_span:
        mid = (lon_max + lon_min) / 2
        lon_min = mid - min_span/2
        lon_max = mid + min_span/2
        
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

def clean_state_dict(sd):
    if "state_dict" in sd: sd = sd["state_dict"]
    if "model" in sd and isinstance(sd["model"], dict): sd = sd["model"]
    new = {}
    for k, v in sd.items():
        nk = k[7:] if k.startswith("module.") else k
        new[nk] = v
    return new

def load_bins_from_ckpt(sd) -> BinSpec:
    if isinstance(sd, dict):
        if "bins" in sd and isinstance(sd["bins"], BinSpec): return sd["bins"]
        if "bins" in sd and isinstance(sd["bins"], dict): return BinSpec(**sd["bins"])
        flat = ["lat_min","lat_max","lon_min","lon_max","sog_max","n_lat","n_lon","n_sog","n_cog"]
        if all(k in sd for k in flat): return BinSpec(**{k: sd[k] for k in flat})
    raise KeyError("Could not find BinSpec in checkpoint.")

def build_model(kind: str, ckpt: str, feat_dim: int, horizon: int):
    sd_top = torch.load(ckpt, map_location="cpu")
    clean_sd = clean_state_dict(sd_top)
    
    meta = {
        "scale_factor": sd_top.get("scale_factor", 100.0), # Default to 100 if missing
        "norm_config": sd_top.get("norm_config", None),
        "bins": None
    }

    if kind.lower() == "tptrans":
        # Auto-detect dimensions
        d_model = 512 # Default for V2
        if 'proj.weight' in clean_sd:
            d_model = clean_sd['proj.weight'].shape[1]
        elif 'conv.net.0.weight' in clean_sd:
            d_model = clean_sd['conv.net.0.weight'].shape[0]
            
        print(f"[Model] Detected TPTrans configuration: d_model={d_model}")
        
        # V2 defaults: nhead=8, enc_layers=4, dec_layers=2
        # We try to detect these if possible, or use safe defaults
        # If the checkpoint has config, we should use it, but here we infer from weights or use defaults
        
        # Check nhead from encoder layer
        nhead = 8
        if 'encoder.layers.0.self_attn.in_proj_weight' in clean_sd:
             # shape is [3 * d_model, d_model]
             # This doesn't directly give nhead.
             pass
             
        # Check layers
        enc_layers = 0
        while f'encoder.layers.{enc_layers}.linear1.weight' in clean_sd:
            enc_layers += 1
        if enc_layers == 0: enc_layers = 4 # Fallback
        
        dec_layers = 0
        while f'dec.weight_ih_l{dec_layers}' in clean_sd:
            dec_layers += 1
        if dec_layers == 0: dec_layers = 2 # Fallback
        
        print(f"[Model] Inferred architecture: enc_layers={enc_layers}, dec_layers={dec_layers}")

        model = TPTrans(feat_dim=feat_dim, d_model=d_model, nhead=nhead, enc_layers=enc_layers, dec_layers=dec_layers, horizon=horizon)
        try:
            model.load_state_dict(clean_sd, strict=True)
        except RuntimeError as e:
            print(f"[Warning] Strict loading failed, retrying with strict=False. Error: {e}")
            model.load_state_dict(clean_sd, strict=False)
        return model, meta

    if kind.lower() == "traisformer":
        bins = load_bins_from_ckpt(sd_top)
        meta["bins"] = bins
        d_model = sd_top.get("d_model", 512)
        nhead = sd_top.get("nhead", 8)
        num_layers = sd_top.get("num_layers", 8)
        dropout = sd_top.get("dropout", 0.1)
        
        model = TrAISformer(bins=bins, d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout)
        model.load_state_dict(clean_sd, strict=False)
        return model, meta
        
    raise ValueError(f"unknown model kind: {kind}")

def normalize_latlon(lat, lon):
    """Normalize lat/lon using preprocessing constants."""
    nlat = (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)
    nlon = (lon - LON_MIN) / (LON_MAX - LON_MIN)
    return nlat, nlon

# ---------------- Core per-trip evaluation ----------------
def evaluate_and_plot_trip(fpath: str, trip: np.ndarray, model, meta, args, sample_idx: int) -> Dict[str, Any]:
    mmsi, tid = parse_trip(fpath)
    
    # trip is normalized [N, 9] (or similar) from pickle
    
    # 1. Denormalize the WHOLE trip for ground truth / plotting using the official function
    # We need to ensure it has 9 columns for de_normalize_track
    if trip.shape[1] < 9:
        # Pad with zeros if needed (though usually it is 9)
        temp_trip = np.zeros((len(trip), 9), dtype=trip.dtype)
        temp_trip[:, :trip.shape[1]] = trip
        trip_deg_full = de_normalize_track(temp_trip)
    else:
        trip_deg_full = de_normalize_track(trip)
        
    # Extract denormalized coordinates
    full_lat_deg = trip_deg_full[:, 0]
    full_lon_deg = trip_deg_full[:, 1]
    
    # Keep normalized version for model input
    # We assume the input 'trip' IS normalized as per preprocessing.py
    full_lat_norm = trip[:, 0]
    full_lon_norm = trip[:, 1]
    full_sog_norm = trip[:, 2]
    full_cog_norm = trip[:, 3]

    past, future_true_all, cut = split_by_percent(trip, args.pred_cut)
    
    if len(past) < 2:
        raise ValueError("too short past")

    if args.pred_len is not None:
        N_future = int(args.pred_len)
    else:
        # Default: predict as much as we have ground truth for (or capped)
        if len(future_true_all) < 2:
             raise ValueError("too short future")
        N_future = len(future_true_all) if args.cap_future is None else min(len(future_true_all), int(args.cap_future))

    lats_past = full_lat_deg[:cut]; lons_past = full_lon_deg[:cut]
    cur_lat = float(lats_past[-1]); cur_lon = float(lons_past[-1])
    lats_true_eval = full_lat_deg[cut:cut+N_future]
    lons_true_eval = full_lon_deg[cut:cut+N_future]
    lats_true_all = full_lat_deg[cut:]; lons_true_all = full_lon_deg[cut:]

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = model.to(device).eval()

    if args.model.lower() == "tptrans":
        # --- TPTrans Logic ---
        # Input is NORMALIZED
        seq_in = np.stack([
            full_lat_norm[:cut],
            full_lon_norm[:cut],
            full_sog_norm[:cut],
            full_cog_norm[:cut]
        ], axis=1).astype(np.float32)
        
        curr_seq_in = seq_in.copy()
        
        # We will build the prediction list in DEGREES
        pred_lat_list = [cur_lat]
        pred_lon_list = [cur_lon]
        
        # Track current state in NORMALIZED space for the model
        curr_lat_norm = curr_seq_in[-1, 0]
        curr_lon_norm = curr_seq_in[-1, 1]
        
        steps_needed = N_future
        steps_generated = 0
        
        while steps_generated < steps_needed:
            Tin = min(args.past_len, len(curr_seq_in))
            X_in = curr_seq_in[-Tin:, :][None, ...]
            X_tensor = torch.from_numpy(X_in).to(device)

            with torch.no_grad():
                # Model outputs [B, Horizon, 2]
                out = model(X_tensor)[0].cpu().numpy()
            
            # 1. Un-scale deltas (model output -> Delta Degrees)
            # The model is trained to predict (Delta_Deg * scale_factor)
            scale_factor = float(meta.get("scale_factor", 100.0))
            out_deltas_deg = out / scale_factor
            
            # 2. Apply user scale factor (default 1.0)
            out_deltas_deg = out_deltas_deg * float(args.pred_scale)
            
            # Process this chunk of predictions
            chunk_len = len(out_deltas_deg)
            
            # We need to append to curr_seq_in for autoregression
            # But we also want to project to water in DEGREES
            
            new_rows_norm = []
            
            for k in range(chunk_len):
                dlat_deg = out_deltas_deg[k, 0]
                dlon_deg = out_deltas_deg[k, 1]
                
                # Current position in DEGREES
                # We must convert from the normalized state the model sees
                curr_lat_deg = curr_lat_norm * (args.lat_max - args.lat_min) + args.lat_min
                curr_lon_deg = curr_lon_norm * (args.lon_max - args.lon_min) + args.lon_min
                
                # Proposed new position in DEGREES
                cand_lat_deg = curr_lat_deg + dlat_deg
                cand_lon_deg = curr_lon_deg + dlon_deg
                
                prev_lat_deg = pred_lat_list[-1]
                prev_lon_deg = pred_lon_list[-1]
                
                # Project to water
                if is_water(cand_lat_deg, cand_lon_deg):
                    fix_lat_deg, fix_lon_deg = cand_lat_deg, cand_lon_deg
                else:
                    fix_lat_deg, fix_lon_deg = project_to_water(prev_lat_deg, prev_lon_deg, cand_lat_deg, cand_lon_deg)
                
                pred_lat_list.append(fix_lat_deg)
                pred_lon_list.append(fix_lon_deg)
                
                # Convert FIXED position back to NORMALIZED for the model history
                fix_lat_norm = (fix_lat_deg - args.lat_min) / (args.lat_max - args.lat_min)
                fix_lon_norm = (fix_lon_deg - args.lon_min) / (args.lon_max - args.lon_min)
                
                # Update current state
                curr_lat_norm = fix_lat_norm
                curr_lon_norm = fix_lon_norm
                
                # Append to history (using last known SOG/COG)
                # Note: TPTrans doesn't predict SOG/COG, so we just repeat the last one?
                # Or should we update SOG based on distance? 
                # For now, repeating last SOG/COG is standard for this model type unless it predicts them.
                last_sog = curr_seq_in[-1, 2]
                last_cog = curr_seq_in[-1, 3]
                
                new_rows_norm.append([fix_lat_norm, fix_lon_norm, last_sog, last_cog])
            
            # Append chunk to history
            new_rows_norm = np.array(new_rows_norm, dtype=np.float32)
            curr_seq_in = np.concatenate([curr_seq_in, new_rows_norm], axis=0)
            
            steps_generated += chunk_len
            if steps_generated >= steps_needed:
                break

        pred_lat = np.array(pred_lat_list)
        pred_lon = np.array(pred_lon_list)

    else:
        # Placeholder for TrAISformer
        pred_lat = np.zeros(N_future + 1) + cur_lat 
        pred_lon = np.zeros(N_future + 1) + cur_lon

    # Trim by distance if requested
    if args.match_distance and len(pred_lat) > 1 and len(lats_true_eval) > 1:
        # Calculate true distance
        dt_true = cumdist(lats_true_eval, lons_true_eval)[-1]
        # Calculate pred distance (ignoring the start point for cumulative dist check relative to start)
        cd = cumdist(pred_lat, pred_lon)
        keep = int(np.searchsorted(cd, dt_true, side="right"))
        keep = max(2, min(keep, len(pred_lat))) # ensure at least 2 points (start + 1 pred)
        pred_lat, pred_lon = pred_lat[:keep], pred_lon[:keep]

    # Metrics
    # NOTE: pred_lat[0] is t=0 (current position). pred_lat[1] is t+1 (first prediction).
    # lats_true_eval starts at t+1.
    # So we compare pred_lat[1:] with lats_true_eval.
    
    preds_to_eval_lat = pred_lat[1:]
    preds_to_eval_lon = pred_lon[1:]
    
    n_comp = min(len(preds_to_eval_lat), len(lats_true_eval))
    
    if n_comp > 0:
        ade = float(np.mean([haversine_km(lats_true_eval[i], lons_true_eval[i], preds_to_eval_lat[i], preds_to_eval_lon[i]) for i in range(n_comp)]))
        fde = float(haversine_km(lats_true_eval[n_comp-1], lons_true_eval[n_comp-1], preds_to_eval_lat[n_comp-1], preds_to_eval_lon[n_comp-1]))
    else:
        ade, fde = np.nan, np.nan

    # ---- Plot ----
    fname_png = "skipped"
    if not args.no_plots:
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
        
        # Background
        drew_bg = False
        if getattr(args, "style", "classic") != "classic" and _HAS_CTX and not args.no_tiles:
            try:
                src = xz.Esri.WorldImagery if args.style == "satellite" else xz.OpenStreetMap.Mapnik
                ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)
                ctx.add_basemap(ax, source=src, crs="EPSG:4326", attribution_size=4)
                drew_bg = True
            except: pass
        
        if not drew_bg:
            try:
                # Dynamic resolution to keep pixels roughly square on the map (Mercator-ish)
                mean_lat = 0.5 * (lat_min + lat_max)
                cos_lat = np.cos(np.radians(mean_lat))
                
                # Aspect ratio for imshow to correct for latitude stretching
                # aspect = y_scale / x_scale = 1 / cos(lat)
                aspect_geo = 1.0 / cos_lat
                
                # Grid resolution
                dlat = lat_max - lat_min
                dlon = lon_max - lon_min
                n_lat_pix = 300
                
                n_lon_pix = int(n_lat_pix * (dlon / dlat) * cos_lat)
                n_lon_pix = max(1, n_lon_pix) # Safety
                
                wm = make_water_mask(lat_min, lat_max, lon_min, lon_max, n_lat_pix, n_lon_pix)
                land = (~wm).astype(float)
                img = np.zeros((land.shape[0], land.shape[1], 4))
                img[...] = (0.85, 0.92, 0.96, 1.0) 
                img[land > 0.5] = (0.80, 0.80, 0.80, 1.0) 
                
                # Explicitly set origin='lower' to match make_water_mask logic (Lat min at index 0)
                ax.imshow(img, extent=(lon_min, lon_max, lat_min, lat_max), origin="lower", aspect=aspect_geo)
            except Exception as e:
                print(f"[Plot Warning] Could not generate water mask: {e}")

        # Force Lat/Lon Labels and Grid
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.grid(True, alpha=0.5, linestyle='--')

        ax.plot(lons_past, lats_past, lw=2.0, color="#2a77ff", label="past")
        ax.plot(lons_true_all, lats_true_all, lw=2.5, color="#2aaa2a", label="true")
        
        if len(pred_lon) > 1:
            # pred_lon includes the start point, so this will draw a connected line
            ax.plot(pred_lon, pred_lat, lw=2.5, color="#d33", label="pred")
        
        ax.scatter([lons_past[-1]], [lats_past[-1]], s=40, c="#d33", edgecolors="k", zorder=5)
        
        ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)
        ax.legend()
        ax.set_title(f"MMSI {mmsi} ADE {ade:.2f} km FDE {fde:.2f} km")
        fig.tight_layout()
        fig.savefig(fname_png, dpi=args.dpi)
        plt.close(fig)

    return {
        "mmsi": mmsi, "trip": tid, "cut_idx": cut,
        "ade_km": ade, "fde_km": fde,
        "png": str(fname_png)
    }

# ---------------- Main ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split_dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--past_len", type=int, default=64)
    p.add_argument("--pred_cut", type=float, default=80.0)
    p.add_argument("--lat_min", type=float, default=None)
    p.add_argument("--lat_max", type=float, default=None)
    p.add_argument("--lon_min", type=float, default=None)
    p.add_argument("--lon_max", type=float, default=None)
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--match_distance", action="store_true")
    p.add_argument("--auto_extent", action="store_true")
    p.add_argument("--style", default="classic")
    p.add_argument("--dpi", type=int, default=180)
    
    # New Args
    p.add_argument("--pred_scale", type=float, default=1.0, help="Multiply predictions by this factor")
    p.add_argument("--no_tiles", action="store_true", help="Disable satellite/map tiles")
    p.add_argument("--pred_len", type=int, default=None, help="Force prediction length (steps). If None, uses ground truth length.")
    
    # TrAISformer dummies
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=60)
    p.add_argument("--cap_future", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--mmsi", default="") 
    p.add_argument("--speed_max", type=float, default=30.0)

    args = p.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.split_dir, "*_processed.pkl")))
    
    # Filter by MMSI if provided
    if args.mmsi:
        target_mmsis = str(args.mmsi).strip().split(',')
        files = [f for f in files if any(os.path.basename(f).startswith(m + "_") for m in target_mmsis)]
        print(f"[Eval] Filtering for MMSIs {target_mmsis}, found {len(files)} trips.")

    feat_dim = 4
    model, meta = build_model(args.model, args.ckpt, feat_dim, args.horizon)
    
    print(f"[Eval] Loaded model with scale_factor={meta.get('scale_factor', 'N/A')}")
    
    # --- Auto-fill bounds from checkpoint if missing ---
    norm_config = meta.get("norm_config")
    if norm_config:
        print(f"[Eval] Found norm_config in checkpoint: {norm_config}")
        if args.lat_min is None: args.lat_min = norm_config.get("LAT_MIN")
        if args.lat_max is None: args.lat_max = norm_config.get("LAT_MAX")
        if args.lon_min is None: args.lon_min = norm_config.get("LON_MIN")
        if args.lon_max is None: args.lon_max = norm_config.get("LON_MAX")
        if "SOG_MAX" in norm_config:
             args.speed_max = norm_config.get("SOG_MAX")
    
    # Fallback to defaults if still None (from preprocessing.py)
    if args.lat_min is None: args.lat_min = 54.0
    if args.lat_max is None: args.lat_max = 59.0
    if args.lon_min is None: args.lon_min = 5.0
    if args.lon_max is None: args.lon_max = 17.0
    


    print(f"[Eval] Using bounds: Lat [{args.lat_min}, {args.lat_max}], Lon [{args.lon_min}, {args.lon_max}], Speed Max {args.speed_max}")

    metrics = []
    for idx, f in enumerate(files):
        try:
            trip = load_trip(f)
            res = evaluate_and_plot_trip(f, trip, model, meta, args, idx)
            metrics.append(res)
            if not args.no_plots:
                print(f"[ok] saved {res['png']}")
            else:
                if idx % 50 == 0: print(f"Processed {idx}/{len(files)}")
        except Exception as e:
            print(f"[skip] {os.path.basename(f)}: {e}")

    if metrics:
        out_csv = Path(args.out_dir) / "metrics.csv"
        with open(out_csv, "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=list(metrics[0].keys()))
            w.writeheader()
            w.writerows(metrics)
        
        ades = [m["ade_km"] for m in metrics]
        print(f"Mean ADE: {np.mean(ades):.2f} km")
        print(f"Saved metrics to {out_csv}")

if __name__ == "__main__":
    main()
