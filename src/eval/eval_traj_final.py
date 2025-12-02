# src/eval/eval_traj_final.py
from __future__ import annotations
import matplotlib
matplotlib.use('Agg') 
import argparse, os, glob, pickle, csv, sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import torch
import matplotlib.pyplot as plt
import joblib
from matplotlib.lines import Line2D

# Maps
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import folium

# ---------------- Models ----------------
# Import both model architectures
try:
    from src.models.traisformer3 import TrAISformer, BinSpec
except ImportError:
    from src.models.traisformer1 import TrAISformer, BinSpec # Fallback

try:
    from src.models.tptrans_V3 import TPTrans
except ImportError:
    from src.models.tptrans import TPTrans

# Kalman
from src.models.kalman_filter.baselines.train_kalman import evaluate_trip_kalman, Bounds
from src.models.kalman_filter.kalman_filter import KalmanFilterParams

# ---------------- Water mask ----------------
from src.utils.water_guidance import is_water, project_to_water

# ---------------- Config ----------------
STYLE = {
    "past": "#1f77b4", "true": "#2ca02c", "pred": "#d62728",
    "land": "#E0E0E0", "edge": "#505050", "water": "#FFFFFF", "grid": "#B0B0B0"
}

plt.rcParams.update({
    "figure.figsize": (8.0, 6.5), "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333", "font.family": "sans-serif", 
    "axes.facecolor": STYLE["water"]
})

# ---------------- Helpers ----------------

def compute_horizon_metrics(lats_true, lons_true, lats_pred, lons_pred, 
                           horizon_steps=[12, 24, 36]) -> Dict[str, Optional[float]]:
    """
    Computes ADE at specific time steps (e.g. 1h=12 steps).
    Assumes approx 5 min sampling.
    """
    metrics = {}
    n_min = min(len(lats_true), len(lats_pred))
    
    for h in horizon_steps:
        key = f"ade_{h//12}h_km"
        if n_min > h:
            # Mean error from step 0 to step h
            errors = [haversine_km(lats_true[i], lons_true[i], lats_pred[i], lons_pred[i]) 
                     for i in range(h + 1)]
            metrics[key] = float(np.mean(errors))
        else:
            metrics[key] = None
    return metrics

def is_path_safe(lat1, lon1, lat2, lon2, steps=5):
    """Ray-casting: Checks if straight line crosses land."""
    for i in range(1, steps + 1):
        frac = i / steps
        test_lat = lat1 + (lat2 - lat1) * frac
        test_lon = lon1 + (lon2 - lon1) * frac
        if not is_water(test_lat, test_lon):
            return False, (test_lat, test_lon)
    return True, None

def parse_trip(fname: str) -> Tuple[int, int]:
    base = os.path.basename(fname).replace("_processed.pkl", "")
    parts = base.split("_")
    if len(parts) >= 2: return int(parts[0]), int(parts[1])
    return 0, 0

def load_trip(path: str, min_points: int = 30) -> np.ndarray:
    try: data = joblib.load(path)
    except: 
        with open(path, "rb") as f: data = pickle.load(f)
    trip = data["traj"] if isinstance(data, dict) and "traj" in data else np.asarray(data)
    trip = np.asarray(trip)
    if len(trip) < int(min_points): raise ValueError(f"too short: {len(trip)}")
    # Sort by time
    ts = trip[:, 7]
    if not np.all(ts[:-1] <= ts[1:]): trip = trip[np.argsort(ts)]
    return trip

def split_by_percent(trip: np.ndarray, pct: float) -> Tuple[np.ndarray, np.ndarray, int]:
    n = len(trip)
    cut = max(1, min(n - 2, int(round(n * pct / 100.0))))
    return trip[:cut], trip[cut:], cut

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1 = np.radians([lat1, lon1]); p2 = np.radians([lat2, lon2])
    dlat = p2[0]-p1[0]; dlon = p2[1]-p1[1]
    a = np.sin(dlat/2.0)**2 + np.cos(p1[0])*np.cos(p2[0])*np.sin(dlon/2.0)**2
    return float(2*R*np.arcsin(np.sqrt(a)))

def clean_state_dict(sd):
    if "state_dict" in sd: sd = sd["state_dict"]
    new = {}
    for k, v in sd.items():
        nk = k[7:] if k.startswith("module.") else k
        new[nk] = v
    return new

def load_bins_from_ckpt(sd) -> BinSpec:
    if isinstance(sd, dict):
        if "bins" in sd: return sd["bins"] if isinstance(sd["bins"], BinSpec) else BinSpec(**sd["bins"])
        flat = ["lat_min","lat_max","lon_min","lon_max","sog_max","n_lat","n_lon","n_sog","n_cog"]
        if all(k in sd for k in flat): return BinSpec(**{k: sd[k] for k in flat})
    raise KeyError("Could not find BinSpec in checkpoint.")

# ---------------- Model Builder ----------------
def build_model(kind: str, ckpt: str, feat_dim: int, horizon: int):
    """
    Unified loader for TPTrans, TrAISformer, and Kalman metadata.
    """
    if kind.lower() == "kalman":
        return None, {"kalman_params": KalmanFilterParams(), "norm_config": None}

    sd_top = torch.load(ckpt, map_location="cpu")
    clean_sd = clean_state_dict(sd_top)
    
    meta = {
        "scale_factor": sd_top.get("scale_factor", 100.0), 
        "norm_config": sd_top.get("norm_config", None),
        "data_bounds": sd_top.get("data_bounds", None),
        "bins": None
    }

    if kind.lower() == "tptrans":
        # Heuristic to find params
        d_model = 512
        if 'proj.weight' in clean_sd: d_model = clean_sd['proj.weight'].shape[1]
        
        enc_layers = 0
        while f'encoder.layers.{enc_layers}.linear1.weight' in clean_sd: enc_layers += 1
        if enc_layers == 0: enc_layers = 4 
        
        dec_layers = 0
        if 'dec.weight_ih_l0' in clean_sd: # V2 GRU
             while f'dec.weight_ih_l{dec_layers}' in clean_sd: dec_layers += 1
        else: # V3 Transformer
             while f'decoder.layers.{dec_layers}.linear1.weight' in clean_sd: dec_layers += 1
        if dec_layers == 0: dec_layers = 2 

        print(f"[TPTrans] d_model={d_model}, enc={enc_layers}, dec={dec_layers}")
        model = TPTrans(feat_dim=feat_dim, d_model=d_model, nhead=8, 
                        enc_layers=enc_layers, dec_layers=dec_layers, horizon=horizon)
        model.load_state_dict(clean_sd, strict=False)
        return model, meta

    if kind.lower() == "traisformer":
        bins = load_bins_from_ckpt(sd_top)
        meta["bins"] = bins
        
        cfg = sd_top.get("config", {}).get("model", {})
        d_model = cfg.get("d_model", sd_top.get("d_model", 512))
        nhead = cfg.get("nhead", sd_top.get("nhead", 8))
        num_layers = cfg.get("num_layers", sd_top.get("num_layers", 8))
        
        print(f"[TrAISformer] d_model={d_model}, layers={num_layers}")
        model = TrAISformer(bins=bins, d_model=d_model, nhead=nhead, num_layers=num_layers,
                            emb_lat=cfg.get("emb_lat", 128), emb_lon=cfg.get("emb_lon", 128),
                            emb_sog=cfg.get("emb_sog", 64), emb_cog=cfg.get("emb_cog", 64))
        model.load_state_dict(clean_sd, strict=False)
        return model, meta
        
    raise ValueError(f"Unknown model kind: {kind}")

# ---------------- Core Eval Logic ----------------
def evaluate_trip_wrapper(fpath: str, trip: np.ndarray, model, meta, args, sample_idx: int) -> Dict[str, Any]:
    mmsi, tid = parse_trip(fpath)
    
    # --- 1. Data Prep ---
    # Convert normalized [0,1] back to degrees for processing
    full_lat = trip[:, 0] * (args.lat_max - args.lat_min) + args.lat_min
    full_lon = trip[:, 1] * (args.lon_max - args.lon_min) + args.lon_min
    # Need normalized versions for TPTrans input
    full_lat_norm = trip[:, 0]
    full_lon_norm = trip[:, 1]
    full_sog_norm = trip[:, 2]
    full_cog_norm = trip[:, 3]
    
    full_sog_kn = full_sog_norm * args.speed_max
    full_cog_deg = full_cog_norm * 360.0

    past, future_true_all, cut = split_by_percent(trip, args.pred_cut)
    
    # Define Future Length
    if args.pred_len: N_future = int(args.pred_len)
    else: N_future = len(future_true_all) if not args.cap_future else min(len(future_true_all), int(args.cap_future))
    
    if len(past) < 2: return None

    # Reference points
    cur_lat, cur_lon = full_lat[cut-1], full_lon[cut-1]
    
    # True Future (for metrics)
    lats_true = full_lat[cut:cut+N_future]
    lons_true = full_lon[cut:cut+N_future]

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # --- 2. Model Prediction Branches ---
    pred_lats_deg = []
    pred_lons_deg = []

    # Branch A: Kalman
    if args.model.lower() == "kalman":
        window = max(1, min(int(args.past_len), cut))
        start_idx = cut - window
        # Slice trip for Kalman (needs all columns usually)
        kf_slice = trip[start_idx : cut + N_future]
        bounds = Bounds(args.lat_min, args.lat_max, args.lon_min, args.lon_max)
        eval_res = evaluate_trip_kalman(
            kf_slice, window_size=window, horizon=N_future,
            kf_params=meta["kalman_params"], bounds=bounds
        )
        # Result is [N, 2]
        pred_lats_deg = eval_res["pred_real"][:, 0]
        pred_lons_deg = eval_res["pred_real"][:, 1]

    # Branch B: TPTrans (Continuous Regression + RayCast + Smooth)
    elif args.model.lower() == "tptrans":
        model = model.to(device).eval()
        
        # Prepare Input
        seq_in = np.stack([full_lat_norm[:cut], full_lon_norm[:cut], 
                           full_sog_norm[:cut], full_cog_norm[:cut]], axis=1).astype(np.float32)
        curr_seq_in = seq_in.copy()
        
        res_lats = [cur_lat]
        res_lons = [cur_lon]
        curr_lat_n, curr_lon_n = curr_seq_in[-1, 0], curr_seq_in[-1, 1]
        
        steps_generated = 0
        while steps_generated < N_future:
            Tin = min(args.past_len, len(curr_seq_in))
            X_tensor = torch.from_numpy(curr_seq_in[-Tin:, :][None, ...]).to(device) # [1, T, 4]
            
            with torch.no_grad():
                out = model(X_tensor)[0].cpu().numpy() # [Horizon, 2]
            
            # Unscale Delta
            scale_factor = float(meta.get("scale_factor", 100.0))
            out_deltas = (out / scale_factor) * args.pred_scale
            
            # Apply Deltas Iteratively
            chunk_new_rows = []
            for k in range(len(out_deltas)):
                dlat, dlon = out_deltas[k]
                
                # Denormalize current to add delta
                c_lat = curr_lat_n * (args.lat_max - args.lat_min) + args.lat_min
                c_lon = curr_lon_n * (args.lon_max - args.lon_min) + args.lon_min
                
                cand_lat = c_lat + dlat
                cand_lon = c_lon + dlon
                
                # Anti-Grounding Check
                prev_l, prev_o = res_lats[-1], res_lons[-1]
                safe, _ = is_path_safe(prev_l, prev_o, cand_lat, cand_lon)
                
                if safe:
                    fix_lat, fix_lon = cand_lat, cand_lon
                else:
                    fix_lat, fix_lon = project_to_water(prev_l, prev_o, cand_lat, cand_lon)
                
                res_lats.append(fix_lat)
                res_lons.append(fix_lon)
                
                # Renormalize for next step input
                curr_lat_n = (fix_lat - args.lat_min) / (args.lat_max - args.lat_min)
                curr_lon_n = (fix_lon - args.lon_min) / (args.lon_max - args.lon_min)
                
                # Append (assuming SOG/COG stay constant for now or use model output if 4 dim)
                chunk_new_rows.append([curr_lat_n, curr_lon_n, curr_seq_in[-1, 2], curr_seq_in[-1, 3]])
            
            curr_seq_in = np.concatenate([curr_seq_in, np.array(chunk_new_rows, dtype=np.float32)], axis=0)
            steps_generated += len(out_deltas)

        pred_lats_deg = np.array(res_lats[1:]) # Drop start point
        pred_lons_deg = np.array(res_lons[1:])

        # Smoothing Kernel (Reduce Jitter)
        if len(pred_lats_deg) > 4:
            kernel = np.ones(3) / 3.0
            pred_lats_deg = np.convolve(np.pad(pred_lats_deg, (1,1), 'edge'), kernel, 'valid')
            pred_lons_deg = np.convolve(np.pad(pred_lons_deg, (1,1), 'edge'), kernel, 'valid')
            
        pred_lats_deg = pred_lats_deg[:N_future]
        pred_lons_deg = pred_lons_deg[:N_future]

    # Branch C: TrAISformer (Discrete Autoregressive)
    elif args.model.lower() == "traisformer":
        model = model.to(device).eval()
        bins = meta["bins"]
        
        # Prepare Binned Input
        # Get past data truncated to window size
        p_slice = slice(max(0, cut - args.past_len), cut)
        past_idxs = {
            "lat": bins.lat_to_bin(torch.tensor(full_lat[p_slice]).float()).unsqueeze(0).to(device),
            "lon": bins.lon_to_bin(torch.tensor(full_lon[p_slice]).float()).unsqueeze(0).to(device),
            "sog": bins.sog_to_bin(torch.tensor(full_sog_kn[p_slice]).float()).unsqueeze(0).to(device),
            "cog": bins.cog_to_bin(torch.tensor(full_cog_deg[p_slice]).float()).unsqueeze(0).to(device),
        }
        
        # Generation Loop
        curr_past = {k:v.clone() for k,v in past_idxs.items()}
        all_pred_idxs = {k:[] for k in curr_past}
        
        steps_generated = 0
        chunk_size = args.horizon 
        
        while steps_generated < N_future:
            step_len = min(chunk_size, N_future - steps_generated)
            with torch.no_grad():
                chunk_out = model.generate(
                    curr_past, L=step_len, 
                    sampling="sample" if args.temperature > 0 else "greedy",
                    temperature=args.temperature, top_k=args.top_k,
                    prevent_stuck=args.prevent_stuck
                )
            
            for k in all_pred_idxs: all_pred_idxs[k].append(chunk_out[k])
            
            # Slide Window
            for k in curr_past:
                curr_past[k] = torch.cat([curr_past[k], chunk_out[k]], dim=1)[:, -args.past_len:]
            
            steps_generated += step_len
            
        # Decode Bins -> Degrees
        out_lat = torch.cat(all_pred_idxs["lat"], dim=1).flatten().cpu()
        out_lon = torch.cat(all_pred_idxs["lon"], dim=1).flatten().cpu()
        
        raw_lats = bins.bin_to_lat_mid(out_lat).numpy()
        raw_lons = bins.bin_to_lon_mid(out_lon).numpy()
        
        # Ray Casting Fix on Discrete Jumps
        fixed_lats, fixed_lons = [], []
        curr_l, curr_o = cur_lat, cur_lon
        
        for k in range(len(raw_lats)):
            cand_l, cand_o = float(raw_lats[k]), float(raw_lons[k])
            safe, _ = is_path_safe(curr_l, curr_o, cand_l, cand_o)
            if safe:
                fix_l, fix_o = cand_l, cand_o
            else:
                fix_l, fix_o = project_to_water(curr_l, curr_o, cand_l, cand_o)
            
            fixed_lats.append(fix_l)
            fixed_lons.append(fix_o)
            curr_l, curr_o = fix_l, fix_o
            
        pred_lats_deg = np.array(fixed_lats)
        pred_lons_deg = np.array(fixed_lons)

    # --- 3. Compute Metrics ---
    # Crop to shortest common length
    n_eval = min(len(lats_true), len(pred_lats_deg))
    if n_eval == 0: return None
    
    # Standard ADE/FDE
    errors = [haversine_km(lats_true[i], lons_true[i], pred_lats_deg[i], pred_lons_deg[i]) for i in range(n_eval)]
    ade = np.mean(errors)
    fde = errors[-1]
    
    # Horizon Metrics (1h, 2h, 3h)
    h_metrics = compute_horizon_metrics(lats_true, lons_true, pred_lats_deg, pred_lons_deg)

    res = {
        "mmsi": mmsi, "trip": tid,
        "ade_km": float(ade), "fde_km": float(fde),
        **h_metrics,
        "plot_data": None,
        "raw_data": None
    }
    
    # --- 4. Plotting / Collection Data ---
    # Prepare data structure for plotting/collecting
    plot_data = {
        "mmsi": mmsi, "ade": ade, "fde": fde,
        "lats_past": full_lat[:cut], "lons_past": full_lon[:cut],
        "lats_true": np.concatenate(([cur_lat], lats_true)), 
        "lons_true": np.concatenate(([cur_lon], lons_true)),
        "lats_pred": np.concatenate(([cur_lat], pred_lats_deg[:n_eval])), 
        "lons_pred": np.concatenate(([cur_lon], pred_lons_deg[:n_eval]))
    }
    
    if args.collect:
        # Lighter version for simple storage
        res["raw_data"] = {
            "mmsi": mmsi,
            "past": np.stack([full_lat[:cut], full_lon[:cut]], axis=1),
            "true": np.stack([lats_true, lons_true], axis=1),
            "pred": np.stack([pred_lats_deg[:n_eval], pred_lons_deg[:n_eval]], axis=1)
        }
        
    if args.same_pic or args.folium or not args.no_plots:
        res["plot_data"] = plot_data

    # Individual Plot (if enabled)
    if not args.no_plots:
        do_individual_plot(mmsi, tid, plot_data, args, sample_idx)
        
    return res

def do_individual_plot(mmsi, tid, d, args, idx):
    outdir = Path(args.out_dir) / str(mmsi)
    outdir.mkdir(parents=True, exist_ok=True)
    fname = outdir / f"traj_{args.model}_{mmsi}_{tid}_{idx}.png"
    
    # Determine bounds
    all_lats = np.concatenate([d['lats_past'], d['lats_true'], d['lats_pred']])
    all_lons = np.concatenate([d['lons_past'], d['lons_true'], d['lons_pred']])
    pad = 0.1
    p_lat_min, p_lat_max = np.min(all_lats)-pad, np.max(all_lats)+pad
    p_lon_min, p_lon_max = np.min(all_lons)-pad*1.5, np.max(all_lons)+pad*1.5
    
    if args.plot_lat_min: 
        p_lat_min, p_lat_max = args.plot_lat_min, args.plot_lat_max
        p_lon_min, p_lon_max = args.plot_lon_min, args.plot_lon_max

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
    ax.set_extent([p_lon_min, p_lon_max, p_lat_min, p_lat_max], crs=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.GSHHSFeature(scale='h', levels=[1], facecolor=STYLE["land"], edgecolor=STYLE["edge"]))
    
    ax.plot(d['lons_past'], d['lats_past'], transform=ccrs.PlateCarree(), c=STYLE["past"], lw=2, label="Past")
    ax.plot(d['lons_true'], d['lats_true'], transform=ccrs.PlateCarree(), c=STYLE["true"], lw=3, label="True")
    ax.plot(d['lons_pred'], d['lats_pred'], transform=ccrs.PlateCarree(), c=STYLE["pred"], lw=3, ls="--", label="Pred")
    
    ax.legend()
    ax.set_title(f"MMSI {mmsi} | ADE {d['ade']:.2f}km")
    fig.savefig(fname, bbox_inches='tight', dpi=args.dpi)
    plt.close(fig)


## folium map generation
def generate_folium_map(plot_data_list, out_dir, args):
    print("-" * 40)
    print(f"Generating Interactive Map for {len(plot_data_list)} trips...")

    all_lats = []
    all_lons = []
    for d in plot_data_list:
        all_lats.extend(d['lats_past'])
        all_lons.extend(d['lons_past'])
    
    if not all_lats: return

    lat_center = np.mean(all_lats)
    lon_center = np.mean(all_lons)
    
    m = folium.Map(location=[lat_center, lon_center], zoom_start=6, tiles=None)

    folium.TileLayer(tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", attr="&copy; OpenStreetMap", name="OSM Streets").add_to(m)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attr="Esri World Imagery", name="Esri Satellite").add_to(m)

    for d in plot_data_list: 
        tooltip_txt = f"MMSI: {d['mmsi']} | ADE: {d['ade']:.2f} km | FDE: {d['fde']:.2f} km | len: {(len(d['lats_true'])*5/60):.1f} h"
        
        folium.PolyLine(locations=list(zip(d['lats_past'], d['lons_past'])), color=STYLE["past"], weight=2.5, opacity=0.8, tooltip=tooltip_txt + " (Past)").add_to(m)
        folium.PolyLine(locations=list(zip(d['lats_true'], d['lons_true'])), color=STYLE["true"], weight=3, opacity=0.8, tooltip=tooltip_txt + " (True)").add_to(m)
        if len(d['lats_pred']) > 1:
            folium.PolyLine(locations=list(zip(d['lats_pred'], d['lons_pred'])), color=STYLE["pred"], weight=3, opacity=0.9, tooltip=tooltip_txt + " (Pred)").add_to(m)
            

    folium.LayerControl().add_to(m)
    out_file = Path(out_dir) / "interactive_map.html"
    m.save(out_file)
    print(f"Interactive map saved to: {out_file}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model", required=True, choices=["tptrans", "traisformer", "kalman"])
    p.add_argument("--ckpt", default=None, help="Required for DL models")
    
    # Data Bounds (Overrides checkpoint if set)
    p.add_argument("--lat_min", type=float, default=None)
    p.add_argument("--lat_max", type=float, default=None)
    p.add_argument("--lon_min", type=float, default=None)
    p.add_argument("--lon_max", type=float, default=None)
    p.add_argument("--speed_max", type=float, default=30.0)
    
    # Plotting Bounds
    p.add_argument("--plot_lat_min", type=float)
    p.add_argument("--plot_lat_max", type=float)
    p.add_argument("--plot_lon_min", type=float)
    p.add_argument("--plot_lon_max", type=float)

    # Eval Params
    p.add_argument("--past_len", type=int, default=64)
    p.add_argument("--pred_cut", type=float, default=80.0)
    p.add_argument("--pred_len", type=int, default=None)
    p.add_argument("--cap_future", type=int, default=None)
    p.add_argument("--horizon", type=int, default=12, help="Model horizon")
    p.add_argument("--pred_scale", type=float, default=1.0)
    
    # Generation (TrAISformer)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--prevent_stuck", action="store_true")

    # Outputs
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--same_pic", action="store_true")
    p.add_argument("--folium", action="store_true")
    p.add_argument("--collect", action="store_true", help="Save raw arrays for custom plotting")
    p.add_argument("--mmsi", default="")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--dpi", type=int, default=150)

    args = p.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # 1. Load Model
    feat_dim = 4
    model, meta = build_model(args.model, args.ckpt, feat_dim, args.horizon)
    
    # 2. Setup Bounds
    # Priority: CLI Args > Checkpoint Meta > Defaults
    d_bounds = meta.get("data_bounds") or meta.get("norm_config") or {}
    if args.lat_min is None: args.lat_min = d_bounds.get("LAT_MIN", 54.0)
    if args.lat_max is None: args.lat_max = d_bounds.get("LAT_MAX", 59.0)
    if args.lon_min is None: args.lon_min = d_bounds.get("LON_MIN", 5.0)
    if args.lon_max is None: args.lon_max = d_bounds.get("LON_MAX", 17.0)
    if "SOG_MAX" in d_bounds: args.speed_max = d_bounds["SOG_MAX"]

    print(f"[Config] Bounds: Lat [{args.lat_min}, {args.lat_max}], Lon [{args.lon_min}, {args.lon_max}]")
    
    files = sorted(glob.glob(os.path.join(args.split_dir, "*_processed.pkl")))
    if args.mmsi:
        targets = args.mmsi.split(',')
        files = [f for f in files if any(m in f for m in targets)]
    
    print(f"Evaluating {len(files)} trips...")

    metrics = []
    plot_collection = []
    raw_collection = []

    for idx, fpath in enumerate(files):
        try:
            trip = load_trip(fpath)
            res = evaluate_trip_wrapper(fpath, trip, model, meta, args, idx)
            if res:
                metrics.append(res)
                if res.get("plot_data"): plot_collection.append(res["plot_data"])
                if res.get("raw_data"): raw_collection.append(res["raw_data"])
            
            if idx % 50 == 0: print(f"Processed {idx}/{len(files)}")
        except Exception as e:
            print(f"Error {fpath}: {e}")

    # --- Save Metrics ---
    if metrics:
        out_csv = Path(args.out_dir) / "metrics.csv"
        # Filter out heavy data for CSV
        clean_metrics = [{k:v for k,v in m.items() if k not in ["plot_data", "raw_data"]} for m in metrics]
        
        # Calculate Aggregates
        ade_list = [m['ade_km'] for m in clean_metrics]
        fde_list = [m['fde_km'] for m in clean_metrics]
        
        print("-" * 40)
        print(f"Mean ADE: {np.mean(ade_list):.3f} km | Median: {np.median(ade_list):.3f}")
        print(f"Mean FDE: {np.mean(fde_list):.3f} km | Median: {np.median(fde_list):.3f}")
        mean_ade, median_ade = np.mean(ade_list), np.median(ade_list)
        mean_fde, median_fde = np.mean(fde_list), np.median(fde_list)
        
        # Horizon stats
        for h_key in ['ade_1h_km', 'ade_2h_km', 'ade_3h_km']:
            vals = [m[h_key] for m in clean_metrics if m.get(h_key) is not None]
            if vals:
                print(f"{h_key}: Mean {np.mean(vals):.3f} | Median {np.median(vals):.3f} (N={len(vals)})")

        with open(out_csv, "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=clean_metrics[0].keys())
            w.writeheader()
            w.writerows(clean_metrics)

    # --- Save Collection (New Feature) ---
    if args.collect and raw_collection:
        save_path = Path(args.out_dir) / "eval_collection.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(raw_collection, f)
        print(f"Saved raw collection data to {save_path}")


    # --- COMBINED PLOT GENERATION ---
    if args.same_pic and plot_collection:
        print("-" * 40)
        print("Generating combined plot...")
        
        if args.plot_lat_min is not None:
             p_lat_min, p_lat_max = args.plot_lat_min, args.plot_lat_max
             p_lon_min, p_lon_max = args.plot_lon_min, args.plot_lon_max
        else:
            all_lats = []
            all_lons = []
            for d in plot_collection:
                all_lats.extend([d['lats_past'], d['lats_true'], d['lats_pred']])
                all_lons.extend([d['lons_past'], d['lons_true'], d['lons_pred']])
            flat_lats = np.concatenate(all_lats)
            flat_lons = np.concatenate(all_lons)
            pad = 0.15
            p_lat_min, p_lat_max = np.min(flat_lats) - pad, np.max(flat_lats) + pad
            p_lon_min, p_lon_max = np.min(flat_lons) - pad * 1.5, np.max(flat_lons) + pad * 1.5

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
        ax.set_extent([p_lon_min, p_lon_max, p_lat_min, p_lat_max], crs=ccrs.PlateCarree())

        land_feature = cfeature.GSHHSFeature(scale='h', levels=[1], 
                                           facecolor=STYLE["land"], 
                                           edgecolor=STYLE["edge"],
                                           linewidth=0.5)
        ax.add_feature(land_feature)
        ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5, linewidth=0.5)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color=STYLE["grid"], alpha=0.5, linestyle='--')
        gl.top_labels = False; gl.right_labels = False
        
        n_plots = len(plot_collection)
        alpha_base = 0.6 if n_plots > 10 else 0.9
        lw_base = 1.0 if n_plots > 10 else 2.0
        
        for d in plot_collection:
            ax.plot(d['lons_past'], d['lats_past'], transform=ccrs.PlateCarree(), 
                    lw=lw_base, color=STYLE["past"], alpha=alpha_base, zorder=3)
            ax.plot(d['lons_true'], d['lats_true'], transform=ccrs.PlateCarree(), 
                    lw=lw_base, color=STYLE["true"], alpha=alpha_base, zorder=4)
            if len(d['lons_pred']) > 1:
                ax.plot(d['lons_pred'], d['lats_pred'], transform=ccrs.PlateCarree(), 
                        lw=lw_base, color=STYLE["pred"], alpha=0.8, zorder=5)
        
        legend_elements = [
            Line2D([0], [0], color=STYLE["past"], lw=2, label='Past'),
            Line2D([0], [0], color=STYLE["true"], lw=2, label='True'),
            Line2D([0], [0], color=STYLE["pred"], lw=2, label='Pred')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fancybox=True, framealpha=0.9)
        
        title_str = (f"Combined Trajectories (N={n_plots})\n"
                     f"Mean ADE: {mean_ade:.2f} km | Median ADE: {median_ade:.2f} km\n"
                     f"Mean FDE: {mean_fde:.2f} km | Median FDE: {median_fde:.2f} km")
        ax.set_title(title_str, fontsize=11, pad=12)

        out_name = Path(args.out_dir) / "combined_trajectories.png"
        fig.tight_layout()
        fig.savefig(out_name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Combined plot saved to: {out_name}")

    if args.folium and plot_collection:
        generate_folium_map(plot_collection, args.out_dir, args)


if __name__ == "__main__":
    main()