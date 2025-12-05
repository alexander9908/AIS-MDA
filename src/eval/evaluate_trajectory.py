# src/eval/evaluate_trajectory.py
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
import itertools

# Maps
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import folium

# ---------------- Models ----------------
try:
    from src.models.traisformer import TrAISformer, BinSpec
except ImportError:
    print("[Warning] Could not import TrAISformer,")

try:
    from src.models.tptrans_transformer import TPTrans
except ImportError:
    print("[Warning] Could not import TPTrans Transformer decoder, falling back to GRU decoder...")
    try:
        from src.models.tptrans_gru import TPTrans
    except ImportError:
        print("[Warning] Could not import TPTrans Transformer decoder")
    

# Kalman
from src.models.kalman_filter.baselines.train_kalman import evaluate_trip_kalman, Bounds
from src.models.kalman_filter.kalman_filter import KalmanFilterParams

# ---------------- Water mask ----------------
from src.utils.water_guidance import is_water, project_to_water

# ---------------- Config ----------------
# Distinct colors for different model predictions
PRED_COLORS = ["#d62728", "#9467bd", "#ff7f0e", "#17becf", "#e377c2"] # Red, Purple, Orange, Cyan, Pink

STYLE = {
    "past": "#1f77b4", "true": "#2ca02c", 
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
    metrics = {}
    n_min = min(len(lats_true), len(lats_pred))
    
    for h in horizon_steps:
        key_ade = f"ade_{h//12}h_km"
        key_fde = f"fde_{h//12}h_km"
        
        if n_min > h:
            errors = [haversine_km(lats_true[i], lons_true[i], lats_pred[i], lons_pred[i]) 
                     for i in range(h + 1)]
            metrics[key_ade] = float(np.mean(errors))
            metrics[key_fde] = float(errors[-1]) 
        else:
            metrics[key_ade] = None
            metrics[key_fde] = None
    return metrics

def is_path_safe(lat1, lon1, lat2, lon2, steps=5):
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

# ---------------- Model Loader ----------------
def load_one_model(model_name: str, ckpt_path: str, feat_dim: int, horizon: int):
    """Loads a single model instance."""
    if model_name.lower() == "kalman":
        return None, {"kalman_params": KalmanFilterParams(), "norm_config": None}

    print(f"Loading {model_name} from {ckpt_path}...")
    sd_top = torch.load(ckpt_path, map_location="cpu")
    clean_sd = clean_state_dict(sd_top)
    
    meta = {
        "scale_factor": sd_top.get("scale_factor", 100.0), 
        "norm_config": sd_top.get("norm_config", None),
        "data_bounds": sd_top.get("data_bounds", None),
        "bins": None
    }

    if model_name.lower() == "tptrans":
        d_model = 512
        if 'proj.weight' in clean_sd: d_model = clean_sd['proj.weight'].shape[1]
        enc_layers = 0
        while f'encoder.layers.{enc_layers}.linear1.weight' in clean_sd: enc_layers += 1
        if enc_layers == 0: enc_layers = 4 
        dec_layers = 0
        if 'dec.weight_ih_l0' in clean_sd: 
             while f'dec.weight_ih_l{dec_layers}' in clean_sd: dec_layers += 1
        else: 
             while f'decoder.layers.{dec_layers}.linear1.weight' in clean_sd: dec_layers += 1
        if dec_layers == 0: dec_layers = 2 

        model = TPTrans(feat_dim=feat_dim, d_model=d_model, nhead=8, 
                        enc_layers=enc_layers, dec_layers=dec_layers, horizon=horizon)
        model.load_state_dict(clean_sd, strict=False)
        return model, meta

    if model_name.lower() == "traisformer":
        bins = load_bins_from_ckpt(sd_top)
        meta["bins"] = bins
        cfg = sd_top.get("config", {}).get("model", {})
        d_model = cfg.get("d_model", sd_top.get("d_model", 512))
        nhead = cfg.get("nhead", sd_top.get("nhead", 8))
        num_layers = cfg.get("num_layers", sd_top.get("num_layers", 8))
        
        model = TrAISformer(bins=bins, d_model=d_model, nhead=nhead, num_layers=num_layers,
                            emb_lat=cfg.get("emb_lat", 128), emb_lon=cfg.get("emb_lon", 128),
                            emb_sog=cfg.get("emb_sog", 64), emb_cog=cfg.get("emb_cog", 64))
        model.load_state_dict(clean_sd, strict=False)
        return model, meta
        
    raise ValueError(f"Unknown model kind: {model_name}")

# ---------------- Core Prediction Logic ----------------
def run_prediction(trip, model, meta, args, model_name, N_future, cut, ref_lat, ref_lon):
    """Runs prediction for ONE model on ONE trip."""
    
    # Common Data Prep
    full_lat = trip[:, 0] * (args.lat_max - args.lat_min) + args.lat_min
    full_lon = trip[:, 1] * (args.lon_max - args.lon_min) + args.lon_min
    
    # 1. Kalman
    if model_name.lower() == "kalman":
        window = max(1, min(int(args.past_len), cut))
        start_idx = cut - window
        kf_slice = trip[start_idx : cut + N_future]
        bounds = Bounds(args.lat_min, args.lat_max, args.lon_min, args.lon_max)
        eval_res = evaluate_trip_kalman(
            kf_slice, window_size=window, horizon=N_future,
            kf_params=meta["kalman_params"], bounds=bounds
        )
        raw_lat, raw_lon = eval_res["pred_real"][:, 0], eval_res["pred_real"][:, 1]
        
        # Stop on Land
        valid_lats, valid_lons = [], []
        cl, co = ref_lat, ref_lon
        for k in range(len(raw_lat)):
            cand_l, cand_o = float(raw_lat[k]), float(raw_lon[k])
            safe, _ = is_path_safe(cl, co, cand_l, cand_o)
            if safe:
                valid_lats.append(cand_l); valid_lons.append(cand_o)
                cl, co = cand_l, cand_o
            else: break 
        return np.array(valid_lats), np.array(valid_lons)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = model.to(device).eval()

    # 2. TPTrans
    if model_name.lower() == "tptrans":
        seq_in = trip[:, :4].astype(np.float32) 
        
        curr_seq = seq_in[:cut].copy()
        res_lats, res_lons = [ref_lat], [ref_lon]
        curr_lat_n, curr_lon_n = curr_seq[-1, 0], curr_seq[-1, 1]
        
        steps_generated = 0
        while steps_generated < N_future:
            Tin = min(args.past_len, len(curr_seq))
            X_tensor = torch.from_numpy(curr_seq[-Tin:, :][None, ...]).to(device)
            with torch.no_grad(): out = model(X_tensor)[0].cpu().numpy()
            
            scale_factor = float(meta.get("scale_factor", 100.0))
            out_deltas = (out / scale_factor) * args.pred_scale
            
            chunk_rows = []
            for k in range(len(out_deltas)):
                dlat, dlon = out_deltas[k]
                c_lat = curr_lat_n * (args.lat_max - args.lat_min) + args.lat_min
                c_lon = curr_lon_n * (args.lon_max - args.lon_min) + args.lon_min
                cand_lat, cand_lon = c_lat + dlat, c_lon + dlon
                
                prev_l, prev_o = res_lats[-1], res_lons[-1]
                safe, _ = is_path_safe(prev_l, prev_o, cand_lat, cand_lon)
                if safe: fix_l, fix_o = cand_lat, cand_lon
                else: fix_l, fix_o = project_to_water(prev_l, prev_o, cand_lat, cand_lon)
                
                res_lats.append(fix_l); res_lons.append(fix_o)
                curr_lat_n = (fix_l - args.lat_min) / (args.lat_max - args.lat_min)
                curr_lon_n = (fix_o - args.lon_min) / (args.lon_max - args.lon_min)
                chunk_rows.append([curr_lat_n, curr_lon_n, curr_seq[-1, 2], curr_seq[-1, 3]])
            
            curr_seq = np.concatenate([curr_seq, np.array(chunk_rows, dtype=np.float32)], axis=0)
            steps_generated += len(out_deltas)
            
        pred_lats = np.array(res_lats[1:])
        pred_lons = np.array(res_lons[1:])
        # Smooth
        if len(pred_lats) > 4:
            k = np.ones(3)/3.0
            pred_lats = np.convolve(np.pad(pred_lats, (1,1), 'edge'), k, 'valid')
            pred_lons = np.convolve(np.pad(pred_lons, (1,1), 'edge'), k, 'valid')
        return pred_lats[:N_future], pred_lons[:N_future]

    # 3. TrAISformer
    if model_name.lower() == "traisformer":
        bins = meta["bins"]
        p_slice = slice(max(0, cut - args.past_len), cut)
        
        # Denorm for binning
        p_lat = full_lat[p_slice]
        p_lon = full_lon[p_slice]
        p_sog = trip[p_slice, 2] * args.speed_max
        p_cog = trip[p_slice, 3] * 360.0
        
        past_idxs = {
            "lat": bins.lat_to_bin(torch.tensor(p_lat).float()).unsqueeze(0).to(device),
            "lon": bins.lon_to_bin(torch.tensor(p_lon).float()).unsqueeze(0).to(device),
            "sog": bins.sog_to_bin(torch.tensor(p_sog).float()).unsqueeze(0).to(device),
            "cog": bins.cog_to_bin(torch.tensor(p_cog).float()).unsqueeze(0).to(device),
        }
        
        curr_past = {k:v.clone() for k,v in past_idxs.items()}
        all_pred = {k:[] for k in curr_past}
        steps = 0
        chunk = args.horizon
        
        while steps < N_future:
            slen = min(chunk, N_future - steps)
            with torch.no_grad():
                out = model.generate(curr_past, L=slen, 
                                     sampling="sample" if args.temperature > 0 else "greedy",
                                     temperature=args.temperature, top_k=args.top_k,
                                     prevent_stuck=args.prevent_stuck)
            for k in all_pred: all_pred[k].append(out[k])
            for k in curr_past:
                curr_past[k] = torch.cat([curr_past[k], out[k]], dim=1)[:, -args.past_len:]
            steps += slen
            
        out_lat = torch.cat(all_pred["lat"], dim=1).flatten().cpu()
        out_lon = torch.cat(all_pred["lon"], dim=1).flatten().cpu()
        raw_lats = bins.bin_to_lat_mid(out_lat).numpy()
        raw_lons = bins.bin_to_lon_mid(out_lon).numpy()
        
        fixed_lats, fixed_lons = [], []
        cl, co = ref_lat, ref_lon
        for k in range(len(raw_lats)):
            cand_l, cand_o = float(raw_lats[k]), float(raw_lons[k])
            safe, _ = is_path_safe(cl, co, cand_l, cand_o)
            if safe: fl, fo = cand_l, cand_o
            else: fl, fo = project_to_water(cl, co, cand_l, cand_o)
            fixed_lats.append(fl); fixed_lons.append(fo)
            cl, co = fl, fo
            
        return np.array(fixed_lats), np.array(fixed_lons)
    
    return np.array([]), np.array([])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split_dir", required=True)
    p.add_argument("--out_dir", required=True)
    
    # MULTI MODEL ARGS
    p.add_argument("--model", required=True, help="Comma separated models (e.g. tptrans,traisformer)")
    p.add_argument("--ckpt", default=None, help="Comma separated ckpts (e.g. path1.pt,path2.pt)")
    
    # Data Bounds
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
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--pred_scale", type=float, default=1.0)
    
    # Generation
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--prevent_stuck", action="store_true")

    # Outputs
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--same_pic", action="store_true")
    p.add_argument("--folium", action="store_true")
    p.add_argument("--collect", action="store_true")
    p.add_argument("--mmsi", default="")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--dpi", type=int, default=150)

    args = p.parse_args()
    
    # --- FIX: Convert out_dir to Path immediately ---
    args.out_dir = Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse Models
    model_names = [m.strip() for m in args.model.split(',')]
    ckpt_paths = [c.strip() for c in args.ckpt.split(',')] if args.ckpt else []
    
    # Validate Checkpoints
    non_kalman_count = sum(1 for m in model_names if m.lower() != 'kalman')
    if non_kalman_count > len(ckpt_paths):
        print(f"[Error] You requested {non_kalman_count} DL models but provided only {len(ckpt_paths)} checkpoints.")
        sys.exit(1)

    # Load All Models
    loaded_models = {}
    ckpt_idx = 0
    bounds_found = False
    
    feat_dim = 4
    for m_name in model_names:
        c_path = None
        if m_name.lower() != 'kalman':
            c_path = ckpt_paths[ckpt_idx]
            ckpt_idx += 1
        
        model, meta = load_one_model(m_name, c_path, feat_dim, args.horizon)
        loaded_models[m_name] = (model, meta)
        
        if not bounds_found:
            d_bounds = meta.get("data_bounds") or meta.get("norm_config")
            if d_bounds:
                if args.lat_min is None: args.lat_min = d_bounds.get("LAT_MIN", 54.0)
                if args.lat_max is None: args.lat_max = d_bounds.get("LAT_MAX", 59.0)
                if args.lon_min is None: args.lon_min = d_bounds.get("LON_MIN", 5.0)
                if args.lon_max is None: args.lon_max = d_bounds.get("LON_MAX", 17.0)
                if "SOG_MAX" in d_bounds: args.speed_max = d_bounds["SOG_MAX"]
                bounds_found = True

    if args.lat_min is None: args.lat_min = 54.0
    if args.lat_max is None: args.lat_max = 59.0
    if args.lon_min is None: args.lon_min = 5.0
    if args.lon_max is None: args.lon_max = 17.0
    
    print(f"[Config] Models: {model_names}")
    print(f"[Config] Bounds: Lat [{args.lat_min}, {args.lat_max}], Lon [{args.lon_min}, {args.lon_max}]")

    files = sorted(glob.glob(os.path.join(args.split_dir, "*_processed.pkl")))
    if args.mmsi:
        targets = args.mmsi.split(',')
        files = [f for f in files if any(m in f for m in targets)]

    print(f"Evaluating {len(files)} trips...")

    metrics_all = []
    plot_data_all = []
    collection_data_all = []

    for idx, fpath in enumerate(files):
        try:
            trip = load_trip(fpath)
            mmsi, tid = parse_trip(fpath)
            
            full_lat = trip[:, 0] * (args.lat_max - args.lat_min) + args.lat_min
            full_lon = trip[:, 1] * (args.lon_max - args.lon_min) + args.lon_min
            
            past, future_true_all, cut = split_by_percent(trip, args.pred_cut)
            if len(past) < 2: continue
            
            if args.pred_len: N_future = int(args.pred_len)
            else: N_future = len(future_true_all) if not args.cap_future else min(len(future_true_all), int(args.cap_future))
            
            cur_lat, cur_lon = full_lat[cut-1], full_lon[cut-1]
            lats_true = full_lat[cut:cut+N_future]
            lons_true = full_lon[cut:cut+N_future]
            
            # --- RUN ALL MODELS ON THIS TRIP ---
            trip_results = {}
            row_metrics = {
                "mmsi": mmsi, "trip": tid,
                "trip_len_h": round(len(lats_true)*5/60, 2)
            }
            
            for m_name, (model, meta) in loaded_models.items():
                p_lats, p_lons = run_prediction(trip, model, meta, args, m_name, N_future, cut, cur_lat, cur_lon)
                
                n_eval = min(len(lats_true), len(p_lats))
                if n_eval > 0:
                    errs = [haversine_km(lats_true[i], lons_true[i], p_lats[i], p_lons[i]) for i in range(n_eval)]
                    ade = np.mean(errs)
                    fde = errs[-1]
                    h_metrics = compute_horizon_metrics(lats_true, lons_true, p_lats, p_lons)
                    
                    row_metrics[f"{m_name}_ade"] = float(ade)
                    row_metrics[f"{m_name}_fde"] = float(fde)
                    for k, v in h_metrics.items():
                        row_metrics[f"{m_name}_{k}"] = v
                        
                    trip_results[m_name] = (p_lats, p_lons, ade, fde)
            
            metrics_all.append(row_metrics)
            
            p_data = {
                "mmsi": mmsi, "tid": tid,
                "past": (full_lat[:cut], full_lon[:cut]),
                "true": (np.concatenate(([cur_lat], lats_true)), np.concatenate(([cur_lon], lons_true))),
                "models": {}
            }
            
            for m_name, (plat, plon, ade, fde) in trip_results.items():
                p_data["models"][m_name] = {
                    "lats": np.concatenate(([cur_lat], plat[:len(lats_true)])),
                    "lons": np.concatenate(([cur_lon], plon[:len(lats_true)])),
                    "ade": ade, "fde": fde
                }
            
            plot_data_all.append(p_data)
            
            if args.collect:
                c_data = {
                    "mmsi": mmsi, "trip_id": tid,
                    "past": np.stack([full_lat[:cut], full_lon[:cut]], axis=1),
                    "true": np.stack([lats_true, lons_true], axis=1),
                    "models": {}
                }
                for m_name, (plat, plon, _, _) in trip_results.items():
                    c_data["models"][m_name] = np.stack([plat, plon], axis=1)
                collection_data_all.append(c_data)

            if not args.no_plots:
                make_single_plot(p_data, args, idx, model_names)

            if idx % 50 == 0: print(f"Processed {idx}/{len(files)}")

        except Exception as e:
            print(f"Error {fpath}: {e}")
            import traceback
            traceback.print_exc()

    # --- Saving ---
    suffix = "_".join(model_names)
    
    if metrics_all:
        csv_name = args.out_dir / f"metrics_{suffix}.csv"
        
        # --- FIX: Explicitly define all possible columns ---
        base_cols = ["mmsi", "trip", "trip_len_h"]
        metric_types = ["ade", "fde", 
                        "ade_1h_km", "fde_1h_km", 
                        "ade_2h_km", "fde_2h_km", 
                        "ade_3h_km", "fde_3h_km"]
        
        model_cols = []
        for m_name in model_names:
            for mt in metric_types:
                model_cols.append(f"{m_name}_{mt}")
                
        all_fieldnames = base_cols + model_cols
        
        with open(csv_name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_fieldnames)
            w.writeheader()
            w.writerows(metrics_all)
        print(f"Saved metrics to {csv_name}")
        
    if args.collect and collection_data_all:
        pkl_name = args.out_dir / f"predictions_{suffix}.pkl"
        with open(pkl_name, "wb") as f:
            pickle.dump(collection_data_all, f)
        print(f"Saved predictions to {pkl_name}")
        
    if args.same_pic and plot_data_all:
        make_combined_plot(plot_data_all, args, model_names, suffix)
        
    if args.folium and plot_data_all:
        make_folium_map(plot_data_all, args, model_names, suffix)


def make_single_plot(d, args, idx, model_names):
    outdir = args.out_dir / str(d["mmsi"])
    outdir.mkdir(parents=True, exist_ok=True)
    fname = outdir / f"traj_{d['mmsi']}_{d['tid']}_{idx}.png"
    
    all_lats = list(d["past"][0]) + list(d["true"][0])
    all_lons = list(d["past"][1]) + list(d["true"][1])
    for m in d["models"].values():
        all_lats.extend(m["lats"])
        all_lons.extend(m["lons"])
        
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
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color=STYLE["grid"], alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    ax.plot(d["past"][1], d["past"][0], transform=ccrs.PlateCarree(), c=STYLE["past"], lw=2, label="Past")
    ax.plot(d["true"][1], d["true"][0], transform=ccrs.PlateCarree(), c=STYLE["true"], lw=3, label="True")
    
    title_parts = [f"MMSI {d['mmsi']}"]
    for i, m_name in enumerate(model_names):
        if m_name in d["models"]:
            md = d["models"][m_name]
            col = PRED_COLORS[i % len(PRED_COLORS)]
            ax.plot(md["lons"], md["lats"], transform=ccrs.PlateCarree(), 
                    c=col, lw=2.5, ls="--", label=f"{m_name} Pred")
            title_parts.append(f"{m_name} ADE:{md['ade']:.2f}/FDE:{md['fde']:.2f}")

    # --- FIX: Smaller Legend ---
    ax.legend(loc="upper right", framealpha=0.9, prop={'size': 8})
    ax.set_title(" | ".join(title_parts), fontsize=9)
    
    fig.savefig(fname, bbox_inches='tight', dpi=args.dpi)
    plt.close(fig)


def make_combined_plot(data_list, args, model_names, suffix):
    print("Generating combined plot...")
    
    if args.plot_lat_min:
         p_lat_min, p_lat_max = args.plot_lat_min, args.plot_lat_max
         p_lon_min, p_lon_max = args.plot_lon_min, args.plot_lon_max
    else:
         p_lat_min, p_lat_max = args.lat_min, args.lat_max
         p_lon_min, p_lon_max = args.lon_min, args.lon_max

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
    ax.set_extent([p_lon_min, p_lon_max, p_lat_min, p_lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.GSHHSFeature(scale='h', levels=[1], facecolor=STYLE["land"], edgecolor=STYLE["edge"]))
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    gl.top_labels = False; gl.right_labels = False

    for d in data_list:
        ax.plot(d["past"][1], d["past"][0], transform=ccrs.PlateCarree(), c=STYLE["past"], lw=1, alpha=0.5)
        ax.plot(d["true"][1], d["true"][0], transform=ccrs.PlateCarree(), c=STYLE["true"], lw=1, alpha=0.5)
        
        for i, m_name in enumerate(model_names):
            if m_name in d["models"]:
                col = PRED_COLORS[i % len(PRED_COLORS)]
                ax.plot(d["models"][m_name]["lons"], d["models"][m_name]["lats"], 
                        transform=ccrs.PlateCarree(), c=col, lw=1, alpha=0.6)

    legs = [
        Line2D([0], [0], color=STYLE["past"], lw=2, label='Past'),
        Line2D([0], [0], color=STYLE["true"], lw=2, label='True')
    ]
    for i, m_name in enumerate(model_names):
        col = PRED_COLORS[i % len(PRED_COLORS)]
        legs.append(Line2D([0], [0], color=col, lw=2, ls="--", label=f"{m_name} Pred"))
    
    ax.legend(handles=legs, loc="upper right", framealpha=0.9, prop={'size': 8})
    ax.set_title(f"Combined Predictions ({len(data_list)} trips)")
    
    out = args.out_dir / f"combined_{suffix}.png"
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved {out}")


def make_folium_map(data_list, args, model_names, suffix):
    print("Generating Interactive Map...")
    if not data_list: return
    
    lat_c = np.mean(data_list[0]["past"][0])
    lon_c = np.mean(data_list[0]["past"][1])
    
    m = folium.Map(location=[lat_c, lon_c], zoom_start=6, tiles=None)
    folium.TileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", attr="OSM", name="Streets").add_to(m)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attr="Esri", name="Satellite").add_to(m)

    for d in data_list:
        base_tt = f"MMSI: {d['mmsi']}"
        
        folium.PolyLine(list(zip(d["past"][0], d["past"][1])), color=STYLE["past"], weight=2.5, opacity=0.7, tooltip=base_tt+" (Past)").add_to(m)
        folium.PolyLine(list(zip(d["true"][0], d["true"][1])), color=STYLE["true"], weight=3, opacity=0.7, tooltip=base_tt+" (True)").add_to(m)
        
        for i, m_name in enumerate(model_names):
            if m_name in d["models"]:
                md = d["models"][m_name]
                col = PRED_COLORS[i % len(PRED_COLORS)]
                tt = f"{base_tt} | {m_name} ADE:{md['ade']:.2f} FDE:{md['fde']:.2f}"
                folium.PolyLine(list(zip(md["lats"], md["lons"])), color=col, weight=2.5, opacity=0.8, tooltip=tt).add_to(m)

    folium.LayerControl().add_to(m)
    out = args.out_dir / f"map_{suffix}.html"
    m.save(out)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()