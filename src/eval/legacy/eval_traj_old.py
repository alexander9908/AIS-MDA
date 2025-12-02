# src/eval/eval_traj.py
from __future__ import annotations
import matplotlib
matplotlib.use('Agg') 
import argparse, os, glob, pickle, csv, datetime as dt
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import torch
import matplotlib.pyplot as plt
import joblib
from matplotlib.lines import Line2D

# Interactive Map
import folium

from src.models.kalman_filter.baselines.train_kalman import evaluate_trip_kalman, Bounds
from src.models.kalman_filter.kalman_filter import KalmanFilterParams

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---------------- Models ----------------
from src.models.legacy.traisformer_old import TrAISformer, BinSpec
try:
    from src.models.tptrans_gru import TPTrans
except ImportError:
    print("[Warning] Could not import TPTrans from src.models.tptrans_V2, falling back to src.models.tptrans")
    from src.models.legacy.tptrans import TPTrans

# ---------------- Water mask ----------------
from src.utils.water_guidance import is_water, project_to_water

# ---------------- Professional Style Config ----------------
STYLE = {
    "past": "#1f77b4",   # Muted Blue
    "true": "#2ca02c",   # Muted Green
    "pred": "#d62728",   # Muted Red
    "land": "#E0E0E0",   # Soft Light Grey
    "edge": "#505050",   # Dark Grey borders
    "water": "#FFFFFF",  # Pure White
    "grid": "#B0B0B0"    # Subtle Grid
}

plt.rcParams.update({
    "figure.figsize": (8.0, 6.5),
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.facecolor": STYLE["water"],
})

# ---------------- Helpers ----------------
def compute_horizon_metrics(lats_true: np.ndarray, lons_true: np.ndarray, 
                           lats_pred: np.ndarray, lons_pred: np.ndarray,
                           horizons_steps: List[int] = [12, 24, 36]) -> Dict[str, Optional[float]]:
    """
    Compute Haversine distance at specific time horizons (1h, 2h, 3h).
    
    Args:
        lats_true: Ground truth latitudes (N,)
        lons_true: Ground truth longitudes (N,)
        lats_pred: Predicted latitudes (M,)
        lons_pred: Predicted longitudes (M,)
        horizons_steps: List of timestep indices (e.g., [12, 24, 36] for 1h/2h/3h at 5min sampling)
    
    Returns:
        Dict with keys like 'ade_1h_km', 'ade_2h_km', 'ade_3h_km' (None if insufficient data)
    """
    metrics = {}
    n_true = len(lats_true)
    n_pred = len(lats_pred)
    
    for h in horizons_steps:
        key = f"ade_{h//12}h_km"  # 12 steps = 1 hour
        
        if n_true > h and n_pred > h:
            # Compute mean error up to this horizon
            errors = [haversine_km(lats_true[i], lons_true[i], lats_pred[i], lons_pred[i]) 
                     for i in range(h + 1)]  # Include step h itself
            metrics[key] = float(np.mean(errors))
        else:
            metrics[key] = None  # Not enough data for this horizon
    
    return metrics


def parse_trip(fname: str) -> Tuple[int, int]:
    base = os.path.basename(fname).replace("_processed.pkl", "")
    parts = base.split("_")
    if len(parts) >= 2:
        return int(parts[0]), int(parts[1])
    return 0, 0

def load_trip(path: str, min_points: int = 30) -> np.ndarray:
    try:
        data = joblib.load(path)
    except Exception:
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
        "scale_factor": sd_top.get("scale_factor", 100.0), 
        "norm_config": sd_top.get("norm_config", None),
        "bins": None
    }

    if kind.lower() == "tptrans":
        d_model = 512 
        if 'proj.weight' in clean_sd:
            d_model = clean_sd['proj.weight'].shape[1]
        elif 'conv.net.0.weight' in clean_sd:
            d_model = clean_sd['conv.net.0.weight'].shape[0]
            
        print(f"[Model] Detected TPTrans configuration: d_model={d_model}")
        
        enc_layers = 0
        while f'encoder.layers.{enc_layers}.linear1.weight' in clean_sd: enc_layers += 1
        if enc_layers == 0: enc_layers = 4 
        
        dec_layers = 0
        while f'dec.weight_ih_l{dec_layers}' in clean_sd: dec_layers += 1
        if dec_layers == 0: dec_layers = 2 

        nhead = 8
        model = TPTrans(feat_dim=feat_dim, d_model=d_model, nhead=nhead, enc_layers=enc_layers, dec_layers=dec_layers, horizon=horizon)
        try:
            model.load_state_dict(clean_sd, strict=True)
        except RuntimeError:
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

# ---------------- Core per-trip evaluation ----------------
def evaluate_and_plot_trip(fpath: str, trip: np.ndarray, model, meta, args, sample_idx: int) -> Dict[str, Any]:
    mmsi, tid = parse_trip(fpath)
    
    # 1. Denormalize using DATA BOUNDS (args.lat_min/max)
    full_lat_norm = trip[:, 0]
    full_lon_norm = trip[:, 1]
    full_sog_norm = trip[:, 2]
    full_cog_norm = trip[:, 3]

    full_lat_deg = full_lat_norm * (args.lat_max - args.lat_min) + args.lat_min
    full_lon_deg = full_lon_norm * (args.lon_max - args.lon_min) + args.lon_min

    past, future_true_all, cut = split_by_percent(trip, args.pred_cut)
    
    if len(past) < 2: raise ValueError("too short past")

    if args.pred_len is not None:
        N_future = int(args.pred_len)
    else:
        if len(future_true_all) < 2: raise ValueError("too short future")
        N_future = len(future_true_all) if args.cap_future is None else min(len(future_true_all), int(args.cap_future))

    lats_past = full_lat_deg[:cut]; lons_past = full_lon_deg[:cut]
    cur_lat = float(lats_past[-1]); cur_lon = float(lons_past[-1])
    
    # --- FIX: Ensure True Path connects to Current Position ---
    raw_lats_future = full_lat_deg[cut:cut+N_future]
    raw_lons_future = full_lon_deg[cut:cut+N_future]
    
    lats_true_plot = np.concatenate(([cur_lat], raw_lats_future))
    lons_true_plot = np.concatenate(([cur_lon], raw_lons_future))
    
    # For metrics, compare only the future steps
    lats_true_eval = raw_lats_future
    lons_true_eval = raw_lons_future

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if args.model.lower() != "kalman":
        model = model.to(device).eval()

    if args.model.lower() == "tptrans":
        # TPTrans Prediction Loop
        seq_in = np.stack([full_lat_norm[:cut], full_lon_norm[:cut], full_sog_norm[:cut], full_cog_norm[:cut]], axis=1).astype(np.float32)
        curr_seq_in = seq_in.copy()
        pred_lat_list = [cur_lat]; pred_lon_list = [cur_lon]
        curr_lat_norm = curr_seq_in[-1, 0]; curr_lon_norm = curr_seq_in[-1, 1]
        
        steps_needed = N_future
        steps_generated = 0
        
        while steps_generated < steps_needed:
            Tin = min(args.past_len, len(curr_seq_in))
            X_in = curr_seq_in[-Tin:, :][None, ...]
            X_tensor = torch.from_numpy(X_in).to(device)

            with torch.no_grad():
                out = model(X_tensor)[0].cpu().numpy()
            
            scale_factor = float(meta.get("scale_factor", 100.0))
            out_deltas_deg = (out / scale_factor) * float(args.pred_scale)
            chunk_len = len(out_deltas_deg)
            new_rows_norm = []
            
            for k in range(chunk_len):
                dlat_deg = out_deltas_deg[k, 0]
                dlon_deg = out_deltas_deg[k, 1]
                
                curr_lat_deg = curr_lat_norm * (args.lat_max - args.lat_min) + args.lat_min
                curr_lon_deg = curr_lon_norm * (args.lon_max - args.lon_min) + args.lon_min
                
                cand_lat_deg = curr_lat_deg + dlat_deg
                cand_lon_deg = curr_lon_deg + dlon_deg
                
                prev_lat_deg = pred_lat_list[-1]
                prev_lon_deg = pred_lon_list[-1]
                
                if is_water(cand_lat_deg, cand_lon_deg):
                    fix_lat_deg, fix_lon_deg = cand_lat_deg, cand_lon_deg
                else:
                    fix_lat_deg, fix_lon_deg = project_to_water(prev_lat_deg, prev_lon_deg, cand_lat_deg, cand_lon_deg)
                
                pred_lat_list.append(fix_lat_deg)
                pred_lon_list.append(fix_lon_deg)
                
                fix_lat_norm = (fix_lat_deg - args.lat_min) / (args.lat_max - args.lat_min)
                fix_lon_norm = (fix_lon_deg - args.lon_min) / (args.lon_max - args.lon_min)
                
                curr_lat_norm, curr_lon_norm = fix_lat_norm, fix_lon_norm
                last_sog = curr_seq_in[-1, 2]
                last_cog = curr_seq_in[-1, 3]
                new_rows_norm.append([fix_lat_norm, fix_lon_norm, last_sog, last_cog])
            
            new_rows_norm = np.array(new_rows_norm, dtype=np.float32)
            curr_seq_in = np.concatenate([curr_seq_in, new_rows_norm], axis=0)
            steps_generated += chunk_len
            if steps_generated >= steps_needed: break

        pred_lat = np.array(pred_lat_list)
        pred_lon = np.array(pred_lon_list)
    elif args.model.lower() == "kalman":
        window = max(1, min(int(args.past_len), cut))
        horizon_steps = N_future
        if len(trip) < cut + horizon_steps:
            raise ValueError("insufficient future points for Kalman forecast")
        start_idx = cut - window
        kf_slice = trip[start_idx:start_idx + window + horizon_steps]
        bounds = Bounds(args.lat_min, args.lat_max, args.lon_min, args.lon_max)
        kalman_params = None
        if isinstance(meta, dict):
            kalman_params = meta.get("kalman_params")
        if kalman_params is None:
            kalman_params = KalmanFilterParams()
        eval_res = evaluate_trip_kalman(
            kf_slice,
            window_size=window,
            horizon=horizon_steps,
            kf_params=kalman_params,
            bounds=bounds,
        )
        pred_lat = np.concatenate(([cur_lat], eval_res["pred_real"][:, 0]))
        pred_lon = np.concatenate(([cur_lon], eval_res["pred_real"][:, 1]))
    else:
        pred_lat = np.zeros(N_future + 1) + cur_lat 
        pred_lon = np.zeros(N_future + 1) + cur_lon

    if args.match_distance and len(pred_lat) > 1 and len(lats_true_eval) > 1:
        dt_true = cumdist(lats_true_eval, lons_true_eval)[-1]
        cd = cumdist(pred_lat, pred_lon)
        keep = int(np.searchsorted(cd, dt_true, side="right"))
        keep = max(2, min(keep, len(pred_lat)))
        pred_lat, pred_lon = pred_lat[:keep], pred_lon[:keep]

    preds_to_eval_lat = pred_lat[1:]
    preds_to_eval_lon = pred_lon[1:]
    n_comp = min(len(preds_to_eval_lat), len(lats_true_eval))
    if n_comp > 0:
        ade = float(np.mean([haversine_km(lats_true_eval[i], lons_true_eval[i], 
                                         preds_to_eval_lat[i], preds_to_eval_lon[i]) 
                            for i in range(n_comp)]))
        fde = float(haversine_km(lats_true_eval[n_comp-1], lons_true_eval[n_comp-1], 
                                preds_to_eval_lat[n_comp-1], preds_to_eval_lon[n_comp-1]))
    else:
        ade, fde = np.nan, np.nan
    
    # NEW: Horizon-specific metrics
    horizon_metrics = compute_horizon_metrics(
        lats_true_eval, lons_true_eval,
        preds_to_eval_lat, preds_to_eval_lon,
        horizons_steps=[12, 24, 36]
    )
    
    res = {
        "mmsi": mmsi, "trip": tid, "cut_idx": cut,
        "ade_km": ade, "fde_km": fde,
        **horizon_metrics,  # Add 'ade_1h_km', 'ade_2h_km', 'ade_3h_km'
        "png": "skipped"
    }
    
    # Store plot data for --same_pic AND --folium
    if args.same_pic or args.folium:
        res["plot_data"] = {
            "mmsi": mmsi,
            "ade": ade,
            "fde": fde,
            "lats_past": lats_past, "lons_past": lons_past,
            "lats_true": lats_true_plot, "lons_true": lons_true_plot, 
            "lats_pred": pred_lat, "lons_pred": pred_lon
        }

    # ---- INDIVIDUAL STATIC PLOTTING ----
    if not args.no_plots:
        outdir_mmsi = Path(args.out_dir) / f"{mmsi}"
        outdir_mmsi.mkdir(parents=True, exist_ok=True)
        fname_png = outdir_mmsi / f"traj_{args.model}_mmsi-{mmsi}_trip-{tid}_cut-{args.pred_cut}_idx-{sample_idx}.png"

        # Bounds Logic
        if args.plot_lat_min is not None and args.plot_lat_max is not None:
            p_lat_min, p_lat_max = args.plot_lat_min, args.plot_lat_max
            p_lon_min, p_lon_max = args.plot_lon_min, args.plot_lon_max
        else:
            all_lats = np.concatenate([lats_past, lats_true_plot, pred_lat])
            all_lons = np.concatenate([lons_past, lons_true_plot, pred_lon])
            pad = 0.15
            p_lat_min, p_lat_max = np.min(all_lats) - pad, np.max(all_lats) + pad
            p_lon_min, p_lon_max = np.min(all_lons) - pad * 1.5, np.max(all_lons) + pad * 1.5

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
        ax.set_extent([p_lon_min, p_lon_max, p_lat_min, p_lat_max], crs=ccrs.PlateCarree())
        
        # Features
        land_feature = cfeature.GSHHSFeature(scale='h', levels=[1], 
                                           facecolor=STYLE["land"], 
                                           edgecolor=STYLE["edge"],
                                           linewidth=0.5)
        ax.add_feature(land_feature)
        ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5, linewidth=0.5)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color=STYLE["grid"], alpha=0.5, linestyle='--')
        gl.top_labels = False; gl.right_labels = False

        # Plot Individual Lines
        ax.plot(lons_past, lats_past, transform=ccrs.PlateCarree(), 
                lw=2.5, color=STYLE["past"], label="Past", zorder=3)
        ax.plot(lons_true_plot, lats_true_plot, transform=ccrs.PlateCarree(), 
                lw=3.0, color=STYLE["true"], label="True", zorder=4)
        if len(pred_lon) > 1:
            ax.plot(pred_lon, pred_lat, transform=ccrs.PlateCarree(), 
                    lw=3.0, color=STYLE["pred"], label="Pred", zorder=5)
            
        # Start/Current Point
        ax.scatter([lons_past[-1]], [lats_past[-1]], transform=ccrs.PlateCarree(), 
                   s=50, c=STYLE["pred"], edgecolors="k", zorder=10, label="Current Pos")

        ax.legend(loc='upper right', frameon=True, framealpha=0.9, fancybox=True)
        ax.set_title(f"MMSI {mmsi} | ADE {ade:.2f} km | FDE {fde:.2f} km")
        
        fig.tight_layout()
        fig.savefig(fname_png, dpi=args.dpi, bbox_inches='tight')
        plt.close(fig)
        res["png"] = str(fname_png)

    return res

def generate_folium_map(plot_data_list, out_dir, args):
    """Generates an interactive HTML map using Folium."""
    print("-" * 40)
    print(f"Generating Interactive Map for {len(plot_data_list)} trips...")

    # Calculate center of the map based on data
    all_lats = []
    all_lons = []
    for d in plot_data_list:
        all_lats.extend(d['lats_past'])
        all_lons.extend(d['lons_past'])
    
    if not all_lats:
        print("No data to plot in Folium.")
        return

    lat_center = np.mean(all_lats)
    lon_center = np.mean(all_lons)
    
    m = folium.Map(location=[lat_center, lon_center], zoom_start=6, tiles=None)

    # 1. OSM Streets
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="&copy; OpenStreetMap contributors",
        name="OSM Streets"
    ).add_to(m)

    # 2. Esri Satellite
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Esri Satellite"
    ).add_to(m)

    # Add trajectories
    for d in plot_data_list:
        tooltip_txt = f"MMSI: {d['mmsi']} | ADE: {d['ade']:.2f} km | FDE: {d['fde']:.2f} km"
        
        # Past (Blue)
        folium.PolyLine(
            locations=list(zip(d['lats_past'], d['lons_past'])),
            color=STYLE["past"], weight=2.5, opacity=0.8, 
            tooltip=tooltip_txt + " (Past)"
        ).add_to(m)

        # True (Green)
        folium.PolyLine(
            locations=list(zip(d['lats_true'], d['lons_true'])),
            color=STYLE["true"], weight=3, opacity=0.8,
            tooltip=tooltip_txt + " (True)"
        ).add_to(m)

        # Pred (Red)
        if len(d['lats_pred']) > 1:
            folium.PolyLine(
                locations=list(zip(d['lats_pred'], d['lons_pred'])),
                color=STYLE["pred"], weight=3, opacity=0.9,
                tooltip=tooltip_txt + " (Pred)"
            ).add_to(m)

    folium.LayerControl().add_to(m)
    
    out_file = Path(out_dir) / "interactive_map.html"
    m.save(out_file)
    print(f"Interactive map saved to: {out_file}")


# ---------------- Main ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split_dir", required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--model", required=True)
    p.add_argument("--out_dir", required=True)
    
    # Model/Data Bounds
    p.add_argument("--lat_min", type=float, default=None)
    p.add_argument("--lat_max", type=float, default=None)
    p.add_argument("--lon_min", type=float, default=None)
    p.add_argument("--lon_max", type=float, default=None)
    
    # Plot Viewport Bounds
    p.add_argument("--plot_lat_min", type=float, default=None)
    p.add_argument("--plot_lat_max", type=float, default=None)
    p.add_argument("--plot_lon_min", type=float, default=None)
    p.add_argument("--plot_lon_max", type=float, default=None)
    
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--past_len", type=int, default=64)
    p.add_argument("--pred_cut", type=float, default=80.0)
    p.add_argument("--no_plots", action="store_true", help="Disable individual trip plots")
    p.add_argument("--same_pic", action="store_true", help="Plot all processed trips on a single combined image")
    p.add_argument("--folium", action="store_true", help="Generate an interactive HTML map with Folium")
    p.add_argument("--match_distance", action="store_true")
    p.add_argument("--auto_extent", action="store_true")
    p.add_argument("--style", default="classic")
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--pred_scale", type=float, default=1.0)
    p.add_argument("--no_tiles", action="store_true")
    p.add_argument("--pred_len", type=int, default=None)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=60)
    p.add_argument("--cap_future", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--mmsi", default="") 
    p.add_argument("--speed_max", type=float, default=30.0)

    args = p.parse_args()
    if args.model.lower() != "kalman" and not args.ckpt:
        p.error("--ckpt is required unless --model kalman")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.split_dir, "*_processed.pkl")))
    if args.mmsi:
        target_mmsis = str(args.mmsi).strip().split(',')
        files = [f for f in files if any(os.path.basename(f).startswith(m + "_") for m in target_mmsis)]

    feat_dim = 4
    if args.model.lower() == "kalman":
        model = None
        meta = {
            "kalman_params": KalmanFilterParams(),
            "norm_config": None,
        }
    else:
        model, meta = build_model(args.model, args.ckpt, feat_dim, args.horizon)
    
    # --- Auto-fill DATA bounds from checkpoint ---
    norm_config = meta.get("norm_config")
    if norm_config:
        if args.lat_min is None: args.lat_min = norm_config.get("LAT_MIN")
        if args.lat_max is None: args.lat_max = norm_config.get("LAT_MAX")
        if args.lon_min is None: args.lon_min = norm_config.get("LON_MIN")
        if args.lon_max is None: args.lon_max = norm_config.get("LON_MAX")
        if "SOG_MAX" in norm_config: args.speed_max = norm_config.get("SOG_MAX")
    
    # Fallbacks
    if args.lat_min is None: args.lat_min = 54.0
    if args.lat_max is None: args.lat_max = 59.0
    if args.lon_min is None: args.lon_min = 5.0
    if args.lon_max is None: args.lon_max = 17.0
    
    print("-" * 40)
    print(f"[DEBUG] DATA BOUNDS (Math):")
    print(f"   Lat: {args.lat_min} to {args.lat_max}")
    print(f"   Lon: {args.lon_min} to {args.lon_max}")
    if args.plot_lat_min:
        print(f"[DEBUG] PLOT BOUNDS (Camera):")
        print(f"   Lat: {args.plot_lat_min} to {args.plot_lat_max}")
        print(f"   Lon: {args.plot_lon_min} to {args.plot_lon_max}")
    print("-" * 40)

    metrics = []
    combined_plot_data = [] 

    total_files = len(files)
    for idx, f in enumerate(files):
        try:
            trip = load_trip(f)
            res = evaluate_and_plot_trip(f, trip, model, meta, args, idx)
            metrics.append(res)
            
            if args.same_pic or args.folium:
                combined_plot_data.append(res["plot_data"])
            
            if not args.no_plots:
                if idx < 5 or idx % 20 == 0:
                    print(f"[{idx+1}/{total_files}] Saved {res['png']}")
            else:
                if idx % 50 == 0: 
                    print(f"Processed {idx}/{total_files}...")
                    
        except Exception as e:
            print(f"[skip] {os.path.basename(f)}: {e}")

    # --- FINAL METRICS ---
    print("-" * 40)
    mean_ade, mean_fde = 0.0, 0.0
    median_ade, median_fde = 0.0, 0.0
    
    if metrics:
        out_csv = Path(args.out_dir) / "metrics.csv"
        
        ade_values = [m['ade_km'] for m in metrics]
        fde_values = [m['fde_km'] for m in metrics]
        
        mean_ade = np.mean(ade_values)
        mean_fde = np.mean(fde_values)
        median_ade = np.median(ade_values)
        median_fde = np.median(fde_values)
        
        metrics_clean = [{k:v for k,v in m.items() if k != 'plot_data'} for m in metrics]
        
        with open(out_csv, "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=list(metrics_clean[0].keys()))
            w.writeheader()
            w.writerows(metrics_clean)
        
        print(f"Total Trips: {len(metrics)}")
        print(f"Mean ADE:    {mean_ade:.2f} km  | Median ADE: {median_ade:.2f} km")
        print(f"Mean FDE:    {mean_fde:.2f} km  | Median FDE: {median_fde:.2f} km")
        # Horizon-specific statistics
        for horizon_key in ['ade_1h_km', 'ade_2h_km', 'ade_3h_km']:
            valid_vals = [m[horizon_key] for m in metrics if m.get(horizon_key) is not None]
            n_valid = len(valid_vals)
            if n_valid > 0:
                mean_val = np.mean(valid_vals)
                median_val = np.median(valid_vals)
                print(f"{horizon_key}: {mean_val:.2f} km (median {median_val:.2f} km) | N={n_valid}")
            else:
                print(f"{horizon_key}: No samples available")
        print(f"Metrics saved to: {out_csv}")
    else:
        print("No metrics collected.")

    # --- COMBINED PLOT GENERATION ---
    if args.same_pic and combined_plot_data:
        print("-" * 40)
        print("Generating combined plot...")
        
        if args.plot_lat_min is not None:
             p_lat_min, p_lat_max = args.plot_lat_min, args.plot_lat_max
             p_lon_min, p_lon_max = args.plot_lon_min, args.plot_lon_max
        else:
            all_lats = []
            all_lons = []
            for d in combined_plot_data:
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
        
        n_plots = len(combined_plot_data)
        alpha_base = 0.6 if n_plots > 10 else 0.9
        lw_base = 1.0 if n_plots > 10 else 2.0
        
        for d in combined_plot_data:
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
        
        # Updated Title with Medians
        title_str = (f"Combined Trajectories (N={n_plots})\n"
                     f"Mean ADE: {mean_ade:.2f} km | Median ADE: {median_ade:.2f} km\n"
                     f"Mean FDE: {mean_fde:.2f} km | Median FDE: {median_fde:.2f} km")
        ax.set_title(title_str, fontsize=11, pad=12)

        out_name = Path(args.out_dir) / "combined_trajectories.png"
        fig.tight_layout()
        fig.savefig(out_name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Combined plot saved to: {out_name}")

    if args.folium and combined_plot_data:
        generate_folium_map(combined_plot_data, args.out_dir, args)

if __name__ == "__main__":
    main()