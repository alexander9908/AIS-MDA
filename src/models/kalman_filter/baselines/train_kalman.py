"""Kalman Filter baseline evaluation utilities.

Includes:
- HPC-optimized evaluation loop using joblib for parallelism.
- Legacy helper functions (`load_trajectories`, `split_trajectories`,
  `create_windows`, `evaluate_kalman`) kept for compatibility with
  downstream scripts.

Usage example:
    python -m src.models.kalman_filter.baselines.train_kalman \
        --final_dir data/map_reduce_final/test --n_jobs -1
"""

from __future__ import annotations
import argparse
import pickle
import os
import json
from dataclasses import dataclass
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from joblib import Parallel, delayed
import folium

from src.models.kalman_filter.kalman_filter import TrajectoryKalmanFilter, KalmanFilterParams
from src.utils.water_guidance import is_water, project_to_water

# Column indices for normalized trajectories (MapReduce output)
LAT_IDX, LON_IDX = 0, 1

# --- Configuration ---
# NOTE: Ensure these match your normalization scaler exactly!
LAT_MIN, LAT_MAX = 54.0, 59.0 
LON_MIN, LON_MAX = 5.0, 17.0
EARTH_RADIUS_M = 6371000.0

@dataclass(frozen=True)
class Bounds:
    """Geographic normalization bounds."""
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


DEFAULT_BOUNDS = Bounds(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)

STYLE = {
    "past": "#1f77b4",
    "true": "#2ca02c",
    "pred": "#d62728",
}

# --- Metric Helpers ---
def normalize(real_coords: np.ndarray, bounds: Optional[Bounds] = None) -> np.ndarray:
    bounds = bounds or DEFAULT_BOUNDS
    real_coords = np.asarray(real_coords, dtype=np.float64)
    if real_coords.size == 0:
        return np.empty_like(real_coords)

    lat_span = bounds.lat_max - bounds.lat_min
    lon_span = bounds.lon_max - bounds.lon_min

    norm = np.zeros_like(real_coords, dtype=np.float64)
    norm[..., 0] = (real_coords[..., 0] - bounds.lat_min) / lat_span
    norm[..., 1] = (real_coords[..., 1] - bounds.lon_min) / lon_span
    return norm

def denormalize(norm_coords: np.ndarray, bounds: Optional[Bounds] = None) -> np.ndarray:
    """Vectorized denormalization: normalized -> real-world lat/lon."""
    bounds = bounds or DEFAULT_BOUNDS
    norm_coords = np.asarray(norm_coords, dtype=np.float64)
    if norm_coords.size == 0:
        return np.empty_like(norm_coords)

    lat_span = bounds.lat_max - bounds.lat_min
    lon_span = bounds.lon_max - bounds.lon_min

    denorm = np.zeros_like(norm_coords, dtype=np.float64)
    denorm[..., 0] = norm_coords[..., 0] * lat_span + bounds.lat_min
    denorm[..., 1] = norm_coords[..., 1] * lon_span + bounds.lon_min
    return denorm


def _anchor_last_water(past_real: np.ndarray) -> Tuple[float, float]:
    anchor_lat, anchor_lon = float(past_real[-1, 0]), float(past_real[-1, 1])
    if is_water(anchor_lat, anchor_lon):
        return anchor_lat, anchor_lon

    for idx in range(len(past_real) - 2, -1, -1):
        cand_lat, cand_lon = float(past_real[idx, 0]), float(past_real[idx, 1])
        if is_water(cand_lat, cand_lon):
            adjusted_lat, adjusted_lon = project_to_water(cand_lat, cand_lon, anchor_lat, anchor_lon)
            return adjusted_lat, adjusted_lon

    return anchor_lat, anchor_lon


def enforce_water_path(past_real: np.ndarray, pred_real: np.ndarray) -> np.ndarray:
    if pred_real.size == 0:
        return pred_real

    past_real = np.asarray(past_real, dtype=np.float64)
    pred_real = np.asarray(pred_real, dtype=np.float64).copy()

    anchor_lat, anchor_lon = _anchor_last_water(past_real)

    for i, (lat, lon) in enumerate(pred_real):
        lat_f, lon_f = float(lat), float(lon)
        if not is_water(lat_f, lon_f):
            lat_f, lon_f = project_to_water(anchor_lat, anchor_lon, lat_f, lon_f)
        anchor_lat, anchor_lon = lat_f, lon_f
        pred_real[i, 0] = anchor_lat
        pred_real[i, 1] = anchor_lon

    return pred_real

def haversine_error(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """
    Calculate error in METERS between predicted and true paths.
    Input shapes: (N, Horizon, 2)
    """
    # Convert to radians
    lat1, lon1 = np.radians(pred[..., 0]), np.radians(pred[..., 1])
    lat2, lon2 = np.radians(true[..., 0]), np.radians(true[..., 1])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return c * EARTH_RADIUS_M


def evaluate_trip_kalman(trip: np.ndarray,
                         window_size: int,
                         horizon: int,
                         kf_params: KalmanFilterParams,
                         bounds: Optional[Bounds] = None) -> Dict[str, np.ndarray | float]:
    """Evaluate Kalman filter on a single trajectory segment.

    Parameters
    ----------
    trip : np.ndarray
        Normalized trajectory with lat/lon in the first two columns.
    window_size : int
        Number of historical observations used for fitting.
    horizon : int
        Prediction horizon (steps ahead) to forecast.
    kf_params : KalmanFilterParams
        Kalman filter configuration.
    bounds : Optional[Bounds]
        Geographic normalization bounds for denormalization.

    Returns
    -------
    Dict[str, np.ndarray | float]
        Dictionary containing past, prediction, target (denormalized), per-step
        errors (meters), and ADE/FDE metrics (meters).
    """
    if len(trip) < window_size + horizon:
        raise ValueError("trajectory shorter than window_size + horizon")

    bounds = bounds or DEFAULT_BOUNDS

    history = trip[:window_size]
    target_norm = trip[window_size:window_size + horizon, LAT_IDX:LON_IDX + 1]

    kf = TrajectoryKalmanFilter(kf_params)
    pred_norm = kf.predict(history, horizon)

    past_real = denormalize(history[:, LAT_IDX:LON_IDX + 1], bounds)
    pred_real = denormalize(pred_norm, bounds)
    pred_real = enforce_water_path(past_real, pred_real)
    target_real = denormalize(target_norm, bounds)

    pred_norm = normalize(pred_real, bounds)

    errors_m = haversine_error(pred_real, target_real)
    errors_km = errors_m / 1000.0
    ade_km = float(np.mean(errors_km))
    fde_km = float(errors_km[-1])

    return {
        "past_real": past_real,
        "pred_real": pred_real,
        "target_real": target_real,
        "pred_norm": pred_norm,
        "target_norm": target_norm,
        "errors_m": errors_m,
        "errors_km": errors_km,
        "ade": ade_km,
        "fde": fde_km,
    }


def build_folium_map(plot_data: List[Dict[str, np.ndarray | float]], out_dir: Path) -> None:
    if not plot_data:
        print("No Folium data available; skipping map generation.")
        return

    all_lats: List[float] = []
    all_lons: List[float] = []
    for entry in plot_data:
        for segment in (entry["past"], entry["true"], entry["pred"]):
            all_lats.extend(segment[:, 0])
            all_lons.extend(segment[:, 1])

    if not all_lats or not all_lons:
        print("Folium map received empty coordinate arrays; skipping.")
        return

    lat_center = float(np.mean(all_lats))
    lon_center = float(np.mean(all_lons))

    fmap = folium.Map(location=[lat_center, lon_center], zoom_start=6, tiles=None)

    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="&copy; OpenStreetMap contributors",
        name="OSM Streets",
    ).add_to(fmap)

    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Esri Satellite",
    ).add_to(fmap)

    for entry in plot_data:
        tooltip = f"MMSI: {entry['mmsi']} | ADE: {entry['ade']:.2f} km | FDE: {entry['fde']:.2f} km"

        folium.PolyLine(
            locations=list(zip(entry["past"][:, 0], entry["past"][:, 1])),
            color=STYLE["past"],
            weight=2.5,
            opacity=0.8,
            tooltip=tooltip + " (Past)",
        ).add_to(fmap)

        folium.PolyLine(
            locations=list(zip(entry["true"][:, 0], entry["true"][:, 1])),
            color=STYLE["true"],
            weight=3.0,
            opacity=0.85,
            tooltip=tooltip + " (True)",
        ).add_to(fmap)

        if entry["pred"].shape[0] > 1:
            folium.PolyLine(
                locations=list(zip(entry["pred"][:, 0], entry["pred"][:, 1])),
                color=STYLE["pred"],
                weight=3.0,
                opacity=0.9,
                tooltip=tooltip + " (Pred)",
            ).add_to(fmap)

    folium.LayerControl().add_to(fmap)

    out_path = out_dir / "kalman_interactive_map.html"
    fmap.save(out_path)
    print(f"Folium map saved to: {out_path}")

# --- Core Processing Logic (Pickleable) ---
def process_trajectory_chunk(trajectories: List[np.ndarray], 
                             window_size: int, 
                             horizon: int,
                             kf_params: KalmanFilterParams) -> Dict:
    """
    Worker function:
    1. Generates windows from a chunk of trajectories.
    2. Runs Kalman Filter prediction.
    3. Computes errors immediately (to save RAM).
    """
    kf = TrajectoryKalmanFilter(kf_params)
    
    total_ade = 0.0
    total_fde = 0.0
    horizon_errors = np.zeros(horizon)
    count = 0
    
    for traj in trajectories:
        if len(traj) <= window_size + horizon:
            continue
            
        # Sliding window generation on-the-fly
        # To speed up, we can stride (skip) windows if data is too large
        # stride = 1 (full), stride = 5 (faster)
        stride = 1 
        
        for i in range(0, len(traj) - window_size - horizon + 1, stride):
            # Prepare Data
            input_window = traj[i : i + window_size]
            target = traj[i + window_size : i + window_size + horizon, LAT_IDX:LON_IDX + 1]
            
            # Run Model
            pred_norm = kf.predict(input_window, horizon)
            
            # Denormalize
            past_real = denormalize(input_window[:, LAT_IDX:LON_IDX + 1])
            pred_real = denormalize(pred_norm)
            pred_real = enforce_water_path(past_real, pred_real)
            target_real = denormalize(target)
            
            # Calc Errors (Meters)
            errors = haversine_error(pred_real, target_real) # Shape: (Horizon,)
            
            total_ade += np.mean(errors)
            total_fde += errors[-1]
            horizon_errors += errors
            count += 1
            
    return {
        "total_ade": total_ade,
        "total_fde": total_fde,
        "horizon_sum": horizon_errors,
        "count": count
    }


# --- Compatibility Helpers (used by other modules/notebooks) ---
def load_trajectories(data_dir: str, max_files: int | None = None) -> List[np.ndarray]:
    """Load processed trajectory pickles from a directory."""
    data_dir_path = Path(data_dir)
    if not data_dir_path.exists():
        print(f"Warning: directory not found: {data_dir_path}")
        return []

    paths = sorted(p for p in data_dir_path.glob("*_processed.pkl"))
    if max_files:
        paths = paths[:max_files]

    trajectories: List[np.ndarray] = []
    print(f"Loading {len(paths)} files from {data_dir_path}...")
    for path in tqdm(paths):
        try:
            with open(path, "rb") as f:
                item = pickle.load(f)
            traj = item.get("traj") if isinstance(item, dict) else item
            if traj is None or len(traj) <= 20:
                continue
            trajectories.append(traj)
        except Exception as exc:
            print(f"Error loading {path}: {exc}")

    return trajectories


def split_trajectories(trajectories: List[np.ndarray],
                       val_frac: float = 0.2,
                       test_frac: float = 0.1,
                       seed: int = 42) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Randomly split trajectories into train/val/test."""
    if not trajectories:
        return [], [], []

    rng = np.random.default_rng(seed)
    n = len(trajectories)
    indices = rng.permutation(n)

    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = max(0, n - n_test - n_val)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_trajs = [trajectories[i] for i in train_idx]
    val_trajs = [trajectories[i] for i in val_idx]
    test_trajs = [trajectories[i] for i in test_idx]

    return train_trajs, val_trajs, test_trajs


def create_windows(trajectories: List[np.ndarray],
                   window_size: int,
                   horizon: int,
                   max_windows: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Materialize sliding windows (legacy helper)."""
    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []

    for traj in trajectories:
        if len(traj) < window_size + horizon:
            continue

        for i in range(len(traj) - window_size - horizon + 1):
            X_list.append(traj[i:i + window_size])
            Y_list.append(traj[i + window_size:i + window_size + horizon, LAT_IDX:LON_IDX + 1])

            if max_windows and len(X_list) >= max_windows:
                break

        if max_windows and len(X_list) >= max_windows:
            break

    if not X_list:
        return (
            np.empty((0, window_size, 9), dtype=np.float32),
            np.empty((0, horizon, 2), dtype=np.float32)
        )

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    return X, Y


def evaluate_kalman(kf: TrajectoryKalmanFilter,
                    X: np.ndarray,
                    Y: np.ndarray,
                    batch_size: int = 100) -> Dict[str, float | List[float]]:
    """Evaluate Kalman Filter on pre-built windows (legacy helper)."""
    if len(X) == 0:
        return {
            "ade_meters": 0.0,
            "fde_meters": 0.0,
            "per_horizon_ade_meters": [],
            "n_samples": 0
        }

    preds: List[np.ndarray] = []
    for start in range(0, len(X), batch_size):
        batch_X = X[start:start + batch_size]
        preds.append(kf.predict_batch(batch_X, horizon=Y.shape[1]))

    predictions = np.concatenate(preds, axis=0)

    pred_real = denormalize(predictions)
    true_real = denormalize(Y)
    errors = haversine_error(pred_real, true_real)

    ade_val = float(np.mean(errors))
    fde_val = float(np.mean(errors[:, -1]))
    per_horizon = [float(val) for val in np.mean(errors, axis=0)]

    return {
        "ade_meters": ade_val,
        "fde_meters": fde_val,
        "per_horizon_ade_meters": per_horizon,
        "n_samples": int(len(X))
    }

# --- Main Pipeline ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dir", "--final_dir", dest="split_dir", required=True)
    parser.add_argument("--out_dir", default="results/baselines")
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=12) # 12 * 5mins = 1 Hour
    parser.add_argument("--n_jobs", type=int, default=-1, help="Num CPU cores")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--folium", action="store_true", help="Generate interactive Folium map")
    parser.add_argument("--folium_max", type=int, default=None, help="Limit number of trajectories plotted in the Folium map")
    args = parser.parse_args()
    
    # 1. Load Data
    all_trajs = load_trajectories(args.split_dir, args.max_files)

    if not all_trajs:
        print("No trajectories loaded. Exiting.")
        return

    # Split (Simple 80/10/10 or fallback to using all trajectories)
    test_size = max(1, int(len(all_trajs) * 0.1))
    if test_size >= len(all_trajs):
        test_trajs = all_trajs
    else:
        test_trajs = all_trajs[-test_size:]
    print(f"Evaluated on Test Set: {len(test_trajs)} trajectories")
    
    # 2. Configure Kalman Filter
    # High velocity noise allows the 'tangent' to update quickly before prediction
    params = KalmanFilterParams(
        process_noise_pos=1e-5,
        process_noise_vel=1e-3, 
        measurement_noise=1e-4,
        dt=300.0
    )
    
    # 3. Parallel Execution
    # Chunk trajectories for workers
    n_jobs = args.n_jobs if args.n_jobs > 0 else os.cpu_count()
    n_jobs = int(n_jobs) if n_jobs else 1
    n_jobs = max(1, n_jobs)
    chunk_size = max(1, len(test_trajs) // n_jobs)
    chunks = [test_trajs[i:i + chunk_size] for i in range(0, len(test_trajs), chunk_size)]
    
    print(f"Starting Parallel Evaluation on {n_jobs} cores...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_trajectory_chunk)(
            chunk, args.window, args.horizon, params
        ) for chunk in tqdm(chunks)
    )

    results = [r for r in results if r and r.get("count", 0) > 0]
    if not results:
        print("No valid windows found across chunks.")
        return
    
    # 4. Aggregation
    total_ade = sum(r['total_ade'] for r in results)
    total_fde = sum(r['total_fde'] for r in results)
    total_horizon = np.sum([r['horizon_sum'] for r in results], axis=0)
    total_count = sum(r['count'] for r in results)
    
    if total_count == 0:
        print("No valid windows found.")
        return

    final_ade = total_ade / total_count
    final_fde = total_fde / total_count
    final_horizon_ade = np.asarray(total_horizon, dtype=np.float64) / total_count
    
    # 5. Reporting
    print("\n" + "="*30)
    print(f"KALMAN FILTER BASELINE RESULTS")
    print(f"Straight Line (CV) Model")
    print("="*30)
    print(f"Samples Evaluated: {total_count}")
    print(f"ADE (Mean Error):  {final_ade:.2f} meters")
    print(f"FDE (Final Error): {final_fde:.2f} meters")
    print("-" * 30)
    print("Error per Time Step (meters):")
    for i, err in enumerate(final_horizon_ade):
        print(f"  T+{i+1} ({(i+1)*5} min): {err:.2f}m")
        
    # Save
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{args.out_dir}/kalman_results.json", "w") as f:
        json.dump({
            "ade": final_ade, 
            "fde": final_fde, 
            "steps": final_horizon_ade.tolist()
        }, f, indent=2)

    if args.folium:
        print("\nPreparing Folium visualization...")
        folium_entries: List[Dict[str, np.ndarray | float]] = []
        for idx, traj in enumerate(test_trajs):
            if len(traj) < args.window + args.horizon:
                continue

            mmsi = int(traj[0, 8]) if traj.shape[1] > 8 else idx
            slice_len = args.window + args.horizon
            trip_slice = traj[:slice_len]
            eval_result = evaluate_trip_kalman(
                trip_slice,
                window_size=args.window,
                horizon=args.horizon,
                kf_params=params,
                bounds=DEFAULT_BOUNDS,
            )

            past = eval_result["past_real"]
            true_future = eval_result["target_real"]
            pred_future = eval_result["pred_real"]

            true_cont = np.vstack([past[-1:], true_future]) if len(true_future) > 0 else past[-1:]
            pred_cont = np.vstack([past[-1:], pred_future]) if len(pred_future) > 0 else past[-1:]

            folium_entries.append({
                "mmsi": mmsi,
                "ade": eval_result["ade"],
                "fde": eval_result["fde"],
                "past": past,
                "true": true_cont,
                "pred": pred_cont,
            })

            if args.folium_max and len(folium_entries) >= args.folium_max:
                break

        build_folium_map(folium_entries, Path(args.out_dir))

if __name__ == "__main__":
    main()