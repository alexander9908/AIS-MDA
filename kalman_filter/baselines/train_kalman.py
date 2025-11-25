"""
HPC-Optimized Kalman Filter Baseline Evaluation.

Features:
1. Parallel Processing (Joblib)
2. On-the-fly Windowing (Memory Efficient)
3. Haversine (Geodesic) Error Metrics

Usage:
    python -m src.baselines.train_kalman --final_dir data/map_reduce_final --n_jobs -1
"""

from __future__ import annotations
import argparse
import pickle
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from joblib import Parallel, delayed

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kalman_filter.kalman_filter import TrajectoryKalmanFilter, KalmanFilterParams

# --- Configuration ---
# NOTE: Ensure these match your normalization scaler exactly!
LAT_MIN, LAT_MAX = 54.0, 59.0 
LON_MIN, LON_MAX = 5.0, 17.0
EARTH_RADIUS_M = 6371000.0

class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer, np.floating)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)

# --- Metric Helpers ---
def denormalize(norm_coords: np.ndarray) -> np.ndarray:
    """Vectorized denormalization: [N, T, 2] -> Real Lat/Lon"""
    denorm = np.zeros_like(norm_coords)
    # lat = norm * (max - min) + min
    denorm[..., 0] = norm_coords[..., 0] * (LAT_MAX - LAT_MIN) + LAT_MIN
    # lon = norm * (max - min) + min
    denorm[..., 1] = norm_coords[..., 1] * (LON_MAX - LON_MIN) + LON_MIN
    return denorm

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
        
        num_windows = (len(traj) - window_size - horizon) // stride + 1
        
        # Allocate arrays for this specific trajectory to vectorize KF
        # Note: KF logic is per-window, so we loop.
        
        for i in range(0, len(traj) - window_size - horizon + 1, stride):
            # Prepare Data
            input_window = traj[i : i + window_size]
            target = traj[i + window_size : i + window_size + horizon, 0:2] # Lat, Lon
            
            # Run Model
            pred_norm = kf.predict(input_window, horizon)
            
            # Denormalize
            pred_real = denormalize(pred_norm)
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

# --- Main Pipeline ---
def load_data(data_dir, max_files):
    paths = sorted([p for p in Path(data_dir).glob("*_processed.pkl")])
    if max_files: paths = paths[:max_files]
    
    trajs = []
    print(f"Loading {len(paths)} files...")
    for p in tqdm(paths):
        try:
            with open(p, "rb") as f:
                data = pickle.load(f)
                # Ensure we just get the numpy array
                t = data['traj'] if isinstance(data, dict) else data
                if len(t) > 50: trajs.append(t)
        except: pass
    return trajs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final_dir", required=True)
    parser.add_argument("--out_dir", default="results/baselines")
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=12) # 12 * 5mins = 1 Hour
    parser.add_argument("--n_jobs", type=int, default=-1, help="Num CPU cores")
    parser.add_argument("--max_files", type=int, default=None)
    args = parser.parse_args()
    
    # 1. Load Data
    all_trajs = load_data(args.final_dir, args.max_files)
    
    # Split (Simple 80/10/10 or just Test set if that's all we need for baseline)
    # Assuming we verify on Test set
    test_size = int(0.1 * len(all_trajs))
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
    chunk_size = max(1, len(test_trajs) // n_jobs)
    chunks = [test_trajs[i:i + chunk_size] for i in range(0, len(test_trajs), chunk_size)]
    
    print(f"Starting Parallel Evaluation on {n_jobs} cores...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_trajectory_chunk)(
            chunk, args.window, args.horizon, params
        ) for chunk in tqdm(chunks)
    )
    
    # 4. Aggregation
    total_ade = sum(r['total_ade'] for r in results)
    total_fde = sum(r['total_fde'] for r in results)
    total_horizon = sum(r['horizon_sum'] for r in results)
    total_count = sum(r['count'] for r in results)
    
    if total_count == 0:
        print("No valid windows found.")
        return

    final_ade = total_ade / total_count
    final_fde = total_fde / total_count
    final_horizon_ade = total_horizon / total_count
    
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

if __name__ == "__main__":
    main()