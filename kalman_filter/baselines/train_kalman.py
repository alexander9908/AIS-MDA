"""Kalman Filter baseline evaluation utilities.

Includes:
- HPC-optimized evaluation loop using joblib for parallelism.
- Legacy helper functions (`load_trajectories`, `split_trajectories`,
  `create_windows`, `evaluate_kalman`) kept for compatibility with
  downstream scripts.

Usage example:
    python -m kalman_filter.baselines.train_kalman \
        --final_dir data/map_reduce_final/test --n_jobs -1
"""

from __future__ import annotations
import argparse
import pickle
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
from joblib import Parallel, delayed

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kalman_filter.kalman_filter import TrajectoryKalmanFilter, KalmanFilterParams

# Column indices for normalized trajectories (MapReduce output)
LAT_IDX, LON_IDX = 0, 1

# --- Configuration ---
# NOTE: Ensure these match your normalization scaler exactly!
LAT_MIN, LAT_MAX = 54.0, 59.0 
LON_MIN, LON_MAX = 5.0, 17.0
EARTH_RADIUS_M = 6371000.0

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
        
        for i in range(0, len(traj) - window_size - horizon + 1, stride):
            # Prepare Data
            input_window = traj[i : i + window_size]
            target = traj[i + window_size : i + window_size + horizon, LAT_IDX:LON_IDX + 1]
            
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
    parser.add_argument("--final_dir", required=True)
    parser.add_argument("--out_dir", default="results/baselines")
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=12) # 12 * 5mins = 1 Hour
    parser.add_argument("--n_jobs", type=int, default=-1, help="Num CPU cores")
    parser.add_argument("--max_files", type=int, default=None)
    args = parser.parse_args()
    
    # 1. Load Data
    all_trajs = load_trajectories(args.final_dir, args.max_files)

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

if __name__ == "__main__":
    main()