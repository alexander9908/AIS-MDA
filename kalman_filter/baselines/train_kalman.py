"""
Train and evaluate Kalman Filter baseline on AIS trajectory data.

Usage:
    python -m src.baselines.train_kalman --final_dir data/map_reduce_final --window 64 --horizon 12
"""

from __future__ import annotations
import argparse
import pickle
import os
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kalman_filter.kalman_filter import TrajectoryKalmanFilter, KalmanFilterParams, tune_kalman_filter
from src.eval.metrics_traj import ade, fde


# Column indices for MapReduce processed data
LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI = list(range(9))


def load_trajectories(data_dir: str, max_files: int | None = None) -> List[np.ndarray]:
    """
    Load processed trajectories from pickle files.
    
    Args:
        data_dir: Directory containing *_processed.pkl files
        max_files: Maximum number of files to load (None = all)
        
    Returns:
        List of trajectory arrays, each shape (T, 9)
    """
    if not Path(data_dir).exists():
        print(f"Warning: Directory not found: {data_dir}")
        return []
        
    paths = [Path(data_dir) / f for f in os.listdir(data_dir) 
             if f.endswith("_processed.pkl")]
    
    if max_files:
        paths = paths[:max_files]
    
    trajectories = []
    print(f"Loading {len(paths)} trajectory files...")
    
    for path in tqdm(paths):
        try:
            with open(path, "rb") as f:
                item = pickle.load(f)
            traj = item["traj"]  # shape: (T, 9)
            
            # Basic validation
            if len(traj) >= 20:  # Minimum length
                trajectories.append(traj)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue
    
    print(f"Loaded {len(trajectories)} valid trajectories")
    return trajectories


def split_trajectories(trajectories: List[np.ndarray], 
                       val_frac: float = 0.2,
                       test_frac: float = 0.1,
                       seed: int = 42) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Split trajectories into train/val/test sets.
    
    Args:
        trajectories: List of trajectories
        val_frac: Fraction for validation
        test_frac: Fraction for test
        seed: Random seed
        
    Returns:
        train_trajs, val_trajs, test_trajs
    """
    rng = np.random.default_rng(seed)
    n = len(trajectories)
    
    # Shuffle
    indices = rng.permutation(n)
    
    # Split
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_test - n_val
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    train_trajs = [trajectories[i] for i in train_idx]
    val_trajs = [trajectories[i] for i in val_idx]
    test_trajs = [trajectories[i] for i in test_idx]
    
    print(f"Split: Train={len(train_trajs)}, Val={len(val_trajs)}, Test={len(test_trajs)}")
    
    return train_trajs, val_trajs, test_trajs


def create_windows(trajectories: List[np.ndarray], 
                   window_size: int, 
                   horizon: int,
                   max_windows: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows from trajectories.
    
    Args:
        trajectories: List of trajectories
        window_size: Input window size
        horizon: Prediction horizon
        max_windows: Maximum windows to create (for speed)
        
    Returns:
        X: Input windows (N, window_size, 9)
        Y: Target positions (N, horizon, 2) - [lat, lon] only
    """
    X_list = []
    Y_list = []
    
    for traj in trajectories:
        if len(traj) < window_size + horizon:
            continue
        
        # Create all possible windows from this trajectory
        for i in range(len(traj) - window_size - horizon + 1):
            X_list.append(traj[i:i + window_size])
            # Target is future lat/lon positions
            Y_list.append(traj[i + window_size:i + window_size + horizon, [LAT, LON]])
        
        # Early exit if we have enough windows
        if max_windows and len(X_list) >= max_windows:
            break
    
    if max_windows:
        X_list = X_list[:max_windows]
        Y_list = Y_list[:max_windows]
    
    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    
    print(f"Created {len(X)} windows: X{X.shape}, Y{Y.shape}")
    return X, Y


def evaluate_kalman(kf: TrajectoryKalmanFilter,
                    X: np.ndarray,
                    Y: np.ndarray,
                    batch_size: int = 100) -> dict:
    """
    Evaluate Kalman Filter on test data.
    
    Args:
        kf: Trained Kalman Filter
        X: Input windows (N, T, F)
        Y: Target positions (N, H, 2)
        batch_size: Batch size for prediction
        
    Returns:
        Dictionary with metrics
    """
    n_samples = len(X)
    all_preds = []
    
    print("Evaluating Kalman Filter...")
    for i in tqdm(range(0, n_samples, batch_size)):
        batch_X = X[i:i + batch_size]
        batch_pred = kf.predict_batch(batch_X, horizon=Y.shape[1])
        all_preds.append(batch_pred)
    
    predictions = np.concatenate(all_preds, axis=0)
    
    # Compute metrics
    ade_val = float(ade(predictions, Y))
    fde_val = float(fde(predictions, Y))
    
    # Per-horizon ADE
    horizon = Y.shape[1]
    per_horizon_ade = []
    for h in range(horizon):
        err = np.linalg.norm(predictions[:, h, :] - Y[:, h, :], axis=-1)
        per_horizon_ade.append(float(err.mean()))
    
    results = {
        "ade": ade_val,
        "fde": fde_val,
        "per_horizon_ade": per_horizon_ade,
        "n_samples": n_samples
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train Kalman Filter baseline for AIS trajectory prediction")
    
    # Arguments for pre-split directories
    parser.add_argument("--train_dir", help="Directory with pre-split training data")
    parser.add_argument("--val_dir", help="Directory with pre-split validation data")
    parser.add_argument("--test_dir", help="Directory with pre-split test data")

    # Original arguments for automatic splitting
    parser.add_argument("--final_dir", help="Directory with all processed pickles (for automatic splitting)")
    
    parser.add_argument("--out_dir", default="data/checkpoints", help="Output directory for results")
    parser.add_argument("--window", type=int, default=64, help="Input window size")
    parser.add_argument("--horizon", type=int, default=12, help="Prediction horizon")
    parser.add_argument("--max_files", type=int, default=None, help="Max trajectory files to load per split")
    parser.add_argument("--max_windows", type=int, default=999999, help="Max windows for eval")
    parser.add_argument("--tune", action="store_true", help="Tune hyperparameters")
    parser.add_argument("--val_frac", type=float, default=0.2, help="Validation fraction (if not using pre-split dirs)")
    parser.add_argument("--test_frac", type=float, default=0.1, help="Test fraction (if not using pre-split dirs)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Kalman Filter parameters
    parser.add_argument("--process_noise_pos", type=float, default=1e-5)
    parser.add_argument("--process_noise_vel", type=float, default=1e-4)
    parser.add_argument("--measurement_noise", type=float, default=1e-4)
    
    args = parser.parse_args()

    # --- Data Loading ---
    if args.train_dir and args.val_dir and args.test_dir:
        print("Loading data from pre-split directories...")
        train_trajs = load_trajectories(args.train_dir, max_files=args.max_files)
        val_trajs = load_trajectories(args.val_dir, max_files=args.max_files)
        test_trajs = load_trajectories(args.test_dir, max_files=args.max_files)
        print(f"Loaded splits: Train={len(train_trajs)}, Val={len(val_trajs)}, Test={len(test_trajs)}")
    elif args.final_dir:
        print("Loading data from single directory and splitting automatically...")
        trajectories = load_trajectories(args.final_dir, max_files=args.max_files)
        if len(trajectories) == 0:
            print("No trajectories loaded. Exiting.")
            return
        train_trajs, val_trajs, test_trajs = split_trajectories(
            trajectories, 
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed
        )
    else:
        raise ValueError("You must provide either --train_dir, --val_dir, and --test_dir OR --final_dir.")

    # Initialize results dictionaries
    val_results, test_results = {}, {}
    
    # Tune parameters if requested
    if args.tune:
        print("\n=== Tuning Kalman Filter Parameters ===")
        best_params = tune_kalman_filter(
            train_trajs, 
            val_trajs,
            window_size=args.window,
            horizon=args.horizon
        )
    else:
        # Use provided parameters
        best_params = KalmanFilterParams(
            process_noise_pos=args.process_noise_pos,
            process_noise_vel=args.process_noise_vel,
            measurement_noise=args.measurement_noise,
            dt=300.0  # 5 minutes for MapReduce data
        )
    
    # Create Kalman Filter
    kf = TrajectoryKalmanFilter(best_params)
    
    # Evaluate on validation set
    print("\n=== Validation Evaluation ===")
    X_val, Y_val = create_windows(val_trajs, args.window, args.horizon, max_windows=args.max_windows)
    if len(X_val) > 0:
        val_results = evaluate_kalman(kf, X_val, Y_val)
        print(f"Validation - ADE: {val_results['ade']:.6f}, FDE: {val_results['fde']:.6f}")
    
    # Evaluate on test set
    print("\n=== Test Evaluation ===")
    X_test, Y_test = create_windows(test_trajs, args.window, args.horizon, max_windows=args.max_windows)
    if len(X_test) > 0:
        test_results = evaluate_kalman(kf, X_test, Y_test)
        print(f"Test - ADE: {test_results['ade']:.6f}, FDE: {test_results['fde']:.6f}")
        
        # Print per-horizon results
        print("\nPer-Horizon ADE:")
        for h, ade_h in enumerate(test_results['per_horizon_ade'], 1):
            print(f"  Step {h}: {ade_h:.6f}")
    
    # Save results
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results_dict = {
        "model": "kalman_filter",
        "window": args.window,
        "horizon": args.horizon,
        "parameters": {
            "process_noise_pos": best_params.process_noise_pos,
            "process_noise_vel": best_params.process_noise_vel,
            "measurement_noise": best_params.measurement_noise,
            "dt": best_params.dt
        },
        "validation": val_results if len(X_val) > 0 else None,
        "test": test_results if len(X_test) > 0 else None
    }
    
    # Save to metrics directory
    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = metrics_dir / "kalman_filter.json"
    
    with open(metrics_file, "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResults saved to {metrics_file}")
    
    # Also save a simple text summary
    summary_file = out_dir / "kalman_filter_summary.txt"
    with open(summary_file, "w") as f:
        f.write("Kalman Filter Baseline Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Window size: {args.window}\n")
        f.write(f"  Horizon: {args.horizon}\n")
        f.write(f"  Process noise (pos): {best_params.process_noise_pos}\n")
        f.write(f"  Process noise (vel): {best_params.process_noise_vel}\n")
        f.write(f"  Measurement noise: {best_params.measurement_noise}\n\n")
        
        if len(X_test) > 0:
            f.write(f"Test Results:\n")
            f.write(f"  ADE: {test_results['ade']:.6f}\n")
            f.write(f"  FDE: {test_results['fde']:.6f}\n")
            f.write(f"  Samples: {test_results['n_samples']}\n")
    
    print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()
