"""
Visualize Kalman Filter predictions vs ground truth.

Creates comparison plots for trajectory predictions.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from kalman_filter.kalman_filter import TrajectoryKalmanFilter, KalmanFilterParams


# Column indices
LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI = list(range(9))


def denormalize_positions(positions: np.ndarray, 
                         lat_min: float = 54.0, lat_max: float = 59.0,
                         lon_min: float = 5.0, lon_max: float = 17.0) -> np.ndarray:
    """Convert normalized [0,1] positions to real lat/lon."""
    denorm = positions.copy()
    denorm[:, 0] = denorm[:, 0] * (lat_max - lat_min) + lat_min  # LAT
    denorm[:, 1] = denorm[:, 1] * (lon_max - lon_min) + lon_min  # LON
    return denorm


def plot_single_trajectory(ax, window, target, prediction, denorm=True):
    """Plot a single trajectory comparison."""
    
    if denorm:
        window = denormalize_positions(window[:, [LAT, LON]])
        target = denormalize_positions(target)
        prediction = denormalize_positions(prediction)
    else:
        window = window[:, [LAT, LON]]
    
    # Plot historical window
    ax.plot(window[:, 1], window[:, 0], 'b-', linewidth=2, label='Historical', alpha=0.7)
    ax.plot(window[-1, 1], window[-1, 0], 'bo', markersize=8, label='Last Known')
    
    # Plot ground truth
    ax.plot(target[:, 1], target[:, 0], 'g-', linewidth=2, label='Ground Truth', alpha=0.7)
    ax.plot(target[-1, 1], target[-1, 0], 'gs', markersize=10, label='True Endpoint')
    
    # Plot prediction
    ax.plot(prediction[:, 1], prediction[:, 0], 'r--', linewidth=2, label='Kalman Prediction', alpha=0.7)
    ax.plot(prediction[-1, 1], prediction[-1, 0], 'r^', markersize=10, label='Predicted Endpoint')
    
    # Connect last known to first predicted
    ax.plot([window[-1, 1], prediction[0, 1]], 
            [window[-1, 0], prediction[0, 0]], 'r:', alpha=0.5)
    
    ax.set_xlabel('Longitude (°)' if denorm else 'Longitude (normalized)')
    ax.set_ylabel('Latitude (°)' if denorm else 'Latitude (normalized)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')


def visualize_predictions(final_dir: str, 
                         window_size: int = 64,
                         horizon: int = 12,
                         n_examples: int = 6,
                         output_file: str = "data/figures/kalman_predictions.png"):
    """
    Create visualization of Kalman Filter predictions.
    
    Args:
        final_dir: Directory with processed pickles
        window_size: Input window size
        horizon: Prediction horizon
        n_examples: Number of examples to plot
        output_file: Where to save the figure
    """
    # Load trajectories
    paths = [Path(final_dir) / f for f in os.listdir(final_dir) 
             if f.endswith("_processed.pkl")]
    
    # Create filter
    params = KalmanFilterParams(
        process_noise_pos=1e-5,
        process_noise_vel=1e-4,
        measurement_noise=1e-4,
        dt=300.0
    )
    kf = TrajectoryKalmanFilter(params)
    
    # Select examples
    np.random.seed(42)
    examples = []
    
    for path in np.random.choice(paths, min(len(paths), 50), replace=False):
        try:
            with open(path, "rb") as f:
                item = pickle.load(f)
            traj = item["traj"]
            
            if len(traj) >= window_size + horizon + 20:
                # Take middle section
                mid = len(traj) // 2
                window = traj[mid:mid + window_size]
                target = traj[mid + window_size:mid + window_size + horizon, [LAT, LON]]
                
                # Predict
                prediction = kf.predict(window, horizon)
                
                # Compute error
                error = np.linalg.norm(prediction - target, axis=-1).mean()
                
                examples.append({
                    'window': window,
                    'target': target,
                    'prediction': prediction,
                    'error': error,
                    'mmsi': item['mmsi']
                })
                
                if len(examples) >= n_examples * 2:
                    break
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue
    
    if len(examples) == 0:
        print("No valid examples found")
        return
    
    # Sort by error and select best, median, worst
    examples.sort(key=lambda x: x['error'])
    
    selected = []
    if len(examples) >= n_examples:
        # Best 2
        selected.extend(examples[:2])
        # Median 2
        mid = len(examples) // 2
        selected.extend(examples[mid-1:mid+1])
        # Worst 2
        selected.extend(examples[-2:])
    else:
        selected = examples[:n_examples]
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    categories = ['Best', 'Best', 'Median', 'Median', 'Worst', 'Worst']
    
    for i, (ex, cat) in enumerate(zip(selected, categories)):
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        
        plot_single_trajectory(ax, ex['window'], ex['target'], ex['prediction'])
        
        error_m = ex['error'] * 111000  # Rough conversion to meters (1° ≈ 111km)
        ax.set_title(f"{cat} Case - MMSI {ex['mmsi']}\nADE: {ex['error']:.6f} ({error_m:.0f}m)",
                    fontsize=10)
    
    fig.suptitle(f'Kalman Filter Trajectory Predictions\n'
                f'Window: {window_size} steps (5h20m), Horizon: {horizon} steps (1h)',
                fontsize=14, fontweight='bold')
    
    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    plt.close()


def plot_error_distribution(final_dir: str,
                           window_size: int = 64,
                           horizon: int = 12,
                           n_samples: int = 500,
                           output_file: str = "data/figures/kalman_error_dist.png"):
    """Plot distribution of prediction errors."""
    
    # Load trajectories
    paths = [Path(final_dir) / f for f in os.listdir(final_dir) 
             if f.endswith("_processed.pkl")]
    
    kf = TrajectoryKalmanFilter()
    
    errors = []
    per_step_errors = [[] for _ in range(horizon)]
    
    print(f"Computing error distribution on {min(len(paths), n_samples)} trajectories...")
    
    for path in paths[:n_samples]:
        try:
            with open(path, "rb") as f:
                item = pickle.load(f)
            traj = item["traj"]
            
            if len(traj) >= window_size + horizon + 10:
                mid = len(traj) // 2
                window = traj[mid:mid + window_size]
                target = traj[mid + window_size:mid + window_size + horizon, [LAT, LON]]
                
                prediction = kf.predict(window, horizon)
                
                # Overall error
                error = np.linalg.norm(prediction - target, axis=-1).mean()
                errors.append(error)
                
                # Per-step errors
                for h in range(horizon):
                    err_h = np.linalg.norm(prediction[h] - target[h])
                    per_step_errors[h].append(err_h)
        except:
            continue
    
    errors = np.array(errors)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Error distribution histogram
    ax = axes[0]
    errors_m = errors * 111000  # Convert to meters
    ax.hist(errors_m, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(np.median(errors_m), color='r', linestyle='--', linewidth=2, label=f'Median: {np.median(errors_m):.0f}m')
    ax.axvline(np.mean(errors_m), color='g', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors_m):.0f}m')
    ax.set_xlabel('Average Displacement Error (meters)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Prediction Errors')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Per-horizon error growth
    ax = axes[1]
    horizon_means = [np.mean(errs) * 111000 for errs in per_step_errors]
    horizon_stds = [np.std(errs) * 111000 for errs in per_step_errors]
    
    steps = np.arange(1, horizon + 1)
    ax.plot(steps, horizon_means, 'b-o', linewidth=2, markersize=6, label='Mean Error')
    ax.fill_between(steps, 
                    np.array(horizon_means) - np.array(horizon_stds),
                    np.array(horizon_means) + np.array(horizon_stds),
                    alpha=0.3, label='±1 Std Dev')
    ax.set_xlabel('Prediction Step (5 min intervals)')
    ax.set_ylabel('Error (meters)')
    ax.set_title('Error Growth Over Prediction Horizon')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(steps)
    
    plt.tight_layout()
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Error distribution plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize Kalman Filter predictions")
    parser.add_argument("--final_dir", default="data/map_reduce_final")
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--n_examples", type=int, default=6)
    parser.add_argument("--n_samples", type=int, default=500)
    
    args = parser.parse_args()
    
    print("Generating Kalman Filter visualizations...")
    
    # Trajectory predictions
    visualize_predictions(
        args.final_dir,
        window_size=args.window,
        horizon=args.horizon,
        n_examples=args.n_examples
    )
    
    # Error distributions
    plot_error_distribution(
        args.final_dir,
        window_size=args.window,
        horizon=args.horizon,
        n_samples=args.n_samples
    )
    
    print("\nDone! Check data/figures/ for output.")


if __name__ == "__main__":
    main()
