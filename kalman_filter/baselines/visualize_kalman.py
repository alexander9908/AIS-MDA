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
    
    # Denormalize positions to real lat/lon for plotting
    if denorm:
        window_deg = denormalize_positions(window[:, [LAT, LON]])
        target_deg = denormalize_positions(target)
        prediction_deg = denormalize_positions(prediction)
    else:
        # If data is already in degrees, just use it
        window_deg = window[:, [LAT, LON]]
        target_deg = target
        prediction_deg = prediction

    # Plot historical window
    ax.plot(window_deg[:, 1], window_deg[:, 0], 'b-', linewidth=1.5, label='History (5h 20m)', alpha=0.8)
    ax.plot(window_deg[-1, 1], window_deg[-1, 0], 'bo', markersize=6, label='Last Known Position')
    
    # Plot ground truth future
    ax.plot(target_deg[:, 1], target_deg[:, 0], 'g-', linewidth=2, label='True Future (1h)', alpha=0.8)
    ax.plot(target_deg[-1, 1], target_deg[-1, 0], 'gs', markersize=8, label='True Endpoint')
    
    # Plot prediction
    ax.plot(prediction_deg[:, 1], prediction_deg[:, 0], 'r--', linewidth=2, label='Kalman Prediction', alpha=0.8)
    ax.plot(prediction_deg[-1, 1], prediction_deg[-1, 0], 'r^', markersize=8, label='Predicted Endpoint')
    
    # Connect last known to first predicted
    ax.plot([window_deg[-1, 1], prediction_deg[0, 1]], 
            [window_deg[-1, 0], prediction_deg[0, 0]], 'r:', alpha=0.6)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)


def add_basemap(ax, water_mask_path: str | None = None):
    """Adds a basemap to the plot, either from a water mask or contextily."""
    
    # Define the geographic bounds for the Denmark region
    LAT_MIN, LAT_MAX = 54.0, 59.0
    LON_MIN, LON_MAX = 5.0, 17.0
    
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect('equal')

    # Try to use contextily for a live map
    try:
        import contextily as ctx
        print("Adding basemap with contextily...")
        ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik, attribution_size=5)
    except ImportError:
        print("Contextily not found. Falling back to water mask.")
        if water_mask_path and Path(water_mask_path).exists():
            try:
                mask = plt.imread(water_mask_path)
                ax.imshow(mask, extent=(LON_MIN, LON_MAX, LAT_MIN, LAT_MAX), origin="lower", aspect="auto")
            except Exception as e:
                print(f"Could not load water mask: {e}. Using plain background.")
                ax.set_facecolor('#c5e3ff') # Light blue for water
        else:
            print("No water mask found. Using plain background.")
            ax.set_facecolor('#c5e3ff') # Light blue for water


def visualize_predictions(final_dir: str, 
                         output_dir: str,
                         water_mask_path: str | None,
                         window_size: int = 64,
                         horizon: int = 12,
                         n_examples: int = 6):
    """
    Create visualization of Kalman Filter predictions.
    
    Args:
        final_dir: Directory with processed pickles
        output_dir: Directory to save the figures
        window_size: Input window size
        horizon: Prediction horizon
        n_examples: Number of examples to plot
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
    
    # Ensure paths is a numpy array for np.random.choice
    path_array = np.array(paths)
    
    for path in np.random.choice(path_array, min(len(path_array), 50), replace=False):
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
    categories = []
    if len(examples) >= n_examples:
        # Best 2
        selected.extend(examples[:2])
        categories.extend(['Best_1', 'Best_2'])
        # Median 2
        mid = len(examples) // 2
        selected.extend(examples[mid-1:mid+1])
        categories.extend(['Median_1', 'Median_2'])
        # Worst 2
        selected.extend(examples[-2:])
        categories.extend(['Worst_1', 'Worst_2'])
    else:
        selected = examples[:n_examples]
        categories = [f"Example_{i+1}" for i in range(len(selected))]

    print(f"Generating {len(selected)} individual trajectory plots...")

    for ex, cat in zip(selected, categories):
        # Create a new figure for each plot
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Add basemap first
        add_basemap(ax, water_mask_path)
        
        # Plot the trajectory
        plot_single_trajectory(ax, ex['window'], ex['target'], ex['prediction'])
        
        error_m = ex['error'] * 111000  # Rough conversion to meters (1° ≈ 111km)
        ax.set_title(f"{cat.split('_')[0]} Case - MMSI {ex['mmsi']}\nADE: {ex['error']:.6f} ({error_m:.0f}m)",
                    fontsize=12)
        
        fig.suptitle(f'Kalman Filter Trajectory Prediction\n'
                     f'Window: {window_size} steps, Horizon: {horizon} steps',
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))

        # Save individual plot
        output_path = Path(output_dir) / f"kalman_prediction_{cat.lower()}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  -> Saved {output_path}")
        plt.close(fig)

    print("Individual trajectory plots saved.")


def plot_error_distribution(final_dir: str,
                           output_dir: str,
                           window_size: int = 64,
                           horizon: int = 12,
                           n_samples: int = 500):
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
    
    output_path = Path(output_dir) / "kalman_error_dist.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Error distribution plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize Kalman Filter predictions")
    parser.add_argument("--final_dir", default="data/map_reduce_final")
    parser.add_argument("--output_dir", default="data/figures/kalman", help="Directory to save visualizations")
    parser.add_argument("--water_mask", default="kalman_filter/assets/water_mask.png", help="Path to water mask image")
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--n_examples", type=int, default=6)
    parser.add_argument("--n_samples", type=int, default=500)
    
    args = parser.parse_args()
    
    print("Generating Kalman Filter visualizations...")
    
    # Trajectory predictions
    visualize_predictions(
        args.final_dir,
        output_dir=args.output_dir,
        water_mask_path=args.water_mask,
        window_size=args.window,
        horizon=args.horizon,
        n_examples=args.n_examples
    )
    
    # Error distributions
    plot_error_distribution(
        args.final_dir,
        output_dir=args.output_dir,
        window_size=args.window,
        horizon=args.horizon,
        n_samples=args.n_samples
    )
    
    print(f"\nDone! Check {args.output_dir} for output.")


if __name__ == "__main__":
    main()
