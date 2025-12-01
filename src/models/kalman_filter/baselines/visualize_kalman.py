"""
Visualize Kalman Filter predictions vs ground truth.

Creates comparison plots for trajectory predictions.
"""

from __future__ import annotations
from pathlib import Path

import argparse
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.models.kalman_filter.kalman_filter import TrajectoryKalmanFilter, KalmanFilterParams


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


def add_basemap(ax, water_mask_path: str | None = None, bounds: tuple[float, float, float, float] | None = None):
    """
    Adds a basemap to the plot, either from a water mask or contextily.
    If bounds are provided, zooms into that area.
    """
    
    # Define the default geographic bounds for the Denmark region
    LAT_MIN_DEFAULT, LAT_MAX_DEFAULT = 54.0, 59.0
    LON_MIN_DEFAULT, LON_MAX_DEFAULT = 5.0, 17.0

    if bounds:
        lon_min, lon_max, lat_min, lat_max = bounds
    else:
        lon_min, lon_max, lat_min, lat_max = LON_MIN_DEFAULT, LON_MAX_DEFAULT, LAT_MIN_DEFAULT, LAT_MAX_DEFAULT

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect('equal')

    # Try to use roaring-landmask for a high-quality, self-contained map
    try:
        from roaring_landmask import RoaringLandmask
        print("Adding basemap with roaring-landmask...")
        # Initialize the landmask (caches data after the first call)
        mask = RoaringLandmask()
        mask.plot_land(ax, facecolor='#c5e3ff', edgecolor='none')
        ax.set_facecolor('#aadaff')
        return
    except ImportError:
        print("roaring-landmask not found. Falling back to contextily or image mask.")
    except Exception as exc:
        print(f"roaring-landmask failed ({exc}). Falling back to contextily or image mask.")

    # Fallback to contextily if available
    try:
        import contextily as ctx
        print("Adding basemap with contextily...")
        ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik, attribution_size=5)
        return
    except ImportError:
        print("Contextily not found. Falling back to water mask image.")
    except Exception as exc:
        print(f"Contextily basemap failed ({exc}). Falling back to water mask image.")

    # Final fallback: water mask image or plain background
    if water_mask_path and Path(water_mask_path).exists():
        try:
            mask_img = plt.imread(water_mask_path)
            ax.imshow(
                mask_img,
                extent=(LON_MIN_DEFAULT, LON_MAX_DEFAULT, LAT_MIN_DEFAULT, LAT_MAX_DEFAULT),
                origin="lower",
                aspect="auto"
            )
        except Exception as exc:
            print(f"Could not load water mask ({exc}). Using plain background.")
            ax.set_facecolor('#aadaff')
    else:
        print("No water mask found. Using plain background.")
        ax.set_facecolor('#aadaff')


def visualize_predictions(final_dir: str,
                          output_dir: str,
                          water_mask_path: str | None,
                          window_size: int = 64,
                          horizon: int = 12,
                          n_examples: int = 6,
                          max_candidates: int = 200,
                          rng_seed: int = 42):
    """Render Kalman Filter predictions for representative trajectories."""

    paths = [Path(final_dir) / f for f in os.listdir(final_dir) if f.endswith("_processed.pkl")]
    if not paths:
        print(f"No *_processed.pkl files found inside {final_dir}")
        return

    params = KalmanFilterParams(
        process_noise_pos=1e-5,
        process_noise_vel=1e-4,
        measurement_noise=1e-4,
        dt=300.0,
    )
    kf = TrajectoryKalmanFilter(params)

    rng = np.random.default_rng(rng_seed)
    shuffled_indices = rng.permutation(len(paths))

    examples: list[dict] = []
    target_pool = max(n_examples * 4, n_examples + 8)

    for idx in shuffled_indices:
        path = paths[int(idx)]
        if len(examples) >= target_pool:
            break

        try:
            with open(path, "rb") as f:
                item = pickle.load(f)
        except Exception as exc:
            print(f"Failed to load {path}: {exc}")
            continue

        traj = item.get("traj")
        if traj is None or len(traj) < window_size + horizon:
            continue

        n_candidates = len(traj) - window_size - horizon + 1
        if n_candidates <= 0:
            continue

        starts = rng.choice(n_candidates, size=min(n_candidates, 5), replace=False)

        for start_idx in starts:
            window = traj[start_idx:start_idx + window_size]
            if window.shape[0] != window_size:
                continue

            target = traj[start_idx + window_size:start_idx + window_size + horizon, [LAT, LON]]
            if target.shape[0] != horizon:
                continue

            try:
                prediction = kf.predict(window, horizon)
            except Exception as exc:
                print(f"Kalman prediction failed for {path.name} (idx {start_idx}): {exc}")
                continue

            error = float(np.mean(np.linalg.norm(prediction - target, axis=-1)))
            examples.append({
                "window": window,
                "target": target,
                "prediction": prediction,
                "error": error,
                "mmsi": item.get("mmsi", "unknown"),
                "path": path.name,
                "start_idx": int(start_idx),
            })

            if len(examples) >= target_pool:
                break

    if not examples:
        print("No valid trajectory windows found for visualization.")
        return

    examples.sort(key=lambda x: x["error"])

    def select_examples(candidates: list[dict], k: int) -> list[tuple[dict, str]]:
        k = max(1, min(k, len(candidates)))
        linspace_idx = np.linspace(0, len(candidates) - 1, k)
        ordered_indices = []
        for idx in linspace_idx:
            rounded = int(round(idx))
            if rounded not in ordered_indices:
                ordered_indices.append(rounded)
        while len(ordered_indices) < k:
            for extra in range(len(candidates)):
                if extra not in ordered_indices:
                    ordered_indices.append(extra)
                if len(ordered_indices) == k:
                    break

        selected_pairs: list[tuple[dict, str]] = []
        for rank, idx in enumerate(ordered_indices, start=1):
            label: str
            if rank == 1:
                label = "best"
            elif rank == len(ordered_indices):
                label = "worst"
            elif rank == (len(ordered_indices) + 1) // 2:
                label = "median"
            else:
                label = f"sample_{rank}"
            selected_pairs.append((candidates[idx], label))
        return selected_pairs

    selected = select_examples(examples, n_examples)
    print(f"Generating {len(selected)} trajectory plots...")

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    for rank, (ex, label) in enumerate(selected, start=1):
        fig, ax = plt.subplots(figsize=(10, 10))

        window_deg = denormalize_positions(ex["window"][:, [LAT, LON]])
        target_deg = denormalize_positions(ex["target"])
        prediction_deg = denormalize_positions(ex["prediction"])

        all_points = np.vstack([window_deg, target_deg, prediction_deg])
        lat_min, lon_min = all_points.min(axis=0)
        lat_max, lon_max = all_points.max(axis=0)

        lat_pad = max((lat_max - lat_min) * 0.2, 1e-3)
        lon_pad = max((lon_max - lon_min) * 0.2, 1e-3)

        bounds = (
            lon_min - lon_pad,
            lon_max + lon_pad,
            lat_min - lat_pad,
            lat_max + lat_pad,
        )

        add_basemap(ax, water_mask_path, bounds=bounds)
        plot_single_trajectory(ax, ex["window"], ex["target"], ex["prediction"], denorm=True)

        error_m = ex["error"] * 111000
        title = (
            f"{label.capitalize()} example – MMSI {ex['mmsi']}\n"
            f"Start idx {ex['start_idx']} in {ex['path']} | ADE ≈ {error_m:.0f} m"
        )
        ax.set_title(title, fontsize=12)
        fig.suptitle("Kalman Filter Trajectory Prediction", fontsize=16, fontweight="bold")
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))

        file_label = label.replace(" ", "_")
        filename = (
            f"kalman_prediction_{rank:02d}_{file_label}_mmsi-{ex['mmsi']}_idx-{ex['start_idx']}.png"
        )
        output_path = output_dir_path / filename
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"  -> Saved {output_path}")
        plt.close(fig)

    print("Finished exporting trajectory plots.")


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
    default_mask = Path(__file__).resolve().parents[1] / "assets" / "water_mask.png"
    parser.add_argument("--water_mask", default=str(default_mask), help="Path to water mask image")
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
    
    # Error distributions are no longer generated by default to avoid warnings
    # on smaller datasets like the test set.
    # plot_error_distribution(
    #     args.final_dir,
    #     output_dir=args.output_dir,
    #     window_size=args.window,
    #     horizon=args.horizon,
    #     n_samples=args.n_samples
    # )
    
    print(f"\nDone! Check {args.output_dir} for output.")


if __name__ == "__main__":
    main()
