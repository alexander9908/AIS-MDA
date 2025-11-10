"""
Compare Kalman Filter baseline with neural network models.

Loads predictions from both models and generates comparative metrics.

Usage:
    python -m kalman_filter.baselines.compare_models --final_dir data/map_reduce_final
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import numpy as np
from tqdm import tqdm

from kalman_filter.baselines.train_kalman import load_trajectories, split_trajectories, create_windows, evaluate_kalman
from kalman_filter.kalman_filter import TrajectoryKalmanFilter, KalmanFilterParams


def compare_with_existing_results(kalman_results: dict, metrics_dir: Path):
    """
    Compare Kalman Filter results with existing model results.
    
    Args:
        kalman_results: Kalman Filter evaluation results
        metrics_dir: Directory containing other model metrics
    """
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    # Load existing results
    results = {"Kalman Filter": kalman_results}
    
    # Try to load TPTrans results
    tptrans_files = list(metrics_dir.glob("traj_tptrans*.json"))
    if tptrans_files:
        with open(tptrans_files[0], "r") as f:
            results["TPTrans"] = json.load(f)
    
    # Try to load GRU results
    gru_files = list(metrics_dir.glob("traj_gru*.json"))
    if gru_files:
        with open(gru_files[0], "r") as f:
            results["GRU"] = json.load(f)
    
    # Print comparison table
    print(f"\n{'Model':<20} {'ADE':<12} {'FDE':<12} {'Improvement vs KF':<20}")
    print("-" * 70)
    
    kf_ade = kalman_results.get("ade", 0)
    kf_fde = kalman_results.get("fde", 0)
    
    for model_name, result in results.items():
        ade = result.get("ade", 0)
        fde = result.get("fde", 0)
        
        if model_name == "Kalman Filter":
            improvement = "-"
        else:
            if kf_ade > 0:
                ade_improv = ((kf_ade - ade) / kf_ade) * 100
                fde_improv = ((kf_fde - fde) / kf_fde) * 100
                improvement = f"{ade_improv:+.1f}% ADE, {fde_improv:+.1f}% FDE"
            else:
                improvement = "N/A"
        
        print(f"{model_name:<20} {ade:<12.6f} {fde:<12.6f} {improvement:<20}")
    
    # Per-horizon comparison if available
    if "per_horizon_ade" in kalman_results:
        print("\n" + "=" * 70)
        print("PER-HORIZON ADE COMPARISON")
        print("=" * 70)
        
        horizon = len(kalman_results["per_horizon_ade"])
        print(f"\n{'Step':<8}", end="")
        for model_name in results.keys():
            print(f"{model_name:<15}", end="")
        print()
        print("-" * 70)
        
        for h in range(horizon):
            print(f"{h+1:<8}", end="")
            for model_name, result in results.items():
                if "per_horizon_ade" in result:
                    ade_h = result["per_horizon_ade"][h]
                    print(f"{ade_h:<15.6f}", end="")
                else:
                    print(f"{'N/A':<15}", end="")
            print()


def analyze_error_by_scenario(kalman_filter: TrajectoryKalmanFilter,
                              trajectories: list,
                              window_size: int,
                              horizon: int) -> dict:
    """
    Analyze Kalman Filter performance by scenario.
    
    Categorizes trajectories by:
    - Speed: Low (<5kn), Medium (5-15kn), High (>15kn)
    - Maneuver: Straight, Turning
    
    Args:
        kalman_filter: Fitted Kalman Filter
        trajectories: List of test trajectories
        window_size: Window size
        horizon: Prediction horizon
        
    Returns:
        Dictionary with scenario-based metrics
    """
    print("\n" + "=" * 70)
    print("SCENARIO-BASED ANALYSIS")
    print("=" * 70)
    
    from src.eval.metrics_traj import ade
    
    scenarios = {
        "straight_low_speed": [],
        "straight_high_speed": [],
        "turning": []
    }
    
    LAT, LON, SOG = 0, 1, 2
    SPEED_MAX = 30.0  # Denormalization constant
    
    for traj in tqdm(trajectories[:200], desc="Analyzing scenarios"):
        if len(traj) < window_size + horizon:
            continue
        
        # Take middle section
        mid = len(traj) // 2
        if mid < window_size + horizon:
            continue
            
        window = traj[mid:mid + window_size]
        target = traj[mid + window_size:mid + window_size + horizon, [LAT, LON]]
        
        # Compute average speed (denormalized)
        avg_speed = window[:, SOG].mean() * SPEED_MAX
        
        # Detect turning (change in direction)
        lat_diff = np.diff(window[:, LAT])
        lon_diff = np.diff(window[:, LON])
        headings = np.arctan2(lat_diff, lon_diff)
        heading_change = np.abs(np.diff(headings)).sum()
        is_turning = heading_change > 0.5  # Arbitrary threshold
        
        # Categorize
        if is_turning:
            scenario = "turning"
        elif avg_speed < 5.0:
            scenario = "straight_low_speed"
        else:
            scenario = "straight_high_speed"
        
        # Predict
        pred = kalman_filter.predict(window, horizon)
        error = ade(pred[np.newaxis, :, :], target[np.newaxis, :, :])
        
        scenarios[scenario].append(error)
    
    # Compute statistics
    results = {}
    print(f"\n{'Scenario':<25} {'Count':<10} {'ADE':<12} {'Std':<12}")
    print("-" * 70)
    
    for scenario, errors in scenarios.items():
        if len(errors) > 0:
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            results[scenario] = {
                "count": len(errors),
                "ade": float(mean_error),
                "std": float(std_error)
            }
            print(f"{scenario:<25} {len(errors):<10} {mean_error:<12.6f} {std_error:<12.6f}")
        else:
            results[scenario] = {"count": 0, "ade": 0.0, "std": 0.0}
            print(f"{scenario:<25} {0:<10} {'N/A':<12} {'N/A':<12}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare Kalman Filter with neural network models")
    parser.add_argument("--final_dir", default="data/map_reduce_final")
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--max_files", type=int, default=500)
    parser.add_argument("--max_windows", type=int, default=10000)
    parser.add_argument("--analyze_scenarios", action="store_true", help="Analyze by scenario")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("KALMAN FILTER vs NEURAL NETWORKS COMPARISON")
    print("=" * 70)
    
    # Load data
    trajectories = load_trajectories(args.final_dir, max_files=args.max_files)
    
    if len(trajectories) == 0:
        print("No trajectories loaded. Exiting.")
        return
    
    # Split
    train_trajs, val_trajs, test_trajs = split_trajectories(trajectories)
    
    # Create Kalman Filter with default params
    params = KalmanFilterParams(
        process_noise_pos=1e-5,
        process_noise_vel=1e-4,
        measurement_noise=1e-4,
        dt=300.0
    )
    kf = TrajectoryKalmanFilter(params)
    
    # Evaluate on test set
    print("\n=== Kalman Filter Evaluation ===")
    X_test, Y_test = create_windows(test_trajs, args.window, args.horizon, max_windows=args.max_windows)
    
    if len(X_test) == 0:
        print("No test windows created. Exiting.")
        return
    
    kalman_results = evaluate_kalman(kf, X_test, Y_test)
    
    # Compare with existing models
    metrics_dir = Path("metrics")
    if metrics_dir.exists():
        compare_with_existing_results(kalman_results, metrics_dir)
    
    # Scenario analysis
    if args.analyze_scenarios:
        scenario_results = analyze_error_by_scenario(kf, test_trajs, args.window, args.horizon)
        
        # Save scenario results
        results_dict = {
            "overall": kalman_results,
            "scenarios": scenario_results
        }
        
        scenario_file = metrics_dir / "kalman_filter_scenarios.json"
        with open(scenario_file, "w") as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nScenario analysis saved to {scenario_file}")


if __name__ == "__main__":
    main()
