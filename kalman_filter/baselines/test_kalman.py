"""
Quick test script to verify Kalman Filter implementation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from kalman_filter.kalman_filter import KalmanFilter, TrajectoryKalmanFilter, KalmanFilterParams


def test_basic_kalman():
    """Test basic Kalman Filter functionality."""
    print("=== Testing Basic Kalman Filter ===")
    
    # Create filter
    kf = KalmanFilter()
    
    # Simulate a simple trajectory (straight line with constant velocity)
    dt = 300.0  # 5 minutes
    n_steps = 20
    
    # True trajectory: moving northeast at constant velocity
    lat_start, lon_start = 0.5, 0.5
    v_lat, v_lon = 0.001, 0.001  # normalized velocity per 5 min
    
    true_positions = []
    for i in range(n_steps):
        lat = lat_start + v_lat * i
        lon = lon_start + v_lon * i
        true_positions.append([lat, lon])
    
    true_positions = np.array(true_positions, dtype=np.float32)
    
    # Add measurement noise
    noise_std = 0.0001
    measurements = true_positions + np.random.randn(n_steps, 2) * noise_std
    
    # Filter the trajectory
    predictions = []
    for i, z in enumerate(measurements):
        if i == 0:
            kf.initialize(z)
        else:
            kf.update(z)
        
        # Predict next position
        x_pred, _ = kf.predict()
        predictions.append([x_pred[0], x_pred[1]])
    
    predictions = np.array(predictions[:-1])  # Exclude last (no ground truth)
    ground_truth = true_positions[1:]  # Shifted by 1
    
    # Compute error
    errors = np.linalg.norm(predictions - ground_truth, axis=-1)
    mae = errors.mean()
    
    print(f"  Mean prediction error: {mae:.6f} (normalized units)")
    print(f"  Expected (should be close to noise level {noise_std:.6f})")
    
    if mae < 0.001:
        print("  ✓ Test PASSED")
        return True
    else:
        print("  ✗ Test FAILED")
        return False


def test_trajectory_wrapper():
    """Test TrajectoryKalmanFilter wrapper."""
    print("\n=== Testing Trajectory Wrapper ===")
    
    # Create synthetic trajectory
    n_points = 100
    trajectory = np.zeros((n_points, 9), dtype=np.float32)
    
    # Simulate movement
    lat, lon = 0.5, 0.5
    v_lat, v_lon = 0.001, 0.0005
    
    for i in range(n_points):
        trajectory[i, 0] = lat + v_lat * i + np.random.randn() * 0.0001  # LAT
        trajectory[i, 1] = lon + v_lon * i + np.random.randn() * 0.0001  # LON
        trajectory[i, 2] = 0.5  # SOG (dummy)
        trajectory[i, 3] = 45.0  # COG (dummy)
        trajectory[i, 7] = i * 300.0  # TIMESTAMP
        trajectory[i, 8] = 123456789  # MMSI
    
    # Test single prediction
    kf = TrajectoryKalmanFilter()
    window = trajectory[:64]
    horizon = 12
    
    predictions = kf.predict(window, horizon)
    
    print(f"  Window shape: {window.shape}")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Expected: ({horizon}, 2)")
    
    if predictions.shape == (horizon, 2):
        print("  ✓ Shape test PASSED")
        
        # Check predictions are reasonable (not NaN, not out of bounds)
        if not np.any(np.isnan(predictions)) and np.all(predictions >= 0) and np.all(predictions <= 1):
            print("  ✓ Validity test PASSED")
            return True
        else:
            print("  ✗ Validity test FAILED (NaN or out of bounds)")
            return False
    else:
        print("  ✗ Shape test FAILED")
        return False


def test_batch_prediction():
    """Test batch prediction."""
    print("\n=== Testing Batch Prediction ===")
    
    # Create batch of synthetic windows
    batch_size = 10
    window_size = 64
    horizon = 12
    
    windows = np.random.rand(batch_size, window_size, 9).astype(np.float32)
    
    # Normalize to [0, 1] for lat/lon
    windows[:, :, 0] = 0.5 + windows[:, :, 0] * 0.1  # LAT around 0.5
    windows[:, :, 1] = 0.5 + windows[:, :, 1] * 0.1  # LON around 0.5
    
    kf = TrajectoryKalmanFilter()
    predictions = kf.predict_batch(windows, horizon)
    
    print(f"  Batch windows shape: {windows.shape}")
    print(f"  Batch predictions shape: {predictions.shape}")
    print(f"  Expected: ({batch_size}, {horizon}, 2)")
    
    if predictions.shape == (batch_size, horizon, 2):
        print("  ✓ Batch test PASSED")
        return True
    else:
        print("  ✗ Batch test FAILED")
        return False


def test_parameter_configuration():
    """Test custom parameter configuration."""
    print("\n=== Testing Parameter Configuration ===")
    
    params = KalmanFilterParams(
        process_noise_pos=1e-6,
        process_noise_vel=1e-5,
        measurement_noise=1e-5,
        dt=300.0
    )
    
    kf = TrajectoryKalmanFilter(params)
    
    # Verify parameters are set
    if (kf.params.process_noise_pos == 1e-6 and
        kf.params.process_noise_vel == 1e-5 and
        kf.params.measurement_noise == 1e-5):
        print("  ✓ Parameters test PASSED")
        return True
    else:
        print("  ✗ Parameters test FAILED")
        return False


if __name__ == "__main__":
    print("Running Kalman Filter Unit Tests\n")
    print("=" * 60)
    
    tests = [
        test_basic_kalman,
        test_trajectory_wrapper,
        test_batch_prediction,
        test_parameter_configuration
    ]
    
    results = [test() for test in tests]
    
    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("✓ All tests PASSED!")
        exit(0)
    else:
        print("✗ Some tests FAILED")
        exit(1)
