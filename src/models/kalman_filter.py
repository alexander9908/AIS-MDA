"""
Kalman Filter for AIS Trajectory Prediction

Implements a constant velocity model with Kalman filtering for vessel trajectory prediction.
Works with normalized AIS data from the MapReduce pipeline.

State vector: [lat, lon, lat_vel, lon_vel]
Measurement vector: [lat, lon]
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class KalmanFilterParams:
    """Parameters for Kalman Filter."""
    # Process noise (uncertainty in the motion model)
    process_noise_pos: float = 1e-5  # Position process noise
    process_noise_vel: float = 1e-4  # Velocity process noise
    
    # Measurement noise (uncertainty in GPS measurements)
    measurement_noise: float = 1e-4  # Position measurement noise
    
    # Initial state uncertainty
    initial_pos_uncertainty: float = 1e-3
    initial_vel_uncertainty: float = 1e-2
    
    # Time step (seconds) - default 300s = 5 minutes for MapReduce data
    dt: float = 300.0


class KalmanFilter:
    """
    Kalman Filter for vessel trajectory prediction.
    
    State space model:
    - State: x = [lat, lon, v_lat, v_lon]^T
    - Measurement: z = [lat, lon]^T
    - Dynamics: constant velocity model
    
    The filter maintains:
    - x: state estimate (4D)
    - P: state covariance (4x4)
    - F: state transition matrix (4x4)
    - H: measurement matrix (2x4)
    - Q: process noise covariance (4x4)
    - R: measurement noise covariance (2x2)
    """
    
    def __init__(self, params: Optional[KalmanFilterParams] = None):
        self.params = params or KalmanFilterParams()
        
        # State dimension
        self.n_state = 4  # [lat, lon, v_lat, v_lon]
        self.n_meas = 2   # [lat, lon]
        
        # Initialize matrices
        self._initialize_matrices()
        
        # State and covariance
        self.x = None  # Will be initialized on first measurement
        self.P = None
        
    def _initialize_matrices(self):
        """Initialize system matrices for constant velocity model."""
        dt = self.params.dt
        
        # State transition matrix (constant velocity)
        # x_{k+1} = F @ x_k
        # [lat_{k+1}  ]   [1  0  dt  0 ] [lat_k  ]
        # [lon_{k+1}  ] = [0  1  0  dt] [lon_k  ]
        # [v_lat_{k+1}]   [0  0  1   0 ] [v_lat_k]
        # [v_lon_{k+1}]   [0  0  0   1 ] [v_lon_k]
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix (we observe position only)
        # z_k = H @ x_k
        # [lat_obs] = [1 0 0 0] [lat  ]
        # [lon_obs]   [0 1 0 0] [lon  ]
        #                       [v_lat]
        #                       [v_lon]
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance (uncertainty in motion model)
        # Models acceleration noise
        q_pos = self.params.process_noise_pos
        q_vel = self.params.process_noise_vel
        
        # Continuous white noise acceleration model
        dt2 = dt * dt
        dt3 = dt2 * dt / 2.0
        dt4 = dt2 * dt2 / 4.0
        
        self.Q = np.array([
            [dt4 * q_pos, 0, dt3 * q_pos, 0],
            [0, dt4 * q_pos, 0, dt3 * q_pos],
            [dt3 * q_pos, 0, dt2 * q_vel, 0],
            [0, dt3 * q_pos, 0, dt2 * q_vel]
        ], dtype=np.float32)
        
        # Measurement noise covariance (GPS/AIS measurement uncertainty)
        r = self.params.measurement_noise
        self.R = np.array([
            [r, 0],
            [0, r]
        ], dtype=np.float32)
        
        # Initial state covariance
        p_pos = self.params.initial_pos_uncertainty
        p_vel = self.params.initial_vel_uncertainty
        self.P_init = np.array([
            [p_pos, 0, 0, 0],
            [0, p_pos, 0, 0],
            [0, 0, p_vel, 0],
            [0, 0, 0, p_vel]
        ], dtype=np.float32)
    
    def initialize(self, z: np.ndarray):
        """
        Initialize filter with first measurement.
        
        Args:
            z: Initial measurement [lat, lon] (normalized)
        """
        # Initialize state with zero velocity
        self.x = np.array([z[0], z[1], 0.0, 0.0], dtype=np.float32)
        self.P = self.P_init.copy()
    
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step (Time Update).
        
        Predicts next state and covariance using motion model:
        x̂_{k|k-1} = F @ x_{k-1|k-1}
        P_{k|k-1} = F @ P_{k-1|k-1} @ F^T + Q
        
        Returns:
            x_pred: Predicted state [lat, lon, v_lat, v_lon]
            P_pred: Predicted covariance (4x4)
        """
        if self.x is None:
            raise RuntimeError("Filter not initialized. Call initialize() first.")
        
        # Predict state
        x_pred = self.F @ self.x
        
        # Predict covariance
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        return x_pred, P_pred
    
    def update(self, z: np.ndarray):
        """
        Update step (Measurement Update).
        
        Updates state estimate with new measurement using Kalman gain:
        K = P_{k|k-1} @ H^T @ (H @ P_{k|k-1} @ H^T + R)^{-1}
        x_{k|k} = x_{k|k-1} + K @ (z_k - H @ x_{k|k-1})
        P_{k|k} = (I - K @ H) @ P_{k|k-1}
        
        Args:
            z: Measurement [lat, lon] (normalized)
        """
        if self.x is None:
            # First measurement - initialize
            self.initialize(z)
            return
        
        # Predict
        x_pred, P_pred = self.predict()
        
        # Innovation (measurement residual)
        y = z - self.H @ x_pred
        
        # Innovation covariance
        S = self.H @ P_pred @ self.H.T + self.R
        
        # Kalman gain
        # K determines how much to trust measurement vs prediction
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = x_pred + K @ y
        
        # Update covariance (Joseph form for numerical stability)
        I = np.eye(self.n_state, dtype=np.float32)
        IKH = I - K @ self.H
        self.P = IKH @ P_pred @ IKH.T + K @ self.R @ K.T
    
    def forecast(self, n_steps: int) -> np.ndarray:
        """
        Forecast future positions for n_steps.
        
        Args:
            n_steps: Number of future steps to predict
            
        Returns:
            predictions: Array of shape (n_steps, 2) with [lat, lon] predictions
        """
        if self.x is None:
            raise RuntimeError("Filter not initialized.")
        
        predictions = np.zeros((n_steps, 2), dtype=np.float32)
        
        # Use current state as starting point
        x_current = self.x.copy()
        
        # Iteratively predict without measurement updates
        for i in range(n_steps):
            x_current = self.F @ x_current
            predictions[i, 0] = x_current[0]  # lat
            predictions[i, 1] = x_current[1]  # lon
        
        return predictions
    
    def reset(self):
        """Reset filter state."""
        self.x = None
        self.P = None


class TrajectoryKalmanFilter:
    """
    Wrapper for processing complete trajectories with Kalman Filter.
    
    Handles:
    - Training on historical trajectory to learn system parameters
    - Prediction of future trajectory steps
    """
    
    def __init__(self, params: Optional[KalmanFilterParams] = None):
        self.params = params or KalmanFilterParams()
        self.kf = KalmanFilter(self.params)
    
    def fit(self, trajectory: np.ndarray):
        """
        Fit filter on historical trajectory.
        
        Args:
            trajectory: Array of shape (T, 2) or (T, 9) with positions
                       If shape[1] == 9, expects normalized data with [LAT, LON, ...]
                       If shape[1] == 2, expects [lat, lon]
        """
        # Extract lat/lon if full trajectory
        if trajectory.shape[1] == 9:
            # Columns: LAT=0, LON=1, SOG=2, COG=3, HEADING=4, ROT=5, NAV_STT=6, TIMESTAMP=7, MMSI=8
            positions = trajectory[:, [0, 1]]  # [LAT, LON]
        elif trajectory.shape[1] == 2:
            positions = trajectory
        else:
            raise ValueError(f"Expected trajectory with 2 or 9 columns, got {trajectory.shape[1]}")
        
        # Reset filter
        self.kf.reset()
        
        # Process each measurement to update filter state
        for pos in positions:
            self.kf.update(pos)
    
    def predict(self, window: np.ndarray, horizon: int) -> np.ndarray:
        """
        Predict future trajectory given a historical window.
        
        Args:
            window: Historical window of shape (T, 2) or (T, 9)
            horizon: Number of future steps to predict
            
        Returns:
            predictions: Array of shape (horizon, 2) with [lat, lon] predictions
        """
        # Fit on the window
        self.fit(window)
        
        # Forecast future positions
        predictions = self.kf.forecast(horizon)
        
        return predictions
    
    def predict_batch(self, windows: np.ndarray, horizon: int) -> np.ndarray:
        """
        Predict for a batch of windows.
        
        Args:
            windows: Array of shape (B, T, F) where B=batch, T=time, F=features
            horizon: Number of future steps to predict
            
        Returns:
            predictions: Array of shape (B, horizon, 2)
        """
        batch_size = windows.shape[0]
        predictions = np.zeros((batch_size, horizon, 2), dtype=np.float32)
        
        for i in range(batch_size):
            predictions[i] = self.predict(windows[i], horizon)
        
        return predictions


def tune_kalman_filter(trajectories: list[np.ndarray], 
                       val_trajectories: list[np.ndarray],
                       window_size: int = 64,
                       horizon: int = 12) -> KalmanFilterParams:
    """
    Simple grid search to tune Kalman Filter parameters.
    
    Args:
        trajectories: List of training trajectories
        val_trajectories: List of validation trajectories
        window_size: Size of input window
        horizon: Prediction horizon
        
    Returns:
        Best parameters found
    """
    from ..eval.metrics_traj import ade
    
    # Parameter grid
    process_noise_pos_values = [1e-6, 1e-5, 1e-4]
    process_noise_vel_values = [1e-5, 1e-4, 1e-3]
    measurement_noise_values = [1e-5, 1e-4, 1e-3]
    
    best_ade = float('inf')
    best_params = None
    
    print("Tuning Kalman Filter parameters...")
    
    for q_pos in process_noise_pos_values:
        for q_vel in process_noise_vel_values:
            for r in measurement_noise_values:
                params = KalmanFilterParams(
                    process_noise_pos=q_pos,
                    process_noise_vel=q_vel,
                    measurement_noise=r
                )
                
                # Evaluate on validation set
                kf = TrajectoryKalmanFilter(params)
                val_ade = 0.0
                n_samples = 0
                
                for traj in val_trajectories[:100]:  # Limit for speed
                    if len(traj) < window_size + horizon:
                        continue
                    
                    # Create windows
                    for i in range(len(traj) - window_size - horizon):
                        window = traj[i:i + window_size]
                        target = traj[i + window_size:i + window_size + horizon, :2]
                        
                        pred = kf.predict(window, horizon)
                        val_ade += np.mean(np.linalg.norm(pred - target, axis=-1))
                        n_samples += 1
                
                if n_samples > 0:
                    val_ade /= n_samples
                    
                    if val_ade < best_ade:
                        best_ade = val_ade
                        best_params = params
                        print(f"New best: ADE={val_ade:.6f}, q_pos={q_pos}, q_vel={q_vel}, r={r}")
    
    print(f"\nBest parameters: ADE={best_ade:.6f}")
    print(f"  process_noise_pos: {best_params.process_noise_pos}")
    print(f"  process_noise_vel: {best_params.process_noise_vel}")
    print(f"  measurement_noise: {best_params.measurement_noise}")
    
    return best_params
