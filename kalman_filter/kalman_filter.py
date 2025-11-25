"""
Kalman Filter for AIS Trajectory Prediction (Straight Line Baseline)

Implements a Constant Velocity (CV) model.
- Prediction Phase: Projects state linearly based on inertia.
- Forecast Phase: Strictly applies the transition matrix F recursively.

State vector: [lat, lon, v_lat, v_lon]
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class KalmanFilterParams:
    """Parameters for Kalman Filter."""
    # High uncertainty in velocity process noise allows the filter 
    # to adapt quickly to turns during the *observation* phase,
    # providing the correct tangent vector for the *straight line* forecast.
    process_noise_pos: float = 1e-5
    process_noise_vel: float = 1e-3  
    measurement_noise: float = 1e-4
    initial_pos_uncertainty: float = 1e-3
    initial_vel_uncertainty: float = 1e-2
    dt: float = 300.0  # 5 minutes (standardize this to your data)

class KalmanFilter:
    def __init__(self, params: Optional[KalmanFilterParams] = None):
        self.params = params or KalmanFilterParams()
        self.n_state = 4
        self.n_meas = 2
        self._initialize_matrices()
        self.x = None 
        self.P = None
        
    def _initialize_matrices(self):
        dt = self.params.dt
        
        # F: State Transition Matrix
        # This defines the "Straight Line" physics: p_new = p_old + v * dt
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # H: Measurement Matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Q: Process Noise Covariance (Discrete White Noise Acceleration)
        q_pos = self.params.process_noise_pos
        q_vel = self.params.process_noise_vel
        dt2 = dt**2
        dt3 = dt**3 / 2
        dt4 = dt**4 / 4
        
        self.Q = np.array([
            [dt4*q_pos, 0,       dt3*q_pos, 0],
            [0,         dt4*q_pos, 0,       dt3*q_pos],
            [dt3*q_pos, 0,       dt2*q_vel, 0],
            [0,         dt3*q_pos, 0,       dt2*q_vel]
        ], dtype=np.float32)
        
        # R: Measurement Noise
        r = self.params.measurement_noise
        self.R = np.array([[r, 0], [0, r]], dtype=np.float32)
        
        # P_init: Initial Covariance
        self.P_init = np.eye(4, dtype=np.float32)
        self.P_init[0:2, 0:2] *= self.params.initial_pos_uncertainty
        self.P_init[2:4, 2:4] *= self.params.initial_vel_uncertainty
    
    def initialize(self, z: np.ndarray):
        """Initialize with first position measurement, zero velocity assumptions."""
        self.x = np.array([z[0], z[1], 0.0, 0.0], dtype=np.float32)
        self.P = self.P_init.copy()
    
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """Time Update (A priori)"""
        if self.x is None: raise RuntimeError("Filter not initialized.")
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        return x_pred, P_pred
    
    def update(self, z: np.ndarray):
        """Measurement Update (A posteriori)"""
        if self.x is None:
            self.initialize(z)
            return
        
        x_pred, P_pred = self.predict()
        y = z - self.H @ x_pred # Innovation
        S = self.H @ P_pred @ self.H.T + self.R # Innovation Covariance
        K = P_pred @ self.H.T @ np.linalg.inv(S) # Kalman Gain
        
        self.x = x_pred + K @ y
        I = np.eye(self.n_state, dtype=np.float32)
        self.P = (I - K @ self.H) @ P_pred
    
    def forecast(self, n_steps: int) -> np.ndarray:
        """
        Pure Kinematic Projection.
        This projects the state forward using F, ignoring any future measurements.
        This results in a straight line trajectory based on the last estimated velocity vector.
        """
        if self.x is None: raise RuntimeError("Filter not initialized.")
        
        predictions = np.zeros((n_steps, 2), dtype=np.float32)
        x_curr = self.x.copy()
        
        for i in range(n_steps):
            # x_{k+1} = F * x_k
            x_curr = self.F @ x_curr
            predictions[i] = x_curr[0:2] # Store lat, lon
            
        return predictions

    def reset(self):
        self.x = None
        self.P = None

class TrajectoryKalmanFilter:
    def __init__(self, params: Optional[KalmanFilterParams] = None):
        self.params = params or KalmanFilterParams()
        self.kf = KalmanFilter(self.params)
    
    def fit(self, trajectory: np.ndarray):
        """Fit on historical observation window."""
        # Expecting trajectory shape (T, 2) [lat, lon] or (T, 9) normalized
        if trajectory.shape[1] >= 2:
            pos = trajectory[:, [0, 1]]
        else:
            raise ValueError("Invalid trajectory shape")
            
        self.kf.reset()
        for p in pos:
            self.kf.update(p)
            
    def predict(self, window: np.ndarray, horizon: int) -> np.ndarray:
        self.fit(window)
        return self.kf.forecast(horizon)