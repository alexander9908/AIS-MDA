# Kalman Filter for AIS Trajectory Prediction

## Overview

The Kalman filter is a recursive Bayesian estimator that combines noisy measurements with a motion model to optimally estimate the state of a system. For AIS trajectory prediction, we use it to track vessel positions and velocities, then forecast future positions.

## Our Scenario

We track vessels using their AIS GPS positions (lat/lon) sampled every 5 minutes. The filter estimates both **position** and **velocity**, allowing us to predict where the vessel will be in the next hour.

### State Vector (What We Track)
```
x = [lat, lon, v_lat, v_lon]ᵀ
```
- **lat, lon**: Current vessel position (normalized to [0,1] for Denmark region)
- **v_lat, v_lon**: Velocity components (degrees per 5 minutes)

### Measurement Vector (What We Observe)
```
z = [lat, lon]ᵀ
```
We only measure **position** from GPS/AIS, not velocity. The filter infers velocity from position changes.

## Motion Model: Constant Velocity

We assume vessels move at approximately constant velocity between time steps. This is encoded in the **state transition matrix**:

```
F = [[1,  0,  Δt, 0 ],
     [0,  1,  0,  Δt],
     [0,  0,  1,  0 ],
     [0,  0,  0,  1 ]]
```

Where Δt = 300 seconds (5 minutes).

This gives the update equations:
```
lat_{k+1}   = lat_k   + Δt × v_lat_k
lon_{k+1}   = lon_k   + Δt × v_lon_k
v_lat_{k+1} = v_lat_k  (constant)
v_lon_{k+1} = v_lon_k  (constant)
```

## Observation Model

The **measurement matrix** H extracts the position from our state:

```
H = [[1, 0, 0, 0],
     [0, 1, 0, 0]]
```

This means: `z = H × x` gives us just the lat/lon (ignoring velocity).

## Uncertainty Modeling

### Process Noise (Q)
Represents uncertainty in our motion model. Vessels don't actually move at perfectly constant velocity - they accelerate, decelerate, and turn.

```
Q = continuous white noise acceleration model
```

**Parameters:**
- `process_noise_pos = 1e-5`: Position uncertainty
- `process_noise_vel = 1e-4`: Velocity uncertainty

This models random accelerations as Gaussian noise.

### Measurement Noise (R)
Represents GPS/AIS measurement uncertainty. Real-world positions have errors.

```
R = [[σ², 0 ],
     [0,  σ²]]
```

**Parameters:**
- `measurement_noise = 1e-4`: Position measurement standard deviation

## The Algorithm: Two-Phase Cycle

For each new AIS message, the filter runs two steps:

### 1. **Prediction Phase** (Time Update)
Use the motion model to predict where the vessel *should* be:

```python
x̂_{k|k-1} = F @ x_{k-1|k-1}
P_{k|k-1} = F @ P_{k-1|k-1} @ Fᵀ + Q
```

- **x̂**: Predicted state (position + velocity)
- **P**: Predicted uncertainty (covariance matrix)

### 2. **Update Phase** (Measurement Update)
When we get a GPS measurement, correct our prediction:

```python
# Innovation: How wrong was our prediction?
y = z_k - H @ x̂_{k|k-1}

# Kalman Gain: How much to trust measurement vs prediction?
K = P_{k|k-1} @ Hᵀ @ (H @ P_{k|k-1} @ Hᵀ + R)⁻¹

# Corrected state estimate
x_{k|k} = x̂_{k|k-1} + K @ y

# Updated uncertainty
P_{k|k} = (I - K @ H) @ P_{k|k-1}
```

The **Kalman gain** K automatically balances:
- **High measurement noise** → Trust the model more
- **High process noise** → Trust the measurements more

## Forecasting Future Positions

After filtering the historical window (64 timesteps = 5h 20min), we forecast the future by:

1. Taking the current best state estimate: `x = [lat, lon, v_lat, v_lon]`
2. Repeatedly applying the motion model **without** measurement updates:
   ```python
   for i in range(horizon):
       x = F @ x
       predictions[i] = [x[0], x[1]]  # Extract lat/lon
   ```

This gives us 12 future positions (1 hour ahead) assuming constant velocity.

## Why It Works

The Kalman filter is **optimal** (minimum mean squared error) when:
1. ✅ Dynamics are **linear** (constant velocity model)
2. ✅ Noise is **Gaussian** (GPS errors, random accelerations)
3. ⚠️ Model matches reality (vessels *approximately* move at constant velocity over short intervals)

For maritime traffic:
- **Strengths**: Works well for vessels on straight courses at steady speed
- **Limitations**: Struggles with sharp turns, acceleration, or complex maneuvers
- **Baseline value**: Simple, interpretable, no training required - perfect for comparing against neural networks

## Our Implementation Results

On 13,804 test windows from real AIS data:
- **Average Displacement Error (ADE)**: 380 meters over 1-hour prediction
- **Final Displacement Error (FDE)**: 845 meters at 60 minutes

Error grows linearly with time as expected for constant velocity extrapolation.
