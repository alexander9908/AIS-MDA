# Kalman Filter Baseline for AIS Trajectory Prediction

This directory contains a comprehensive Kalman Filter implementation for vessel trajectory prediction, serving as a classical baseline to compare against neural network models (TPTrans, GRU).

## 📚 Background

The **Kalman Filter** is an optimal recursive state estimator for linear dynamical systems with Gaussian noise. It's widely used in navigation, tracking, and control systems. For AIS trajectory prediction, it provides:

- **Physically-motivated predictions** based on laws of motion
- **Uncertainty quantification** through covariance estimation
- **Real-time performance** with minimal computational cost
- **Interpretable parameters** related to vessel dynamics

## 🧮 Mathematical Foundation

### State Space Model

**State Vector** (4D):
```
x = [lat, lon, v_lat, v_lon]ᵀ
```
- `lat`, `lon`: Vessel position (normalized [0,1])
- `v_lat`, `v_lon`: Velocity components (normalized units per 5 min)

**Measurement Vector** (2D):
```
z = [lat, lon]ᵀ
```
Only position is observed; velocity is inferred by the filter.

### System Dynamics (Constant Velocity Model)

**State Transition**:
```
x_{k+1} = F @ x_k + w_k
```

where `F` is the state transition matrix (with time step `dt = 300s`):
```
F = [ 1  0  dt  0 ]
    [ 0  1  0  dt ]
    [ 0  0  1   0 ]
    [ 0  0  0   1 ]
```

**Measurement Model**:
```
z_k = H @ x_k + v_k
```

where `H` is the measurement matrix:
```
H = [ 1  0  0  0 ]
    [ 0  1  0  0 ]
```

### Noise Models

**Process Noise** (`Q`): Represents uncertainty in the motion model (e.g., random accelerations)
```
Q = [ q_pos*dt⁴/4   0           q_pos*dt³/2   0         ]
    [ 0             q_pos*dt⁴/4   0           q_pos*dt³/2 ]
    [ q_pos*dt³/2   0           q_vel*dt²     0         ]
    [ 0             q_pos*dt³/2   0           q_vel*dt²   ]
```

**Measurement Noise** (`R`): Represents GPS/AIS sensor uncertainty
```
R = [ r  0 ]
    [ 0  r ]
```

### Kalman Filter Algorithm

#### 1. Prediction Phase (Time Update)

Predict next state:
```
x̂_{k|k-1} = F @ x_{k-1|k-1}
```

Predict covariance:
```
P_{k|k-1} = F @ P_{k-1|k-1} @ Fᵀ + Q
```

#### 2. Update Phase (Measurement Update)

Compute innovation (measurement residual):
```
y_k = z_k - H @ x̂_{k|k-1}
```

Innovation covariance:
```
S_k = H @ P_{k|k-1} @ Hᵀ + R
```

**Kalman Gain** (determines trust between prediction vs measurement):
```
K_k = P_{k|k-1} @ Hᵀ @ S_k⁻¹
```

Update state estimate:
```
x̂_{k|k} = x̂_{k|k-1} + K_k @ y_k
```

Update covariance:
```
P_{k|k} = (I - K_k @ H) @ P_{k|k-1}
```

## 🏗️ Implementation

### Classes

#### `KalmanFilterParams`
Configuration dataclass for filter parameters:
- `process_noise_pos`: Position process noise (default: 1e-5)
- `process_noise_vel`: Velocity process noise (default: 1e-4)
- `measurement_noise`: GPS measurement noise (default: 1e-4)
- `dt`: Time step in seconds (default: 300.0 for 5-minute sampling)

#### `KalmanFilter`
Core filter implementation:
- `initialize(z)`: Initialize with first measurement
- `predict()`: Perform prediction step
- `update(z)`: Perform update step with new measurement
- `forecast(n_steps)`: Multi-step ahead prediction

#### `TrajectoryKalmanFilter`
High-level wrapper for trajectory processing:
- `fit(trajectory)`: Train filter on historical trajectory
- `predict(window, horizon)`: Predict future positions
- `predict_batch(windows, horizon)`: Batch prediction

### Key Features

1. **Numerical Stability**: Uses Joseph form for covariance update
2. **Continuous White Noise Acceleration Model**: Proper discretization of process noise
3. **Batch Processing**: Efficient prediction on multiple windows
4. **Parameter Tuning**: Grid search for optimal noise parameters

## 🚀 Usage

### Basic Usage

```python
from src.models.kalman_filter import TrajectoryKalmanFilter, KalmanFilterParams

# Create filter with default parameters
kf = TrajectoryKalmanFilter()

# Or with custom parameters
params = KalmanFilterParams(
    process_noise_pos=1e-5,
    process_noise_vel=1e-4,
    measurement_noise=1e-4,
    dt=300.0
)
kf = TrajectoryKalmanFilter(params)

# Predict on a single window
window = trajectory[:64]  # Historical positions
predictions = kf.predict(window, horizon=12)  # Predict 12 steps ahead

# Batch prediction
predictions = kf.predict_batch(windows, horizon=12)
```

### Training and Evaluation

```bash
# Evaluate with default parameters
python -m src.baselines.train_kalman \
    --final_dir data/map_reduce_final \
    --window 64 \
    --horizon 12 \
    --max_files 500

# With hyperparameter tuning
python -m src.baselines.train_kalman \
    --final_dir data/map_reduce_final \
    --window 64 \
    --horizon 12 \
    --tune

# Using the convenience script
bash scripts/eval_kalman.sh data/map_reduce_final 64 12

# With tuning
bash scripts/eval_kalman.sh data/map_reduce_final 64 12 --tune
```

### Command-Line Arguments

- `--final_dir`: Directory with `*_processed.pkl` files (default: `data/map_reduce_final`)
- `--window`: Input window size (default: 64)
- `--horizon`: Prediction horizon (default: 12)
- `--max_files`: Max trajectory files to load (default: None = all)
- `--max_windows`: Max evaluation windows (default: 10000)
- `--tune`: Enable hyperparameter tuning
- `--process_noise_pos`: Position process noise (default: 1e-5)
- `--process_noise_vel`: Velocity process noise (default: 1e-4)
- `--measurement_noise`: Measurement noise (default: 1e-4)

## 📊 Expected Performance

### Typical Results

For AIS data with 5-minute sampling:

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| **ADE** | 0.005 - 0.020 | In normalized [0,1] coordinates |
| **FDE** | 0.010 - 0.030 | Higher at long horizons |
| **Real-world ADE** | 500m - 2km | Depends on vessel behavior |

### Performance Characteristics

**Strengths**:
- Excellent for straight-line motion (constant velocity assumption)
- Fast inference (no GPU needed)
- Works well with sparse data
- Provides uncertainty estimates

**Limitations**:
- Poor performance during maneuvers (turns, accelerations)
- Linear model cannot capture complex patterns
- Assumes Gaussian noise distributions
- No learning from historical patterns

## 🔧 Parameter Tuning

### Grid Search

The `tune_kalman_filter()` function performs grid search over:
- `process_noise_pos`: [1e-6, 1e-5, 1e-4]
- `process_noise_vel`: [1e-5, 1e-4, 1e-3]
- `measurement_noise`: [1e-5, 1e-4, 1e-3]

### Interpretation

**High process noise** → Filter trusts measurements more (follows track closely)
**Low process noise** → Filter trusts motion model more (smoother predictions)

**High measurement noise** → Filter discounts noisy measurements
**Low measurement noise** → Filter trusts GPS measurements

**Rule of thumb**:
- For erratic vessel behavior: Increase process noise
- For poor GPS quality: Increase measurement noise
- For smooth cruising: Decrease process noise

## 📈 Comparison with Neural Networks

### Advantages over NNs

1. **Interpretability**: All parameters have physical meaning
2. **Data efficiency**: Works with minimal training data
3. **Uncertainty**: Built-in covariance estimation
4. **Speed**: 100-1000x faster inference
5. **Robustness**: No overfitting or convergence issues

### When NNs Should Win

1. **Complex maneuvers**: Turning, acceleration patterns
2. **Multi-vessel interactions**: Traffic-aware predictions
3. **Long horizons**: Learning long-term dependencies
4. **Non-linear dynamics**: Port approach, pilotage

### Expected Performance Gap

| Scenario | KF Performance | NN Advantage |
|----------|---------------|--------------|
| Straight cruise | Excellent | 0-10% |
| Gradual turns | Good | 10-30% |
| Sharp maneuvers | Poor | 50-200% |
| Port approach | Poor | 100-500% |

## 📁 Output Files

### Metrics JSON (`metrics/kalman_filter.json`)

```json
{
  "model": "kalman_filter",
  "window": 64,
  "horizon": 12,
  "parameters": {
    "process_noise_pos": 1e-5,
    "process_noise_vel": 1e-4,
    "measurement_noise": 1e-4,
    "dt": 300.0
  },
  "validation": {
    "ade": 0.0123,
    "fde": 0.0234,
    "per_horizon_ade": [0.001, 0.002, ..., 0.023],
    "n_samples": 5000
  },
  "test": {
    "ade": 0.0125,
    "fde": 0.0236,
    "per_horizon_ade": [...],
    "n_samples": 2500
  }
}
```

### Summary Text (`data/checkpoints/kalman_filter_summary.txt`)

Human-readable summary with configuration and results.

## 🔬 Advanced Usage

### Custom State-Space Model

You can extend the `KalmanFilter` class to implement:
- **Coordinated turn model** (constant turn rate)
- **Singer model** (adaptive acceleration)
- **Interacting Multiple Model (IMM)** filter

Example:
```python
class CoordinatedTurnKalmanFilter(KalmanFilter):
    def _initialize_matrices(self):
        # Implement turn rate dynamics
        # State: [lat, lon, v_lat, v_lon, omega]
        # where omega is turn rate
        ...
```

### Integration with Neural Networks

Use Kalman Filter as:
1. **Preprocessing**: Smooth noisy trajectories before NN training
2. **Post-processing**: Refine NN predictions with physics constraints
3. **Ensemble**: Combine KF + NN predictions (weighted average)

## 📚 References

1. **Kalman, R. E.** (1960). "A New Approach to Linear Filtering and Prediction Problems"
2. **Bar-Shalom, Y., et al.** (2001). "Estimation with Applications to Tracking and Navigation"
3. **Ristic, B., et al.** (2004). "Beyond the Kalman Filter: Particle Filters for Tracking Applications"
4. **Perera, L. P., et al.** (2015). "Maritime Traffic Monitoring Based on Vessel Detection, Tracking, State Estimation, and Trajectory Prediction"

## 🐛 Troubleshooting

### Issue: Poor Performance on All Trajectories

**Solution**: Check data normalization. Kalman Filter expects [0,1] normalized positions.

### Issue: Diverging Predictions

**Solution**: Process noise too high. Reduce `process_noise_pos` and `process_noise_vel`.

### Issue: Not Following Measurements

**Solution**: Measurement noise too high. Reduce `measurement_noise`.

### Issue: Too Smooth (Not Tracking Turns)

**Solution**: Process noise too low. Increase `process_noise_vel` to allow velocity changes.

---

**Note**: This implementation is designed for the MapReduce preprocessed data with 5-minute sampling. For different sampling rates, adjust the `dt` parameter accordingly.
