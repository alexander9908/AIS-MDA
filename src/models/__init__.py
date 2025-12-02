from .kinematic import constant_velocity_predict
from .legacy.rnn_seq2seq import GRUSeq2Seq
from .legacy.tptrans import TPTrans

try:
    from .kalman_filter.kalman_filter import (
        KalmanFilter,
        TrajectoryKalmanFilter,
        KalmanFilterParams,
    )
except ImportError:  # Optional legacy module not always present
    KalmanFilter = None
    TrajectoryKalmanFilter = None
    KalmanFilterParams = None

__all__ = [
    "constant_velocity_predict",
    "GRUSeq2Seq",
    "TPTrans",
    "KalmanFilter",
    "TrajectoryKalmanFilter",
    "KalmanFilterParams",
]
