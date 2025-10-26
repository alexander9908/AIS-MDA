from __future__ import annotations
import pandas as pd
import numpy as np

def label_eta_targets(df: pd.DataFrame, port_id_col: str = "dest_port_id", eta_seconds_col: str = "time_to_port_sec") -> pd.DataFrame:
    """Placeholder: assumes port membership already computed.
    Adds label columns for next port classification and ETA regression.
    """
    if port_id_col not in df.columns or eta_seconds_col not in df.columns:
        df = df.copy()
        df[port_id_col] = -1
        df[eta_seconds_col] = np.nan
    return df
