from __future__ import annotations
import numpy as np

EARTH_RADIUS_M = 6371000.0

def haversine_distance_m(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arcsin(np.sqrt(a))
    return float(EARTH_RADIUS_M * c)

def project_to_local_xy(lat: np.ndarray, lon: np.ndarray, lat0: float | None = None, lon0: float | None = None):
    """Simple equirectangular projection around (lat0,lon0). For small regions only."""
    if lat0 is None: lat0 = float(np.nanmean(lat))
    if lon0 is None: lon0 = float(np.nanmean(lon))
    x = np.radians(lon - lon0) * EARTH_RADIUS_M * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_RADIUS_M
    return x, y
