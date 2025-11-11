# src/utils/water_guidance.py
# from __future__ import annotations
# import numpy as np
# from typing import Tuple
# from roaring_landmask import RoaringLandmask
# 
# _rl = None
# def _get_rl() -> RoaringLandmask:
#     global _rl
#     if _rl is None:
#         _rl = RoaringLandmask.new()
#     return _rl
# 
# def is_water(lat: float, lon: float) -> bool:
#     """True if the coordinate is water (i.e., NOT land)."""
#     return not _get_rl().contains(float(lon), float(lat))
# 
# def project_to_water(prev_lat: float, prev_lon: float,
#                      lat: float, lon: float,
#                      iters: int = 18) -> Tuple[float, float]:
#     """
#     If (lat, lon) lies on land, bisection along the segment from the last water point
#     (prev_lat, prev_lon) towards (lat, lon) to find the nearest water point on that ray.
#     Returns a coordinate guaranteed to be on water (or as close as possible).
#     """
#     rl = _get_rl()
#     # Ensure 'a' is water; if not, nudge backward slightly
#     a_lat, a_lon = float(prev_lat), float(prev_lon)
#     if rl.contains(a_lon, a_lat):  # last point somehow on land -> search a tiny ring
#         # try small radial nudges (few meters in deg)
#         eps = 1e-4
#         for dy in (-eps, 0.0, eps):
#             for dx in (-eps, 0.0, eps):
#                 if dx == 0.0 and dy == 0.0: continue
#                 if not rl.contains(a_lon + dx, a_lat + dy):
#                     a_lat += dy; a_lon += dx
#                     break
# 
#     b_lat, b_lon = float(lat), float(lon)
#     # Quick accept
#     if not rl.contains(b_lon, b_lat):
#         return b_lat, b_lon
# 
#     # Bisection toward shoreline
#     for _ in range(iters):
#         m_lat = 0.5 * (a_lat + b_lat)
#         m_lon = 0.5 * (a_lon + b_lon)
#         if rl.contains(m_lon, m_lat):
#             # midpoint on land -> move b inward
#             b_lat, b_lon = m_lat, m_lon
#         else:
#             # midpoint on water -> advance a
#             a_lat, a_lon = m_lat, m_lon
#     # 'a' is the closest water point along the segment
#     return a_lat, a_lon
# 
# def make_water_mask_grid(lat_min: float, lat_max: float,
#                          lon_min: float, lon_max: float,
#                          n_lat: int, n_lon: int) -> np.ndarray:
#     """
#     Convenience for precomputing a raster mask aligned to a bin grid:
#     returns (n_lat, n_lon) with True = water.
#     """
#     from roaring_landmask import RoaringLandmask
#     rl = _get_rl()
#     lat_edges = np.linspace(lat_min, lat_max, int(n_lat) + 1, dtype=np.float64)
#     lon_edges = np.linspace(lon_min, lon_max, int(n_lon) + 1, dtype=np.float64)
#     latc = 0.5*(lat_edges[:-1] + lat_edges[1:])
#     lonc = 0.5*(lon_edges[:-1] + lon_edges[1:])
#     Lon, Lat = np.meshgrid(lonc, latc)  # [n_lat, n_lon]
#     on_land = rl.contains_many(Lon.ravel(), Lat.ravel()).reshape(int(n_lat), int(n_lon))
#     return ~on_land  # True = water
# 
# def apply_water_mask_to_scores(score_2d: np.ndarray, wm: np.ndarray) -> None:
#     """
#     In-place: set land cells to -inf in a 2D score map.
#     score_2d shape: [n_lat, n_lon], wm True=water, False=land.
#     """
#     score_2d[~wm] = -np.inf




# src/utils/water_guidance.py
from __future__ import annotations
from typing import Tuple
import numpy as np
from roaring_landmask import RoaringLandmask

_RL = None
def _rl() -> RoaringLandmask:
    global _RL
    if _RL is None:
        _RL = RoaringLandmask.new()
    return _RL

def is_water(lat: float, lon: float) -> bool:
    """True if (lat, lon) is water."""
    return not _rl().contains(float(lon), float(lat))

def project_to_water(prev_lat: float, prev_lon: float,
                     lat: float, lon: float,
                     iters: int = 18) -> Tuple[float, float]:
    """
    If target is on land, bisection along the segment from last water point
    (prev_lat, prev_lon) toward (lat, lon) to find nearest water.
    """
    rl = _rl()
    a_lat, a_lon = float(prev_lat), float(prev_lon)
    # Ensure the anchor is on water; if not, nudge a hair.
    if rl.contains(a_lon, a_lat):
        eps = 1e-4
        for dy in (-eps, 0, eps):
            for dx in (-eps, 0, eps):
                if dx == 0 and dy == 0:
                    continue
                if not rl.contains(a_lon + dx, a_lat + dy):
                    a_lat += dy; a_lon += dx
                    break
    b_lat, b_lon = float(lat), float(lon)
    if not rl.contains(b_lon, b_lat):
        return b_lat, b_lon
    for _ in range(iters):
        m_lat = 0.5 * (a_lat + b_lat)
        m_lon = 0.5 * (a_lon + b_lon)
        if rl.contains(m_lon, m_lat):
            b_lat, b_lon = m_lat, m_lon
        else:
            a_lat, a_lon = m_lat, m_lon
    return a_lat, a_lon

