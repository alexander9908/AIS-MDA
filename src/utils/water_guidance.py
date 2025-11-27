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

