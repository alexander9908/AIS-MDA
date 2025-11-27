# src/eval/build_water_mask_V2.py
"""
Build a raster water mask using the roaring-landmask package.

Public API (kept identical to V1):
  - make_water_mask(lat_min, lat_max, lon_min, lon_max, n_lat, n_lon) -> np.ndarray[bool]
      Returns a boolean array of shape [n_lat, n_lon] with True = water, False = land.
  - snap_to_water_path(lat_idx_seq, lon_idx_seq, wm) -> (lat_idx_seq, lon_idx_seq)
      If any index is on land (wm=False), snap it to the nearest water pixel using a BFS
      over 8-neighborhood to preserve path connectivity.

Notes
-----
- Grid cell centers are used when querying the landmask.
- This file requires: `pip install roaring-landmask`
"""

from __future__ import annotations
import numpy as np
from collections import deque

try:
    # roaring-landmask: fast point-in-land queries
    # API per README: RoaringLandmask.new(); .contains_many(lons, lats) -> array[bool] for "on land"
    from roaring_landmask import RoaringLandmask
except Exception:
    RoaringLandmask = None


def _grid_centers(lat_min: float, lat_max: float, lon_min: float, lon_max: float,
                  n_lat: int, n_lon: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return 1D arrays of cell-center latitudes and longitudes (size n_lat and n_lon).
    """
    # Edges then centers for robust numerics
    lat_edges = np.linspace(lat_min, lat_max, int(n_lat) + 1, dtype=np.float64)
    lon_edges = np.linspace(lon_min, lon_max, int(n_lon) + 1, dtype=np.float64)
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    return lat_centers, lon_centers


def make_water_mask(lat_min: float, lat_max: float,
                    lon_min: float, lon_max: float,
                    n_lat: int, n_lon: int) -> np.ndarray:
    """
    Build a boolean mask [n_lat, n_lon] with True = water, False = land.

    Implementation details:
    - Uses roaring-landmask to test cell centers.
    - roaring-landmask returns True for "on land"; we invert to get water.
    """
    if RoaringLandmask is None:
        # Fallback: all water (so we never block sampling) — matches prior V1 fallback behavior.
        # You can raise here if you prefer hard failure.
        return np.ones((int(n_lat), int(n_lon)), dtype=bool)

    # Prepare grid centers
    latc, lonc = _grid_centers(float(lat_min), float(lat_max),
                               float(lon_min), float(lon_max),
                               int(n_lat), int(n_lon))
    # Meshgrid in the same orientation as the mask: [lat, lon]
    # NOTE: roaring-landmask expects (lon, lat) arrays when calling contains_many.
    Lon, Lat = np.meshgrid(lonc, latc)

    # Query roaring-landmask
    rl = RoaringLandmask.new()
    on_land = rl.contains_many(Lon.ravel(), Lat.ravel())  # True where land
    on_land = np.asarray(on_land, dtype=bool).reshape(int(n_lat), int(n_lon))

    # We return True for water
    water = ~on_land
    return water


def _neighbors_8(i: int, j: int) -> list[tuple[int, int]]:
    """8-neighborhood offsets for BFS."""
    return [
        (i - 1, j - 1), (i - 1, j), (i - 1, j + 1),
        (i, j - 1),                 (i, j + 1),
        (i + 1, j - 1), (i + 1, j), (i + 1, j + 1),
    ]


def snap_to_water_path(lat_idx_seq: np.ndarray,
                       lon_idx_seq: np.ndarray,
                       wm: np.ndarray,
                       max_radius: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """
    Snap an index path to the nearest water pixels according to a boolean water mask.

    Parameters
    ----------
    lat_idx_seq, lon_idx_seq : int arrays of equal length
        Row/column indices into wm for each predicted timestep.
    wm : 2D boolean array (H, W), True = water, False = land.
    max_radius : int
        Safety bound for BFS expansion.

    Returns
    -------
    (lat_idx_seq, lon_idx_seq) : potentially modified arrays where any land
    pixel at step t is replaced by the nearest water pixel (in 8-neighborhood BFS distance).
    """
    H, W = wm.shape
    lat_idx_seq = np.asarray(lat_idx_seq, dtype=int)
    lon_idx_seq = np.asarray(lon_idx_seq, dtype=int)

    for t in range(len(lat_idx_seq)):
        i, j = int(lat_idx_seq[t]), int(lon_idx_seq[t])

        # Clamp indices just in case
        if not (0 <= i < H and 0 <= j < W):
            i = min(max(i, 0), H - 1)
            j = min(max(j, 0), W - 1)

        if wm[i, j]:
            # already on water; keep as is
            lat_idx_seq[t], lon_idx_seq[t] = i, j
            continue

        # BFS to nearest water
        q = deque()
        q.append((i, j))
        seen = set([(i, j)])
        found = None
        radius = 0

        # Layered BFS to allow early stop within max_radius
        while q and radius <= max_radius and found is None:
            # Expand one layer
            for _ in range(len(q)):
                ci, cj = q.popleft()
                if 0 <= ci < H and 0 <= cj < W and wm[ci, cj]:
                    found = (ci, cj)
                    break
                for ni, nj in _neighbors_8(ci, cj):
                    if 0 <= ni < H and 0 <= nj < W and (ni, nj) not in seen:
                        seen.add((ni, nj))
                        q.append((ni, nj))
            radius += 1

        if found is not None:
            lat_idx_seq[t], lon_idx_seq[t] = found  # snap to nearest water

    return lat_idx_seq, lon_idx_seq
