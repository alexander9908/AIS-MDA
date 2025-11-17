# src/eval/build_water_mask.py
import numpy as np
from collections import deque
try:
    import cartopy.io.shapereader as shpreader
    from shapely.geometry import shape, Point
except Exception:
    shpreader = None

def make_water_mask(lat_min, lat_max, lon_min, lon_max, n_lat, n_lon):
    """
    Returns a boolean array [n_lat, n_lon] where True = water, False = land.
    Very simple/slow rasterization using Natural Earth land polygons.
    """
    # Fallback: if cartopy is missing, return "all water"
    if shpreader is None:
        return np.ones((n_lat, n_lon), dtype=bool)

    reader = shpreader.natural_earth(resolution='10m', category='physical', name='land')
    geoms = [shape(r.geometry) for r in shpreader.Reader(reader).records()]

    lats = np.linspace(lat_min, lat_max, n_lat, endpoint=False) + (lat_max-lat_min)/n_lat/2
    lons = np.linspace(lon_min, lon_max, n_lon, endpoint=False) + (lon_max-lon_min)/n_lon/2

    mask = np.ones((n_lat, n_lon), dtype=bool)  # start all water
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            p = Point(lon, lat)
            # mark land cells as False
            if any(g.contains(p) for g in geoms):
                mask[i, j] = False

    # Lightly erode water so coastline has a buffer (safer snapping)
    return _erode_water(mask, radius=3)

def _erode_water(wm, radius=1):
    """Turn any water pixel that touches land within 'radius' 8-neighborhood steps into land."""
    H, W = wm.shape
    out = wm.copy()
    # 8-neighborhood offsets
    neigh = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for _ in range(int(max(0, radius))):
        to_land = []
        for i in range(H):
            for j in range(W):
                if not wm[i, j]:   # already land
                    continue
                for di, dj in neigh:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < H and 0 <= nj < W and not wm[ni, nj]:
                        to_land.append((i, j)); break
        for (i, j) in to_land:
            out[i, j] = False
        wm = out.copy()
    return out



def snap_to_water_path(lat_idx_seq, lon_idx_seq, wm):
    """Snap any land cell in a path to the nearest water cell (grid BFS)."""
    H, W = wm.shape
    lat_idx_seq = lat_idx_seq.copy()
    lon_idx_seq = lon_idx_seq.copy()

    # 8-neighborhood
    neigh = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    for t in range(len(lat_idx_seq)):
        i, j = int(lat_idx_seq[t]), int(lon_idx_seq[t])
        if 0 <= i < H and 0 <= j < W and wm[i, j]:
            continue  # already water
        # BFS from (i,j) to nearest water
        q = deque([(i, j)])
        seen = set([(i, j)])
        found = None
        while q:
            ci, cj = q.popleft()
            if 0 <= ci < H and 0 <= cj < W and wm[ci, cj]:
                found = (ci, cj)
                break
            for di, dj in neigh:
                ni, nj = ci + di, cj + dj
                if 0 <= ni < H and 0 <= nj < W and (ni, nj) not in seen:
                    seen.add((ni, nj))
                    q.append((ni, nj))
        if found is not None:
            lat_idx_seq[t], lon_idx_seq[t] = found
    return lat_idx_seq, lon_idx_seq
