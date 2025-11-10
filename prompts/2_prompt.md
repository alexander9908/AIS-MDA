🔧 Fix eval_traj_newnewnew: show actual vs. predicted on a Europe map (with smart zoom) and correct de-normalization

Context

I’m running a trajectory prediction pipeline. The evaluation plots are wrong:
	•	When I render Europe, I see a base map but no trajectories.
	•	When I pass --auto_extent, I see trajectories but the map is blank (no basemap features).
	•	I also suspect predictions are still normalized, not in real lat/lon.

Commands I run

# Europe view (broken: map shows, lines missing)
python -m src.eval.eval_traj_newnewnew \
  --split_dir data/map_reduced/val \
  --ckpt data/checkpoints/traj_tptrans.pt \
  --model tptrans \
  --lat_idx 1 --lon_idx 0 \
  --past_len 64 --max_plots 8 \
  --out_dir data/figures

# Auto-zoom (broken: lines show, map blank)
python -m src.eval.eval_traj_newnewnew \
  --split_dir data/map_reduced/val \
  --ckpt data/checkpoints/traj_tptrans.pt \
  --model tptrans \
  --lat_idx 1 --lon_idx 0 \
  --past_len 64 --max_plots 8 \
  --out_dir data/figures \
  --auto_extent   # omit to see full Europe view (-25, 45, 30, 72)

Data columns (order):

LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI

Available de-normalizer

In src/preprocessing/preprocessing.py:

def de_normalize_track(track: np.ndarray) -> np.ndarray:
    """Denormalizes a single track."""
    denorm_track = copy.deepcopy(track)
    denorm_track[:, LAT] = denorm_track[:, LAT] * (LAT_MAX - LAT_MIN) + LAT_MIN
    denorm_track[:, LON] = denorm_track[:, LON] * (LON_MAX - LON_MIN) + LON_MIN
    denorm_track[:, SOG] = denorm_track[:, SOG] * SPEED_MAX
    denorm_track[:, COG] = denorm_track[:, COG] * 360.0
    return denorm_track


⸻

What I need you to do

Goal: Make src.eval.eval_traj_newnewnew reliably plot actual vs predicted trajectories on a Europe basemap with an optional auto-zoom to the trajectory region, saving figures to --out_dir.

Requirements
	1.	Cartopy + correct CRS
	•	Use cartopy.crs.PlateCarree() for data coordinates (lat/lon).
	•	Use consistent transforms when plotting:

ax.plot(lons, lats, transform=ccrs.PlateCarree(), ...)


	•	Draw coastlines, borders, and land/ocean (Natural Earth features) so maps work both at continent scale and when zoomed. Avoid tile providers unless already configured; features must render offline.

	2.	Europe extent (default)
	•	If --auto_extent is not provided, set:

europe_extent = (-25, 45, 30, 72)  # (lon_min, lon_max, lat_min, lat_max)
ax.set_extent(europe_extent, crs=ccrs.PlateCarree())


	3.	Auto-extent (when --auto_extent is set)
	•	Compute extent from both actual & predicted points for that sample.
	•	Add padding (e.g., 0.5–1.0 degrees) and clamp to the Europe bbox above.
	•	Handle single-point cases (ensure nonzero span so Cartopy shows features).
	4.	De-normalization
	•	Detect if inputs/predictions are normalized (e.g., lat/lon within ~[0,1]) and de-normalize exactly once using de_normalize_track from src.preprocessing.preprocessing.
	•	Ensure you do not de-normalize data that is already in degrees.
	•	Apply the same logic to model outputs (predictions) before plotting.
	5.	Lat/Lon indexing
	•	Respect --lat_idx and --lon_idx, and never swap order in plotting:

lats = arr[:, lat_idx]
lons = arr[:, lon_idx]
ax.plot(lons, lats, transform=ccrs.PlateCarree(), ...)


	6.	Visibility & styling
	•	Plot actual trajectory as a solid line with small markers.
	•	Plot predicted as a dashed line with contrasting markers.
	•	Add legend: “Actual”, “Predicted”.
	•	Add gridlines with labels (lat/lon) and title with MMSI and time span.
	•	Save figures as PNG in --out_dir, e.g. traj_{mmsi}_{idx}.png.
	7.	Robustness & diagnostics
	•	If no points fall inside the current extent, log a warning and auto-zoom anyway (unless --no_auto_fallback is set).
	•	Before plotting, assert lat ∈ [-90, 90], lon ∈ [-180, 180] after de-normalization; if not, print a clear error suggesting likely double-normalization or wrong indices.
	•	Print whether de-normalization was applied, and the computed extent per plot.
	8.	CLI parity
	•	Keep all current flags working; add --no_auto_fallback (optional).
	•	Keep --past_len, --max_plots, and output directory behavior.

⸻

Suggested code changes (you can implement equivalent)

Imports & features

import cartopy.crs as ccrs
import cartopy.feature as cfeature

def add_basemap(ax):
    ax.add_feature(cfeature.LAND, zorder=0)
    ax.add_feature(cfeature.OCEAN, zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.right_labels = False
    gl.top_labels = False

Safe de-normalization

from src.preprocessing.preprocessing import de_normalize_track

def maybe_denorm(track, lat_idx, lon_idx):
    lat = track[:, lat_idx]; lon = track[:, lon_idx]
    # Heuristic: normalized if both inside [0, 1] with margin
    if (lat.min() >= -0.05 and lat.max() <= 1.05 and
        lon.min() >= -0.05 and lon.max() <= 1.05):
        track = de_normalize_track(track)
        applied = True
    else:
        applied = False
    # Validate
    lat = track[:, lat_idx]; lon = track[:, lon_idx]
    if not (-90 <= lat.min() <= 90 and -90 <= lat.max() <= 90 and
            -180 <= lon.min() <= 180 and -180 <= lon.max() <= 180):
        raise ValueError("Lat/Lon out of bounds after (de)normalization. Check indices or duplicate denorm.")
    return track, applied

Extent logic

EUROPE = (-25, 45, 30, 72)  # (lon_min, lon_max, lat_min, lat_max)

def compute_auto_extent(lats, lons, pad=0.75):
    lon_min, lon_max = float(lons.min()), float(lons.max())
    lat_min, lat_max = float(lats.min()), float(lats.max())
    # Handle degenerate extents
    if abs(lon_max - lon_min) < 0.2:
        lon_min -= 0.5; lon_max += 0.5
    if abs(lat_max - lat_min) < 0.2:
        lat_min -= 0.5; lat_max += 0.5
    lon_min -= pad; lon_max += pad; lat_min -= pad; lat_max += pad
    # Clamp to Europe bbox
    lon_min = max(EUROPE[0], lon_min); lon_max = min(EUROPE[1], lon_max)
    lat_min = max(EUROPE[2], lat_min); lat_max = min(EUROPE[3], lat_max)
    return (lon_min, lon_max, lat_min, lat_max)

Plotting snippet

proj = ccrs.PlateCarree()
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=proj))
add_basemap(ax)

actual_denorm, den_a = maybe_denorm(actual, lat_idx, lon_idx)
pred_denorm, den_p = maybe_denorm(pred,   lat_idx, lon_idx)

lats_a, lons_a = actual_denorm[:, lat_idx], actual_denorm[:, lon_idx]
lats_p, lons_p = pred_denorm[:, lat_idx],   pred_denorm[:, lon_idx]

if args.auto_extent:
    ext = compute_auto_extent(
        np.concatenate([lats_a, lats_p]),
        np.concatenate([lons_a, lons_p])
    )
else:
    ext = EUROPE
ax.set_extent(ext, crs=proj)

# Plot after extent so features redraw correctly
ax.plot(lons_a, lats_a, '-',  marker='o', markersize=2, linewidth=1.2, transform=proj, label='Actual')
ax.plot(lons_p, lats_p, '--', marker='x', markersize=3, linewidth=1.2, transform=proj, label='Predicted')
ax.legend(loc='lower left')

title = f"MMSI {mmsi} • {start_ts} → {end_ts} • denorm(actual={den_a}, pred={den_p})"
ax.set_title(title)
fig.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close(fig)

Fallback if lines not visible at Europe scale

If no points fall inside EUROPE, automatically compute and use compute_auto_extent(...) unless --no_auto_fallback is set; emit a warning log.

⸻

Acceptance criteria
	•	Running the two commands above produces PNGs in data/figures/.
	•	Europe view shows coastlines/borders and actual/predicted lines (if in Europe).
	•	--auto_extent view shows the same basemap features (not blank) and zooms around the tracks.
	•	Lat/Lon are in degrees, correctly oriented (lon on x, lat on y).
	•	Script logs whether it applied de-normalization for actual/predicted.
	•	For at least one sample near Denmark, the auto-zoom frames the track around Denmark.

⸻

Notes & likely root causes (for you to check)
	•	Missing transform=ccrs.PlateCarree() on ax.plot often makes lines invisible.
	•	Using a tile provider without internet/cache can cause a blank map at zoom. Use Cartopy’s Natural Earth features instead (offline-safe).
	•	Predictions left in normalized space (0–1) will never show on a degrees map. Fix via de_normalize_track.




# output

• Here’s my plan and what I changed to make trajectory evaluation work as you described.

  Plan

  - Diagnose issues: coord normalization, Y order, model output (deltas vs abs), mapping.
  - Fix evaluation: convert deltas→absolute with last past point; handle Y order; optional de-normalization to degrees; robust basemap.
  - Validate interface: clear CLI flags to align with your trained model and dataset.

  What I fixed

  - Correctly handles TPTrans/GRU outputs as deltas, converts to absolute by cumsum + last input point. src/eval/eval_traj_newnewnew.py:300
  - Supports Y column order via --y_order {latlon,lonlat} (default latlon to match this repo’s dataset). src/eval/eval_traj_newnewnew.py:269
  - Adds optional de-normalization to degrees (--denorm) with bounds (--lat_min/--lat_max/--lon_min/--lon_max). Defaults read from preprocessing
    constants if available. src/eval/eval_traj_newnewnew.py:18, 267–273
  - Basemap rendering is resilient: uses a provided --basemap_path, or GeoPandas legacy dataset, GeoDatasets, or pyogrio’s Natural Earth fixture.
    src/eval/eval_traj_newnewnew.py:24–114
  - Plot overlays past, true, and predicted trajectories with equal aspect and extents; supports --auto_extent zoom or fixed Europe view. src/
    eval/eval_traj_newnewnew.py:120–171
  - Updated README usage with these flags. README.md:153–164

  How to run (for your pipeline)
  Your dataset is normalized (0–1) to LAT_MIN=54 LAT_MAX=58 LON_MIN=6 LON_MAX=16, and X features are [lat, lon, sog, cog]. Use:

  - lat index 0, lon index 1
  - Y is [lat, lon] (order latlon)
  - TPTrans outputs deltas → use denorm for plotting

  Example:

  - Europe zoom per-sample:
    python -m src.eval.eval_traj_newnewnew \
    --split_dir data/map_reduced/val \
    --ckpt data/checkpoints/traj_tptrans.pt \
    --model tptrans \
    --lat_idx 0 --lon_idx 1 \
    --y_order latlon \
    --past_len 64 --max_plots 8 \
    --out_dir data/figures \
    --denorm --lat_min 54 --lat_max 58 --lon_min 6 --lon_max 16 \
    --auto_extent
  - Fixed Europe frame:
    omit --auto_extent or pass --map_extent -25 45 30 72

  What likely broke before

  - Normalized [0..1] lat/lon were plotted directly on a Europe basemap (degrees), so tracks looked wrong or off-map.
  - Y order mismatch (script assumed [lon,lat] but dataset provided [lat,lon]).
  - Model outputs deltas; plotting as absolutes created offset/zero-length lines.

  Questions to ensure perfect alignment

  - Confirm Y ordering: Is ground-truth Y saved as [lat,lon] (default) or [lon,lat]? If [lon,lat], run with --y_order lonlat.
  - Confirm X ordering for your run: do you want --lat_idx 1 --lon_idx 0? The repo’s pipeline uses [lat,lon,...] so the correct flags here are
    --lat_idx 0 --lon_idx 1.
  - Different normalization bounds? If so, pass your ROI via --lat_min/--lat_max/--lon_min/--lon_max.

  If you want ADE/FDE in meters (instead of normalized units), I can add on-the-fly geodesic conversion for metrics as a next step.