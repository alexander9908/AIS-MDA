🔧 Fix Prompt — Denmark Basemap + Correct Anchoring + Matching Tail Length (Real Lat/Lon)

🧠 Problem recap (from attached figures)
	1.	Basemap missing — only axes appear; no land/sea/coastlines despite Denmark clamp.
	2.	Pred tail detached — red dashed segment does not start at the last blue point.
	3.	Pred tail longer than true tail — red extends beyond green; and coordinates look offset (normalization mismatch).

We must:
	•	Always render Denmark basemap (features + extent),
	•	Produce real lat/lon (proper de-normalization),
	•	Attach red to the last blue point,
	•	Make red no longer than green unless explicitly rolled out,
	•	Keep all previous features (full-trip pipeline, --mmsi modes, metadata, etc.).

⸻

✅ Changes to implement

A) Basemap: one shared builder, always called, with Denmark clamp & fallbacks

In src/eval/eval_traj_newnewnew, create a single map builder and use it in all code paths (multi/single/all):

# top of file
import cartopy.crs as ccrs
import cartopy.feature as cfeature

DEFAULT_DK_EXTENT = [6.0, 16.0, 54.0, 58.0]  # lon_min, lon_max, lat_min, lat_max

def build_denmark_axes(auto_extent, extent_source_points, sigma, figsize=(10,6)):
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})

    # features with robust fallback
    try:
        ax.add_feature(cfeature.OCEAN, zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="0.92", zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=2)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=2)
    except Exception as e:
        print(f"[warn] cartopy feature load failed: {e}; using coastlines only")
        ax.coastlines(resolution="50m", linewidth=0.6)

    # gridlines
    try:
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
        gl.top_labels = False; gl.right_labels = False
    except Exception as e:
        print(f"[warn] gridlines failed: {e}")

    # extent
    if auto_extent and len(extent_source_points) >= 1:
        # extent_source_points is Nx2 of [lon, lat] AFTER DENORM
        arr = np.asarray(extent_source_points, dtype=float)
        # sigma trim
        lon = arr[:,0]; lat = arr[:,1]
        m_lon, s_lon = np.nanmean(lon), np.nanstd(lon)
        m_lat, s_lat = np.nanmean(lat), np.nanstd(lat)
        lon_min, lon_max = m_lon - sigma*s_lon, m_lon + sigma*s_lon
        lat_min, lat_max = m_lat - sigma*s_lat, m_lat + sigma*s_lat
        # clamp to Denmark
        lon_min = max(lon_min, DEFAULT_DK_EXTENT[0]); lon_max = min(lon_max, DEFAULT_DK_EXTENT[1])
        lat_min = max(lat_min, DEFAULT_DK_EXTENT[2]); lat_max = min(lat_max, DEFAULT_DK_EXTENT[3])
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
    else:
        ax.set_extent(DEFAULT_DK_EXTENT, crs=proj)

    return fig, ax, proj

Use it exactly once per figure and ensure every plot/scatter call includes transform=proj.

⸻

B) Normalization/De-normalization: do it once, in the right order
	•	Model input must use the same 4 features as training: [LAT, LON, SOG, COG], normalized.
	•	Model output (pred) must be de-normalized to real degrees before anchoring/plotting.
	•	Past and true future also de-normalized to real degrees, using the same constants used at training:

# (if your files are normalized already, skip re-normalization)
# de_normalize_track expects the training-time constants
from src.preprocessing.preprocessing import de_normalize_track

# Apply once:
past_denorm       = de_normalize_track(past.copy())
true_future_denorm= de_normalize_track(true_future.copy())
pred_denorm       = de_normalize_track(pred.copy())   # do this after model forward
full_denorm       = de_normalize_track(full_trip.copy())

Double-check you are not calling de-norm twice on any segment.
Add temporary prints of min/max lat/lon per segment to confirm they fall inside Denmark (54–58N, 6–16E).

⸻

C) Pred anchoring: hard-attach red to last blue point

Add robust utilities:

from math import radians, sin, cos, asin, sqrt

def haversine_km(lon1, lat1, lon2, lat2):
    R=6371.0
    dlon, dlat = radians(lon2-lon1), radians(lat2-lat1)
    a=sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*R*asin(sqrt(a))

def anchor_absolute(pred_denorm, last_lat, last_lon, lat_idx, lon_idx):
    """Shift abs predictions so pred[0] == last past (lat/lon)."""
    shift_lat = last_lat - pred_denorm[0, lat_idx]
    shift_lon = last_lon - pred_denorm[0, lon_idx]
    out = pred_denorm.copy()
    out[:, lat_idx] += shift_lat
    out[:, lon_idx] += shift_lon
    return out

def reconstruct_from_delta(delta_denorm, last_lat, last_lon, lat_idx, lon_idx):
    """If model outputs deltas (degrees), cum-sum starting from last past."""
    out = delta_denorm.copy()
    out[0, lat_idx] = last_lat + out[0, lat_idx]
    out[0, lon_idx] = last_lon + out[0, lon_idx]
    for t in range(1, len(out)):
        out[t, lat_idx] = out[t-1, lat_idx] + delta_denorm[t, lat_idx]
        out[t, lon_idx] = out[t-1, lon_idx] + delta_denorm[t, lon_idx]
    return out

Add CLI flag:

--pred_mode {absolute,delta}   # default: absolute

After de-norm and before plotting:

lat_i, lon_i = args.lat_idx, args.lon_idx
last_lat, last_lon = past_denorm[-1, lat_i], past_denorm[-1, lon_i]

if args.pred_mode == "delta":
    pred_denorm = reconstruct_from_delta(pred_denorm, last_lat, last_lon, lat_i, lon_i)
else:
    pred_denorm = anchor_absolute(pred_denorm, last_lat, last_lon, lat_i, lon_i)

d0 = haversine_km(last_lon, last_lat, pred_denorm[0, lon_i], pred_denorm[0, lat_i])
if d0 > 0.5:
    print(f"[warn] first pred still {d0:.2f} km away — check y_order/indices/denorm.")

If delta fixes it, your model is outputting deltas; leave --pred_mode delta as default for this checkpoint.

⸻

D) Make pred length ≤ true tail by default

Guarantee red isn’t longer than green unless the user explicitly asks to roll out:
	•	If --iter_rollout off → pred_len = min(model.horizon, len(true_future_denorm))
	•	If --cap_future given → pred_len = min(pred_len, cap_future)
	•	Slice: pred_denorm = pred_denorm[:pred_len]

Add logs:

[pred] horizon=<H> true_tail=<T> pred_len=<P> mode=absolute|delta rollout=on|off


⸻

E) Plot using consistent (lon, lat), always with transform

Use a safe helper to produce lon/lat arrays:

def to_lonlat(arr, lat_i, lon_i):
    return arr[:, lon_i], arr[:, lat_i]

Then:

full_lon, full_lat = to_lonlat(full_denorm, lat_i, lon_i)
past_lon, past_lat = to_lonlat(past_denorm, lat_i, lon_i)
true_lon, true_lat = to_lonlat(true_future_denorm, lat_i, lon_i)
pred_lon, pred_lat = to_lonlat(pred_denorm, lat_i, lon_i)

fig, ax, proj = build_denmark_axes(
    auto_extent=args.auto_extent,
    extent_source_points=np.vstack([
        np.column_stack([full_lon, full_lat]),
        np.column_stack([past_lon, past_lat]),
        np.column_stack([true_lon, true_lat]),
        np.column_stack([pred_lon, pred_lat]),
    ]),
    sigma=args.extent_outlier_sigma,
)

ax.plot(full_lon, full_lat, "-", color="0.85", lw=2, transform=proj, zorder=1, label="full trip (context)")
ax.plot(past_lon, past_lat, "-", lw=2, transform=proj, zorder=3, label="past (input)")
ax.scatter([past_lon[-1]], [past_lat[-1]], s=16, c="k", transform=proj, zorder=4, label="current pos")
ax.plot(true_lon, true_lat, "-", lw=2, color="green", transform=proj, zorder=3, label="true future")
ax.plot(pred_lon, pred_lat, "--", lw=2, color="red", transform=proj, zorder=3, label="pred future")


⸻

🧪 Quick validation commands
	1.	Absolute mode (default) — verify attachment & Denmark map:

python -m src.eval.eval_traj_newnewnew \
  --split_dir data/map_reduced/val \
  --ckpt data/checkpoints/traj_tptrans.pt \
  --model tptrans \
  --max_plots 8 \
  --out_dir data/figures \
  --full_trip \
  --auto_extent --extent_source actual --extent_outlier_sigma 3.0 \
  --denorm --lat_min 54 --lat_max 58 --lon_min 6 --lon_max 16 \
  --lat_idx 0 --lon_idx 1 --y_order latlon \
  --pred_mode absolute \
  --log_skip_reasons --seed 0

	2.	If still offset, try delta (common if model outputs displacements):

... --pred_mode delta


⸻

✅ Acceptance criteria
	•	Every figure shows Denmark basemap (land/ocean/coastline/borders), with default clamp or robust auto-extent.
	•	Red tail starts exactly at the last blue point (first-point haversine < 0.5 km; warn otherwise).
	•	Red length ≤ green unless --iter_rollout is on; with rollout, length follows tail or --cap_future.
	•	Coordinates on all lines are real lat/lon degrees (54–58N, 6–16E), not normalized values.
	•	Works identically across default multi-sample, single --mmsi, and --mmsi all.

⸻

Notes (why your current plots look wrong)
	•	Missing basemap = features never added in the active code path (or exception with no fallback). The builder above fixes that everywhere.
	•	Red offset = anchoring done on normalized coords, or lat/lon swapped, or model produces deltas but treated as absolutes. Doing post-denorm anchoring and supporting --pred_mode delta removes the ambiguity.
	•	Red longer than green = horizon/rollout uncapped. The slicing rule fixes it.

