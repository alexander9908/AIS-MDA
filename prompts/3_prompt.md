🧭 Follow-up: Fix wrong auto-zoom and south-shifted predictions (Denmark data)

Concrete symptoms (from the figures I generated)
	•	The true/current position is around Denmark (~55–57°N, 8–13°E).
	•	The predicted future appears far south (≈ 0–3°N) near the equator, clearly wrong for this dataset.
	•	With --auto_extent, the plot does not zoom to Denmark; it shows a wide Europe view because the extent is influenced by those south-shifted predicted points.
	•	Without --auto_extent, the Europe map renders, but the predicted track is still wrongly shifted south.

The four saved figures (e.g., traj_full_tptrans_val_0.png … _3.png) all show the same pattern: green “true future / current pos” near Denmark, red dashed “pred future” down near the equator.

⸻

Likely root causes (please check and fix)
	1.	Predictions not de-normalized (or de-normalized with wrong bounds).
	•	Ensure model outputs are brought back to degrees using the same global bounds used in training.
	•	Use the provided de_normalize_track from src/preprocessing/preprocessing.py for both actual and predicted—but only once.
	2.	Lat/Lon order mismatch in predictions.
	•	Predictions might be (lon, lat) while the script assumes (lat, lon) (or vice-versa). Respect --lat_idx and --lon_idx consistently for every array (past, current, true future, predicted future).
	3.	Auto-extent contaminated by outliers.
	•	The current auto-zoom likely unions actual+predicted without outlier rejection, so any equatorial point forces a huge vertical span.
	•	Add an outlier/validity filter before computing the extent.
	4.	Map drawn but lines invisible (CRS/transform mistakes).
	•	Double-check that every ax.plot(...) uses transform=ccrs.PlateCarree() and that ax.set_extent(...) also uses PlateCarree.

⸻

What to implement in src.eval.eval_traj_newnewnew

A) Safe de-normalization (actual & predicted)
	•	Import and use:

from src.preprocessing.preprocessing import de_normalize_track


	•	Create a helper that (a) applies de_normalize_track if needed, (b) validates range, (c) logs what happened:

def maybe_denorm(track, lat_idx, lon_idx, name="array"):
    lat = track[:, lat_idx]; lon = track[:, lon_idx]

    # Heuristic: normalized if both mostly inside [0,1]
    looks_norm = (
        np.nanmin(lat) >= -0.1 and np.nanmax(lat) <= 1.1 and
        np.nanmin(lon) >= -0.1 and np.nanmax(lon) <= 1.1
    )

    if looks_norm:
        track = de_normalize_track(track)
        print(f"[denorm] Applied de_normalize_track to {name}.")
    else:
        print(f"[denorm] {name} already in degrees.")

    # Validate after (de)norm
    lat = track[:, lat_idx]; lon = track[:, lon_idx]
    if not (-90 <= np.nanmin(lat) <= 90 and -90 <= np.nanmax(lat) <= 90):
        raise ValueError(f"{name}: latitude out of bounds after (de)norm. Check indices/normalization.")
    if not (-180 <= np.nanmin(lon) <= 180 and -180 <= np.nanmax(lon) <= 180):
        raise ValueError(f"{name}: longitude out of bounds after (de)norm. Check indices/normalization.")

    return track


	•	Apply this to:
	•	the past+current+true future stack,
	•	the predicted future stack, using the same lat_idx/lon_idx.

B) Strict coordinate order everywhere
	•	When extracting for plotting, always do:

lats_a = actual[:, args.lat_idx]
lons_a = actual[:, args.lon_idx]
lats_p = pred[:,   args.lat_idx]
lons_p = pred[:,   args.lon_idx]


	•	Never swap to (lat, lon) implicitly in a single call—x must be lon, y must be lat when plotting.

C) Outlier-robust auto-extent (zoom to Denmark)
	•	Add two new flags:
	•	--extent_source {both,actual,pred} (default both)
	•	--extent_outlier_sigma 3.0 (optional)
	•	Compute extent from the chosen source with outlier rejection + padding:

EUROPE = (-25, 45, 30, 72)  # lon_min, lon_max, lat_min, lat_max

def robust_extent(lats, lons, pad=0.75, clamp=EUROPE, sigma=3.0):
    # Drop NaNs
    lats = lats[np.isfinite(lats)]; lons = lons[np.isfinite(lons)]
    if lats.size == 0 or lons.size == 0:
        return clamp

    # sigma-clipping to remove wild outliers (e.g., equatorial predictions)
    def clip(arr):
        m, s = np.mean(arr), np.std(arr)
        if s == 0 or not np.isfinite(s):
            return arr
        return arr[(arr >= m - sigma*s) & (arr <= m + sigma*s)]

    lats_c = clip(lats); lons_c = clip(lons)
    if lats_c.size >= 2 and lons_c.size >= 2:
        lat_min, lat_max = float(lats_c.min()), float(lats_c.max())
        lon_min, lon_max = float(lons_c.min()), float(lons_c.max())
    else:
        lat_min, lat_max = float(lats.min()), float(lats.max())
        lon_min, lon_max = float(lons.min()), float(lons.max())

    # Ensure non-degenerate window
    if abs(lat_max - lat_min) < 0.2: lat_min -= 0.5; lat_max += 0.5
    if abs(lon_max - lon_min) < 0.2: lon_min -= 0.5; lon_max += 0.5

    # Pad & clamp
    lat_min -= pad; lat_max += pad; lon_min -= pad; lon_max += pad
    lon_min = max(clamp[0], lon_min); lon_max = min(clamp[1], lon_max)
    lat_min = max(clamp[2], lat_min); lat_max = min(clamp[3], lat_max)
    return (lon_min, lon_max, lat_min, lat_max)


	•	Extent selection logic:

if args.auto_extent:
    # Choose which points control the zoom
    if args.extent_source == "actual":
        lats_src = lats_a; lons_src = lons_a
    elif args.extent_source == "pred":
        lats_src = lats_p; lons_src = lons_p
    else:
        lats_src = np.concatenate([lats_a, lats_p])
        lons_src = np.concatenate([lons_a, lons_p])

    ext = robust_extent(lats_src, lons_src, sigma=args.extent_outlier_sigma)
else:
    ext = EUROPE

ax.set_extent(ext, crs=ccrs.PlateCarree())
print(f"[extent] Using extent {ext} (auto={args.auto_extent}, source={args.extent_source})")


	•	Important default for your dataset:
	•	Make --extent_source actual the default (or at least recommend it) so a bad prediction can’t drag the view away from Denmark.

D) Always render features; always set transform
	•	At figure creation:

proj = ccrs.PlateCarree()
fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(projection=proj))
ax.add_feature(cfeature.LAND, zorder=0)
ax.add_feature(cfeature.OCEAN, zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
gl.top_labels = False; gl.right_labels = False


	•	For every plotted line/marker:

ax.plot(lons_a, lats_a, '-',  marker='o', markersize=2, linewidth=1.2, transform=proj, label='past (input)')
ax.plot([lons_a[-1]], [lats_a[-1]], 'o', color='k', markersize=4, transform=proj, label='current pos')
ax.plot(lons_true, lats_true, '-',  linewidth=1.6, transform=proj, label='true future')
ax.plot(lons_p,    lats_p,    '--', linewidth=1.6, transform=proj, label='pred future')



E) Sanity checks and logs (catch the south-shift quickly)
	•	After (de)norm, compute medians:

print(f"[diag] med(lat) actual={np.median(lats_a):.2f}, pred={np.median(lats_p):.2f}")

	•	If pred median lat < 20° while actual median lat > 50°, log a warning:
“Prediction latitude far from actual; check normalization or index order.”

	•	If predicted points fail bounds, drop them from extent and still plot them (so we can see the error) but base zoom on actual (or on inlier preds).

⸻

CLI updates
	•	New optional flags (default values shown):

--extent_source both      # choices: both|actual|pred (recommend default=actual for this dataset)
--extent_outlier_sigma 3.0


	•	Keep all existing flags unchanged.

⸻

Acceptance criteria
	•	With no flags changed, Europe view renders correctly and predicted future is in Denmark region, not near the equator.
	•	With --auto_extent --extent_source actual, the plot zooms around Denmark even if predictions are outliers.
	•	Script prints whether it de-normalized actual/pred and shows the extent used.
	•	No CRS errors; all lines visible on both Europe view and auto-zoom.
	•	Saved PNGs under --out_dir reflect the above.

⸻

If you need more from me
	•	I can share:
	•	The exact normalization constants (LAT_MIN/MAX, LON_MIN/MAX, etc.).
	•	A small saved batch (N=1–4) of model inputs/targets/predictions to reproduce the issue.
	•	The tptrans model’s output shape and the code that maps it back to track columns.

Please implement the changes above so the plots center on Denmark and the predictions overlay the true tracks in correct lat/lon degrees.


# Output
• Here’s my plan and what I changed to fully address the issues in prompts/3_prompt.md.

  Plan

  - Normalize vs degrees: Guarantee both actual and predicted are de-normalized with the same bounds using the preprocessing helper; eliminate
    heuristics that could miss predictions.
  - Lat/lon order: Enforce a single source of truth (lat_idx/lon_idx) across past/current/true/pred so we don’t silently swap.
  - Auto-extent: Add robust, outlier-resistant auto-zoom driven by actual points by default; avoid equator outliers dragging the view.
  - Map + CRS: Prefer Cartopy (PlateCarree) with proper transforms; gracefully fall back to static basemap if Cartopy is not installed.
  - Diagnostics: Log denorm actions, chosen extent, and sanity-check medians to catch south-shift fast.

  What I implemented

  - Safe de-normalization using preprocessing function
      - Imports de_normalize_track and adds maybe_denorm(track, lat_idx, lon_idx, name), which:
          - Detects if track looks normalized (per your heuristic).
          - Reorders to [lat, lon, sog, cog] for de_normalize_track, applies it once, and writes results back to the original columns (so it
            works regardless of lat_idx/lon_idx).
          - Validates and logs the action.
      - Applied to the assembled actual track (past+true) and the predicted track. See src/eval/eval_traj_newnewnew.py:92 and usage at src/eval/
        eval_traj_newnewnew.py:205.
  - Strict coordinate order everywhere
      - Build track-like arrays with lat in lat_idx and lon in lon_idx:
          - For the past: from x[:, lat_idx]/x[:, lon_idx].
          - For true future: from y_abs with y_order mapping.
          - For predicted: from y_pred_abs with y_order mapping (after converting deltas→absolute).
      - Extract lats/lons for plotting only via lat_idx/lon_idx. See src/eval/eval_traj_newnewnew.py:184–src/eval/eval_traj_newnewnew.py:226.
  - Outlier-robust auto-extent with source control
      - Added robust_extent(lats, lons, pad=0.75, clamp=EUROPE, sigma=3.0) (sigma-clipping + padding + clamp to Europe).
      - New CLI:
          - --extent_source {both,actual,pred} (default actual).
          - --extent_outlier_sigma (default 3.0).
      - When --auto_extent is set, compute extent from chosen source; otherwise use Europe or --map_extent.
      - Logs [extent] Using extent .... See src/eval/eval_traj_newnewnew.py:56 and src/eval/eval_traj_newnewnew.py:236.
  - Cartopy map with proper transforms (fallback safe)
      - Try Cartopy:
          - projection=ccrs.PlateCarree(); ax.set_extent(ext, crs=proj).
          - Features: LAND, OCEAN, COASTLINE, BORDERS, gridlines with labels.
          - All ax.plot(..., transform=proj).
      - Fallback: GeoPandas basemap or plain Matplotlib with xlim/ylim extent if Cartopy isn’t installed.
      - See src/eval/eval_traj_newnewnew.py:211–src/eval/eval_traj_newnewnew.py:267.
  - Diagnostics and ordering for metrics
      - Logs [denorm] and [diag] sample N: med(lat) actual=.., pred=.. with warning if pred far south.
      - Convert model’s delta outputs to absolute with last past point, then align pred order with dataset ground truth (which is latlon) before
        ADE/FDE. See src/eval/eval_traj_newnewnew.py:300–src/eval/eval_traj_newnewnew.py:322.
  - Defaults tuned for this dataset
      - Default --lat_idx 0, --lon_idx 1 (dataset has [lat, lon, sog, cog]).
      - Default --extent_source actual so bad predictions don’t ruin zoom.
      - Kept existing flags so no breaking changes.

  How to run

  - Europe view with robust auto-zoom around actual tracks:
    python -m src.eval.eval_traj_newnewnew \
    --split_dir data/map_reduced/val \
    --ckpt data/checkpoints/traj_tptrans.pt \
    --model tptrans \
    --lat_idx 0 --lon_idx 1 \
    --y_order latlon \
    --past_len 64 --max_plots 8 \
    --out_dir data/figures \
    --auto_extent --extent_source actual --extent_outlier_sigma 3.0
  - Fixed Europe frame:
    drop --auto_extent, or use --map_extent -25 45 30 72.

  Notes

  - Cartopy fallback: If Cartopy isn’t installed, the script auto-falls back to a static basemap or plain axes. You’ll still get correct plots in
    lon/lat, just without CRS transforms.
  - Metrics remain in coordinate units; if you want meter-based ADE/FDE next, I can add a geodesic conversion.

  Open questions

  - Confirm Y order of your dataset (I default to --y_order latlon, which matches src/utils/datasets.py). If your Y is [lon,lat], set --y_order
    lonlat.
  - If you want me to force denormalization always (even if values look outside the heuristic bounds), I can change maybe_denorm to be more
    permissive.

  Files changed

  - src/eval/eval_traj_newnewnew.py: full plotting overhaul with denorm, robust extents, Cartopy support, and stricter lat/lon handling.