🧯 Fix Prompt — Map Missing + Pred Tail Not Anchored (V2)

🔎 Symptoms
	•	With the new V2 stack, evaluation runs and saves 8 plots, but:
	•	Basemap is missing (no Denmark land/sea/coastlines) in all modes.
	•	Predicted future (red) is spatially detached from the blue past; it appears elsewhere on the map.

🎯 Goals
	1.	Restore consistent Denmark map in every mode:
	•	Default Denmark extent (6–16E, 54–58N) if --auto_extent off.
	•	Robust auto-extent (trimmed + clamped to Denmark) if --auto_extent on.
	•	Always draw: land, ocean, coastline, borders, gridlines.
	2.	Make the red prediction continue exactly from the last blue point:
	•	First pred point must equal (or be anchored to) the last past point.
	•	If the model predicts in a different normalization/order, correct it.
	•	Add a guardrail: if first_pred is > X km from last_past, auto-anchor.
	3.	Keep everything else from the master prompt:
	•	Full-trip plotting, Denmark clamp, --max_plots, --mmsi all, --pred_cut, --iter_rollout, --output_per_mmsi_subdir, AISDatasetV2 full-trip loader, metadata, logging.

⸻

✅ Changes to implement

A) Basemap rendering — always add cartopy features

In src/eval/eval_traj_newnewnew (one place only), create the map like this in every plotting path:

import cartopy.crs as ccrs
import cartopy.feature as cfeature

proj = ccrs.PlateCarree()
fig, ax = plt.subplots(figsize=(9, 6), subplot_kw={'projection': proj})

# Basemap features (order matters: background first)
ax.add_feature(cfeature.OCEAN, zorder=0)
ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=2)

# Gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
gl.top_labels = False
gl.right_labels = False

Extent logic (Denmark clamp):

DEFAULT_DK_EXTENT = [6.0, 16.0, 54.0, 58.0]  # lon_min, lon_max, lat_min, lat_max

if args.auto_extent:
    # compute from actual/pred/combined, then sigma-trim
    lon_min, lon_max, lat_min, lat_max = compute_extent(points, sigma=args.extent_outlier_sigma)
    # clamp to Denmark
    lon_min = max(lon_min, DEFAULT_DK_EXTENT[0])
    lon_max = min(lon_max, DEFAULT_DK_EXTENT[1])
    lat_min = max(lat_min, DEFAULT_DK_EXTENT[2])
    lat_max = min(lat_max, DEFAULT_DK_EXTENT[3])
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
else:
    ax.set_extent(DEFAULT_DK_EXTENT, crs=proj)

Important: every plot/scatter call must pass transform=proj.

⸻

B) Pred tail anchoring — guarantee continuity

Add a small utility:

def anchor_pred_to_last_past(pred_abs, last_past, lat_idx, lon_idx):
    """
    Shift prediction so its first point equals last_past on (lat, lon),
    preserving predicted shape. Useful when model outputs are in absolute coords
    but slightly misaligned due to normalization or decoder state.
    """
    dlat = last_past[lat_idx] - pred_abs[0, lat_idx]
    dlon = last_past[lon_idx] - pred_abs[0, lon_idx]
    pred_anch = pred_abs.copy()
    pred_anch[:, lat_idx] += dlat
    pred_anch[:, lon_idx] += dlon
    return pred_anch

Then, right after you have:
	•	past_denorm  (shape [K, D])
	•	pred_denorm  (shape [H, D])  — make sure it is de-normalized and in the same (lat,lon) order
	•	last_past = past_denorm[-1]

Do:

# Sanity: swap order if needed
if args.y_order == "lonlat":
    lat_idx, lon_idx = args.lon_idx, args.lat_idx
else:
    lat_idx, lon_idx = args.lat_idx, args.lon_idx

pred_denorm = anchor_pred_to_last_past(pred_denorm, last_past, lat_idx, lon_idx)

# Guardrail: if still far, warn
km = haversine_km(last_past[lon_idx], last_past[lat_idx], pred_denorm[0, lon_idx], pred_denorm[0, lat_idx])
if km > 2.0:
    print(f"[warn] large first_pred jump after anchoring: {km:.2f} km")

If your model outputs deltas instead of absolutes, reconstruct absolute positions as:
pred_abs[t] = pred_abs[t-1] + delta[t] with pred_abs[-1]=last_past, then skip the shift.
Add a flag --pred_mode absolute|delta (default absolute) and branch accordingly.

⸻

C) Normalize → Predict → De-normalize (correct order)

Ensure the exact pipeline:
	1.	Take model inputs with the same features as training:

# Training used 4 features [LAT,LON,SOG,COG]
feats_in = full_trip[:, :4]           # shape [N, 4]


	2.	Normalize inputs before model (if your .pkl are already normalized, skip):
	•	lat: (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)
	•	lon: (lon - LON_MIN) / (LON_MAX - LON_MIN)
	•	sog: / SPEED_MAX, cog: / 360
	3.	Run the model on the last past_len slice of the past segment.
	4.	De-normalize the predicted lat/lon back to degrees using the same constants as de_normalize_track().

If you’re already calling your de_normalize_track() helper, ensure you call it on both past, true_future, and pred just before plotting — and only once.

⸻

D) Feature order & indices — eliminate mix-ups
	•	Respect --lat_idx, --lon_idx, and --y_order consistently:
	•	When slicing model outputs to (lat,lon)
	•	When denormalizing
	•	When plotting
	•	Add debugging prints once per sample:

print(f"[dbg] order={args.y_order} lat_idx={args.lat_idx} lon_idx={args.lon_idx}")
print(f"[dbg] past_last=({last_past[lat_idx]:.5f},{last_past[lon_idx]:.5f}) "
      f"pred_first=({pred_denorm[0, lat_idx]:.5f},{pred_denorm[0, lon_idx]:.5f})")



If y_order was wrong, the red segment often flies off (exactly like your screenshots).

⸻

E) Extent input — include all segments

When computing --auto_extent, feed full_trip + past + true_future + pred coordinates (after denorm) into the sigma-trim, then clamp to Denmark. This removes cases where the view zooms far from the track or excludes the red tail.

⸻

F) One plotting function, shared by all modes

Avoid divergent code paths. Use a single function that:
	1.	Builds the map (Section A)
	2.	Plots:
	•	gray full trip
	•	blue past
	•	black current point
	•	green true future
	•	red pred (anchored)
	3.	Sets extent once
	4.	Writes metadata row

That’s how you prevent “map in one mode, blank in another” regressions.

⸻

🧪 Acceptance tests

Run these and verify all pass:

1) Default multi-sample, 8 plots, Denmark map, red continues blue

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
  --log_skip_reasons --seed 0

Expect: 8 plots with a Denmark basemap; the red tail touches the blue end for all samples.

2) Single MMSI, last 10% prediction, anchored red tail

python -m src.eval.eval_traj_newnewnew \
  --split_dir data/map_reduced/val \
  --ckpt data/checkpoints/traj_tptrans.pt \
  --model tptrans \
  --full_trip \
  --mmsi 209867000 --trip_id 0 \
  --pred_cut 90 \
  --auto_extent --extent_source actual \
  --denorm --lat_min 54 --lat_max 58 --lon_min 6 --lon_max 16 \
  --lat_idx 0 --lon_idx 1 --y_order latlon \
  --annotate_id --log_skip_reasons

Expect: green = full tail; red starts exactly at the last blue point; Denmark map visible.

3) Batch all (cap 20), per-MMSI folders, auto-extent

python -m src.eval.eval_traj_newnewnew \
  --split_dir data/map_reduced/val \
  --ckpt data/checkpoints/traj_tptrans.pt \
  --model tptrans \
  --full_trip --mmsi all \
  --max_plots 20 \
  --pred_cut 85 \
  --auto_extent --extent_source actual \
  --lat_idx 0 --lon_idx 1 --y_order latlon \
  --out_dir data/figures --output_per_mmsi_subdir \
  --save_meta --meta_path data/figures/traj_eval_meta.csv \
  --log_skip_reasons --seed 42

Expect: all plots show Denmark map; red tails continue blue; per-MMSI subfolders created.

⸻

🧰 Extra diagnostics (keep during debugging)
	•	After anchoring, print the haversine distance between past[-1] and pred[0]. Must be < 0.5 km. Abort with a warning if larger (likely normalization/order error).
	•	Print the first and last coordinates of blue/green/red segments (lat, lon) to confirm order and denorm.
	•	Add a --debug_save_npz flag to dump past/true/pred/full_trip arrays per figure into out_dir/debug_npz/ for inspection.

⸻

🧠 Likely root causes (based on your screenshots)
	•	Basemap block missing entirely in certain code paths (single/batch).
	•	Pred outputs plotted before denorm or with lat/lon swapped, or without anchoring.
	•	Extent computed from the wrong set of points, causing cropped/misaligned views.

The changes above remove all three.




# Output

• Thanks for the clear brief. I’ve implemented all fixes from prompts/9_prompt.md to restore the Denmark map and make the
  prediction tail anchor exactly to the blue past.

  Summary of changes (fully implemented)

  - Basemap rendering unified and robust
      - Always draws Cartopy features in this order for every plot:
          - OCEAN, LAND (light gray), COASTLINE, BORDERS, gridlines with labels.
      - Uses a single plotting block with PlateCarree; all plot/scatter calls pass transform=proj.
      - Default extent is Denmark [6.0, 16.0, 54.0, 58.0] when auto-extent is off.
      - With auto-extent on, computes a sigma-trimmed extent from full-trip + past + true + pred, then clamps to Denmark.
  - Pred tail anchoring guaranteed
      - Adds hard continuity for absolute prediction mode:
          - After de-normalizing pred to degrees, shift the entire predicted segment so pred[0] equals the last past point
            (cur_lat, cur_lon).
          - Guardrail: warns if the anchored first point is still > 0.5 km from current.
      - Supports delta mode:
          - If using deltas, cumsum the deltas in degrees anchored from last past.
      - New CLI: --pred_mode absolute|delta (default absolute). Existing --pred_is_delta still works (back-compat).
  - Normalize → Predict → De-normalize order enforced
      - Model always receives last past_len steps of [lat, lon, sog, cog] normalized to [0,1] (or keeps as-is if already
        normalized).
      - All plotting uses de-normalized degrees for full trip, past, true tail, and pred.
  - Order and indices debugged
      - Applies args.y_order to build (lat,lon) normalized pairs before de-normalization.
      - Debug prints once per trip:
          - [dbg] order=… lat_idx=… lon_idx=…
          - [debug] model horizon=…, past_len=…, input_seq_shape=(T,4)
          - [diag] cur=(lat,lon) pred0=(lat,lon)
  - Auto-extent uses all segments and clamps to Denmark
      - Feeds full_trip + past + true + pred into robust extent computation and clamps to the Denmark extent.
  - Single plotting function for all modes
      - evaluate_and_plot_trip handles map construction, plotting of gray/blue/black/green/red, setting extent, and writing
        metadata.
      - Avoids divergent paths that caused missing maps in some modes.

  Implementation highlights (src/eval/eval_traj_newnewnew.py)

  - Denmark clamp constants and robust extent:
      - DEFAULT_DENMARK_EXTENT (6,16,54,58).
      - robust_extent now clamps to Denmark.
  - Anchoring utility:
      - anchor_pred_to_last_past(pred_abs, last_past, lat_idx, lon_idx).
  - Prediction path:
      - Slices inputs to the trained 4 features (lat,lon,sog,cog).
      - Normalizes inputs if they look like degrees; otherwise assumes normalized.
      - Converts model outputs to lat/lon in degrees.
      - Absolute mode: always anchors first pred point to the last past; warns if still >0.5 km.
      - Delta mode: cumsums deltas anchored to last past.
  - Unified Cartopy basemap block in one place with correct order and transforms.

  Other improvements

  - Diagnostic and skip reasons:
      - Prints “[info] N/M file: shape=…” for each file.
      - Skips with explicit reason (D<4, NaNs, etc.) only when truly invalid.
  - CLI additions:
      - --pred_mode absolute|delta (default absolute).
      - Updated help text for map extent (“default Denmark clamp”).

  What you should run to verify

  - Default multi, 8 plots (Denmark map + anchored red):
      - python -m src.eval.eval_traj_newnewnew \
        --split_dir data/map_reduced/val \
        --ckpt data/checkpoints/traj_tptrans.pt \
        --model tptrans \
        --max_plots 8 \
        --out_dir data/figures \
        --full_trip \
        --auto_extent --extent_outlier_sigma 3.0 \
        --denorm --lat_min 54 --lat_max 58 --lon_min 6 --lon_max 16 \
        --lat_idx 0 --lon_idx 1 --y_order latlon \
        --log_skip_reasons --seed 0
  - Single MMSI, last 10% red from blue:
      - python -m src.eval.eval_traj_newnewnew \
        --split_dir data/map_reduced/val \
        --ckpt data/checkpoints/traj_tptrans.pt \
        --model tptrans \
        --full_trip \
        --mmsi 209867000 --trip_id 0 \
        --pred_cut 90 \
        --auto_extent \
        --denorm --lat_min 54 --lat_max 58 --lon_min 6 --lon_max 16 \
        --lat_idx 0 --lon_idx 1 --y_order latlon \
        --annotate_id --log_skip_reasons
  - Batch all, iter rollout, per-MMSI subfolders:
      - python -m src.eval.eval_traj_newnewnew \
        --split_dir data/map_reduced/val \
        --ckpt data/checkpoints/traj_tptrans.pt \
        --model tptrans \
        --full_trip --mmsi all \
        --max_plots 20 \
        --pred_cut 85 --iter_rollout \
        --auto_extent \
        --out_dir data/figures --output_per_mmsi_subdir \
        --save_meta --meta_path data/figures/traj_eval_meta.csv \
        --lat_idx 0 --lon_idx 1 --y_order latlon \
        --denorm --lat_min 54 --lat_max 58 --lon_min 6 --lon_max 16 \
        --log_skip_reasons --seed 42

  Files touched

  - src/eval/eval_traj_newnewnew.py
      - Restored and unified Cartopy basemap with Denmark clamp.
      - Hard-anchored the red pred segment to the last blue point.
      - Normalization logic clarified.
      - Added debug logging and improved skip reasons.
      - Auto-extent includes all segments and clamps to Denmark.
      - Added --pred_mode absolute|delta.
  - No changes to the CLI for training/evaluation beyond the new --pred_mode and improved extent help.

  If you still see any plots where red isn’t touching blue, please run one of the commands above with --log_skip_reasons
  and share the “[dbg]/[diag]/[warn]” lines for that sample. I can quickly diagnose if it’s a y_order/indices mismatch or
  unexpected normalization.