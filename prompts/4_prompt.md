🚑 Follow-up: Predicted line missing / not continuing from last past point

What I run (two cases)

# A) Auto-zoomed Denmark view (zoom is correct, red pred line missing)
python -m src.eval.eval_traj_newnewnew \
  --split_dir data/map_reduced/val \
  --ckpt data/checkpoints/traj_tptrans.pt \
  --model tptrans \
  --lat_idx 0 --lon_idx 1 \
  --y_order latlon \
  --past_len 64 --max_plots 8 \
  --out_dir data/figures \
  --auto_extent --extent_source actual --extent_outlier_sigma 3.0

# B) Full Europe view (map OK, red pred line still missing)
python -m src.eval.eval_traj_newnewnew \
  --split_dir data/map_reduced/val \
  --ckpt data/checkpoints/traj_tptrans.pt \
  --model tptrans \
  --lat_idx 0 --lon_idx 1 \
  --y_order latlon \
  --past_len 64 --max_plots 8 \
  --out_dir data/figures

What I see (text description of the figures I attached)
	•	For Denmark auto-zoom: blue “past (input)” and green “true future” look correct; red dashed “pred future” is not visible at all (not even a dot).
	•	For Europe view: same—pred future not visible.
	•	Before these changes, I previously saw predictions far south (near the equator), but now after fixing normalization/CRS, the pred line seems gone or plotted out-of-frame.

Important requirement: the predicted trajectory must continue from the very last blue past point (the current position) and be on the same scale (degrees). Almost all data are around Denmark (≈ 54–57°N, 8–14°E).

⸻

Suspected root causes to check (and fix)
	1.	Wrong channel mapping for the model output (y-order vs. lat_idx/lon_idx).
	•	We pass --y_order latlon, --lat_idx 0, --lon_idx 1. Ensure the code uses these consistently when unpacking the model’s y_pred.
	•	No implicit swapping: x=lon, y=lat in plotting.
	2.	Prediction is relative (deltas) but plotted as absolute (or vice versa).
	•	If the model outputs Δlat/Δlon per step, we must re-anchor predictions to the last past point:

pred_abs[t] = last_past + cumsum(pred_deltas[:t])


	•	If the model outputs absolute normalized values, we must de-normalize once, not twice.

	3.	NaNs / all identical values / too few points after slicing.
	•	If we accidentally slice a wrong axis, y_pred_lat/y_pred_lon could be all zeros/NaNs, which renders nothing.
	4.	Masking / teacher forcing length mismatch.
	•	Confirm we take exactly the future horizon the same as the true target length plotted in green.

⸻

Please implement the following in src.eval.eval_traj_newnewnew

A) Explicit output mapping

Add a small helper so there is no ambiguity:

def split_lat_lon(arr, lat_idx, lon_idx):
    # arr shape: [T, C] where C >= max(lat_idx, lon_idx)+1
    lats = arr[:, lat_idx]
    lons = arr[:, lon_idx]
    return lats, lons

When --y_order is provided, ensure lat_idx/lon_idx are derived from it once, and reused everywhere (past, true, pred). For --y_order latlon: lat_idx=0, lon_idx=1.

B) Prediction mode & anchoring

Add two flags (defaults shown):

--pred_is_delta false
--anchor_pred true

Implement:

# last known point from past (current position)
cur_lat = past[:, args.lat_idx][-1]
cur_lon = past[:, args.lon_idx][-1]

# y_pred_raw: shape [T_future, C] from model
y_pred = y_pred_raw.copy()

# Optional de-normalization (same heuristic & function as actual)
y_pred = maybe_denorm(y_pred, args.lat_idx, args.lon_idx, name="pred")

pred_lat, pred_lon = split_lat_lon(y_pred, args.lat_idx, args.lon_idx)

if args.pred_is_delta:
    # Ensure deltas are in degrees (if normalized deltas, de-normalize deltas on lat/lon channels first)
    # Re-anchor to continue from last past point
    pred_lat = cur_lat + np.cumsum(pred_lat)
    pred_lon = cur_lon + np.cumsum(pred_lon)
else:
    # Absolute: ensure the first predicted point starts ~at current pos
    # If it doesn't (e.g., normalized to another frame), re-anchor if --anchor_pred
    d0_km = haversine_km(cur_lat, cur_lon, pred_lat[0], pred_lon[0])
    if args.anchor_pred and np.isfinite(d0_km) and d0_km > 5.0:
        dlat = pred_lat - pred_lat[0]
        dlon = pred_lon - pred_lon[0]
        pred_lat = cur_lat + dlat
        pred_lon = cur_lon + dlon
        print(f"[anchor] Shifted absolute predictions to start at current pos (Δ≈{d0_km:.1f} km).")

Add a small haversine:

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p = np.radians([lat1, lon1, lat2, lon2])
    dlat = p[2] - p[0]; dlon = p[3] - p[1]
    a = np.sin(dlat/2)**2 + np.cos(p[0])*np.cos(p[2])*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

C) Plot predictions after anchoring, with diagnostics

Right before plotting:

print(f"[diag] cur=({cur_lat:.5f},{cur_lon:.5f}) "
      f"pred[0]=({pred_lat[0]:.5f},{pred_lon[0]:.5f}) "
      f"pred[min/max lat]=({pred_lat.min():.2f},{pred_lat.max():..2f}) "
      f"pred[min/max lon]=({pred_lon.min():.2f},{pred_lon.max():.2f}) "
      f"n_pred={len(pred_lat)}")

# Sanity: first pred point should be ~current (<= 5 km) after anchoring
if haversine_km(cur_lat, cur_lon, pred_lat[0], pred_lon[0]) > 5.0:
    print("[warn] first predicted point still far from current position.")

# Now plot
ax.plot(pred_lon, pred_lat, '--', linewidth=1.6, transform=proj, label='pred future', zorder=5)

D) Extent should not hide predictions

Keep your improved auto-extent, but when --extent_source actual, still ensure predictions that fall within a 3× padded box around the actual track remain visible. If not, expand the extent slightly until both are visible (up to Europe clamp).

ext = compute_extent_from_actual(...)
if not any_point_in_extent(pred_lat, pred_lon, ext):
    ext = enlarge_extent(ext, factor=1.5, clamp=EUROPE)
ax.set_extent(ext, crs=proj)

E) Hard checks to catch the “invisible line” case
	•	If np.allclose(pred_lat, pred_lat[0]) and np.allclose(pred_lon, pred_lon[0]) → warn “flat prediction (single point)”.
	•	If any np.isnan(pred_lat) | np.isnan(pred_lon) → warn and skip NaNs before plotting.
	•	If line is zero-length after filtering → print explicit message: “no valid predicted points to plot”.

⸻

Acceptance criteria
	•	With the first command (auto-zoom on actual), the red dashed predicted line starts exactly at the last blue dot and continues from there.
	•	With the Europe command, the red dashed line is visible and in Denmark—not near the equator.
	•	Logs print: whether de-normalization was applied to pred, whether anchoring was applied, first-point distance (km), and min/max lat/lon of predictions.
	•	No CRS/transform issues; features render at all zooms.

⸻

If you need more from me

I can provide:
	•	The model’s output definition (absolute vs. delta).
	•	The normalization constants used in training.
	•	A small dump of one batch (past, y_true, y_pred) to reproduce the figure.

Please implement the steps above so the predicted trajectory is always stitched to the last past point and plotted in degrees at the correct location.



# Output
• I’ve read prompts/4_prompt.md and implemented the requested fixes so that predictions stitch from the last past point, always plot in degrees,
  and remain visible in both auto-zoom and Europe views.

  Plan

  - Clarify channel order and reuse consistently: introduce a helper to extract lat/lon channels, and use a single source of truth for lat_idx/
    lon_idx and y_order.
  - Handle model prediction mode robustly:
      - If outputs are deltas → cumsum + anchor to last past point (in degrees).
      - If outputs are absolute → de-normalize and optionally anchor first point to current position if far.
  - De-normalize safely using the pipeline’s function:
      - Use src.preprocessing.preprocessing.de_normalize_track when predictions/actuals are normalized.
      - Add delta-specific de-normalization (scale by range only; no shifting).
  - Make auto-extent robust and inclusive:
      - Sigma-clip outliers, pad, and clamp to Europe.
      - When extent_source=actual, enlarge slightly if predictions lie just outside the extent.
  - Diagnostics and sanity checks:
      - Print whether (de)norm happened, first-point distance from current, prediction min/max, flat/NaN warnings.
      - Ensure CRS transforms are correct when Cartopy is present; fall back to GeoPandas/static Matplotlib otherwise.

  Key changes

  - src/eval/eval_traj_newnewnew.py
      - Added helpers:
          - split_lat_lon(arr, lat_idx, lon_idx)
          - haversine_km(...)
          - robust_extent(...)
          - maybe_denorm(...) using preprocessing’s de_normalize_track
          - maybe_denorm_deltas(...) that scales by [min,max] ranges without shifting
      - New CLI flags:
          - --pred_is_delta (default false)
          - --anchor_pred (default true; --no_anchor_pred to disable)
          - --extent_source {both,actual,pred} (default actual)
          - --extent_outlier_sigma (default 3.0)
      - Inference:
          - If --pred_is_delta, convert model output to absolute by cumsum + last past point; otherwise treat outputs as absolute.
          - For ADE/FDE, align prediction order to dataset Y order (metric arrays are latlon).
      - Plotting:
          - Build actual (past + true) and predicted tracks using the same lat_idx/lon_idx and y_order.
          - Use maybe_denorm/maybe_denorm_deltas for actual/pred depending on prediction mode.
          - If absolute predictions and first predicted point is far from current (> 5 km) and --anchor_pred, shift whole pred sequence to start
            at current.
          - Extensive diagnostics printouts (medians, first-point distance, min/max, flat/NaN).
          - Auto-extent now sigma-clipped and, when extent_source=actual, expands slightly if pred points fall outside the computed extent.
          - Cartopy PlateCarree transforms if available; otherwise basemap/axes with xlim/ylim.

  How to run (matching your dataset)

  - Your data uses [lat, lon, sog, cog] with lat/lon normalized to [0,1] during preprocessing, and target Y is [lat, lon]. The model was trained
    to predict absolute normalized lat/lon (not deltas). Use:
      - auto-zoom on actual (recommended):
        python -m src.eval.eval_traj_newnewnew \
        --split_dir data/map_reduced/val \
        --ckpt data/checkpoints/traj_tptrans.pt \
        --model tptrans \
        --lat_idx 0 --lon_idx 1 \
        --y_order latlon \
        --past_len 64 --max_plots 8 \
        --out_dir data/figures \
        --auto_extent --extent_source actual --extent_outlier_sigma 3.0 \
        --denorm --lat_min 54 --lat_max 58 --lon_min 6 --lon_max 16
      - full Europe view:
        python -m src.eval.eval_traj_newnewnew \
        --split_dir data/map_reduced/val \
        --ckpt data/checkpoints/traj_tptrans.pt \
        --model tptrans \
        --lat_idx 0 --lon_idx 1 \
        --y_order latlon \
        --past_len 64 --max_plots 8 \
        --out_dir data/figures \
        --denorm --lat_min 54 --lat_max 58 --lon_min 6 --lon_max 16

  Notes

  - If your model actually outputs deltas, add --pred_is_delta. The script will cumsum and anchor automatically.
  - With --anchor_pred (default), absolute predictions that don’t start at the current position will be shifted to start exactly at the last past
    point, so the red line continues from the blue dot.
  - The logs now print:
      - Whether de-normalization was applied to actual/pred
      - First predicted point distance from current (km) and if anchoring was applied
      - Prediction min/max lat/lon, medians, flat/NaN warnings
      - Selected extent and whether it was enlarged to include pred

  If issues persist

  - If you still don’t see the red line:
      - Please run with --pred_is_delta toggled off/on and paste a snippet of the console logs (the “[diag]” and “[anchor]” lines), so I can see
        whether deltas/absolutes are being handled correctly.
      - If your normalization bounds differ from 54–58 and 6–16, pass the correct --lat_min/--lat_max/--lon_min/--lon_max.
      - If possible, share one small batch (x, y_true, y_pred) or confirm definitively whether the model output is absolute or delta.

  This should resolve the “pred line missing” and ensure predictions continue from the last past point in the Denmark area for both auto-zoom and Europe plots.