AIS Trajectory Evaluation (Full-Trip, Denmark Clamp)

Overview
- Evaluates and visualizes full AIS trips stored as `<MMSI>_<TRIP_ID>_processed.pkl`.
- Always plots the entire trip (gray), the past/context (blue), the true future tail (green), and the predicted continuation (red).
- Default map clamp is Denmark `[6, 16, 54, 58]` (lon_min, lon_max, lat_min, lat_max). Auto-extent is sigma-trimmed and clamped within Denmark.
- Supports: single MMSI, batch all, or a random multi-sample subset. Dry-run listing, per‑MMSI subfolders, metadata CSV, and iterative rollout.

Core Commands
- Default multi-sample (8 full-trip plots):
  - `python -m src.eval.eval_traj_newnewnew \
    --split_dir data/map_reduced/val \
    --ckpt data/checkpoints/traj_tptrans.pt \
    --model tptrans \
    --max_plots 8 \
    --out_dir data/figures \
    --auto_extent --extent_outlier_sigma 3.0 \
    --denorm --lat_min 54 --lat_max 58 --lon_min 6 --lon_max 16 \
    --log_skip_reasons --seed 0`

- Single MMSI (last 10% prediction):
  - `python -m src.eval.eval_traj_newnewnew \
    --split_dir data/map_reduced/val \
    --ckpt data/checkpoints/traj_tptrans.pt \
    --model tptrans \
    --mmsi 209867000 --trip_id 0 \
    --pred_cut 90 \
    --auto_extent \
    --lat_idx 0 --lon_idx 1 --y_order latlon \
    --denorm --lat_min 54 --lat_max 58 --lon_min 6 --lon_max 16 \
    --annotate_id --log_skip_reasons`

- Batch all MMSIs, 20 plots max, iterative rollout, per-MMSI folders:
  - `python -m src.eval.eval_traj_newnewnew \
    --split_dir data/map_reduced/val \
    --ckpt data/checkpoints/traj_tptrans.pt \
    --model tptrans \
    --mmsi all --max_plots 20 \
    --pred_cut 85 --iter_rollout \
    --auto_extent \
    --out_dir data/figures \
    --output_per_mmsi_subdir \
    --save_meta --meta_path data/figures/traj_eval_meta.csv \
    --log_skip_reasons --seed 42`

Plot Semantics
- Full context (gray): the entire trip, from first to last timestamp.
- Past/context (blue): from trip start up to the cut index (see `--pred_cut`); if not set, past ends at `len(trip) - horizon`.
- True future (green): the entire tail from the cut index to the end of the trip.
- Predicted future (red): model output starting at the last blue point.
  - Truncates to `min(model.horizon, --cap_future)` by default.
  - With `--iter_rollout`, iteratively predicts forward (feeding prior predictions back as input) to cover the whole tail or `--cap_future` (whichever is smaller).
- Current position (black dot): the last past point.

Map Rendering
- Uses Cartopy PlateCarree (with LAND/OCEAN/COASTLINE/BORDERS) if available; otherwise falls back to Matplotlib with extent in degrees.
- Default extent (without `--auto_extent`) is Denmark `[6, 16, 54, 58]`.
- With `--auto_extent`, computes sigma-trimmed bounds from past + true + predicted, then clamps to the Denmark box.

Inputs and Normalization
- Trips are expected to be normalized by preprocessing to approximately `[0..1]` for lat/lon within Denmark. Use `--denorm` with `--lat_min/--lat_max/--lon_min/--lon_max` to convert back to degrees for plotting.
- Feature layout is `[lat, lon, sog, cog, ... timestamp, mmsi]`. Set `--lat_idx/--lon_idx` appropriately (defaults to `0, 1`).
- If the model predicts deltas (Δlat/Δlon), set `--pred_is_delta` to anchor+cumsum from the last past point; otherwise predictions are treated as absolute and optionally re-anchored if the first predicted point is far from current.

Arguments (Most Useful)
- `--split_dir` (path): directory containing `<MMSI>_<TRIP_ID>_processed.pkl` files.
- `--ckpt` (path): model checkpoint (required unless `--list_only`).
- `--model` (str): `tptrans` or `gru`.
- `--mmsi` (int | `all` | omit): single MMSI, batch all, or default random multi-sample.
- `--trip_id` (int): trip index for single-MMSI mode.
- `--max_plots` (int): maximum number of plots (caps selection in multi/batch modes).
- `--pred_cut` (float): past percentage; if omitted, past ends at `len(trip) - horizon`.
- `--cap_future` (int): cap predicted steps.
- `--iter_rollout` (flag): iteratively roll predictions to match tail length/cap.
- `--auto_extent` (flag): auto-fit extent; otherwise uses Denmark default clamp.
- `--extent_outlier_sigma` (float): sigma for outlier trimming in auto-extent.
- `--denorm` (flag) + `--lat_min/--lat_max/--lon_min/--lon_max` (float): convert normalized to degrees.
- `--lat_idx/--lon_idx` (int): column indices of `lat`/`lon`.
- `--y_order` (str): `latlon` or `lonlat` ordering for the model outputs.
- `--pred_is_delta` (flag): treat outputs as per-step deltas.
- `--anchor_pred`/`--no_anchor_pred`: anchor absolute predictions to start at current point if needed.
- `--output_per_mmsi_subdir` (flag): save under `data/figures/<MMSI>/` with a local CSV.
- `--save_meta` (flag), `--meta_path` (path): global metadata CSV.
- `--list_only` (flag): print selected files and exit (no ckpt needed).
- `--log_skip_reasons` (flag): print skip reasons per trip.
- `--seed` (int): random seed for selection.

Outputs
- Figures: PNGs showing gray (context), blue (past), green (true tail), red (pred). If `--output_per_mmsi_subdir` is set, PNGs are saved to `data/figures/<MMSI>/`; otherwise into `data/figures/`.
- Filenames include model, MMSI, trip_id, and cut: `traj_<model>_mmsi-<id>_trip-<tid>_cut-<p>_idx-<k>.png`.
- Metadata CSV rows (global and/or per-MMSI):
  - `sample_idx, model, mmsi, trip_id, mode, pred_cut, n_total, n_past, n_true_future, n_pred, t_start_iso, t_end_iso, lat_min, lat_max, lon_min, lon_max, out_path`

Troubleshooting
- If predictions appear offset, ensure `--denorm` bounds match preprocessing and that `--y_order` and `--pred_is_delta` reflect your model.
- If Cartopy is missing, install it or rely on the Matplotlib fallback.
- Use `--list_only` to verify selected files.

