🔖 Follow-up: Identify each plotted sample (MMSI + time range) and export metadata

What works now
	•	Denmark auto-zoom, CRS, de-norm, and “pred continues from last past point.”

What I need next

Add clear identification for every plotted sample so I can tell which ship (MMSI) and from which timestamp to which timestamp the plot covers. Also clarify whether the blue “past (input)” line is the full trip or just a window.

Data columns (order)

LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI

TIMESTAMP is Unix seconds (assume UTC). MMSI is constant for a track/window.

⸻

Please implement in src.eval.eval_traj_newnewnew

1) Extract MMSI + time span per sample
	•	From the combined arrays you already have (past input, current point, true future, predicted future), compute:
	•	mmsi: mode/unique value of the MMSI column across the entire sample window
	•	t_start: min of TIMESTAMP across (past + true future)
	•	t_end: max of TIMESTAMP across (past + true future)
	•	Convert to human-readable UTC ISO strings

def to_iso(ts):
    # ts can be float or int (seconds since epoch)
    import datetime as dt
    return dt.datetime.utcfromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S UTC")

def extract_id_and_span(arrays, idx_timestamp=7, idx_mmsi=8):
    """
    arrays: list of np.ndarray with same column layout (e.g., [past, true_future])
    Returns: mmsi(str), t_start(float), t_end(float)
    """
    import numpy as np
    all_stacks = np.concatenate([a for a in arrays if a is not None and a.size > 0], axis=0)

    ts = all_stacks[:, idx_timestamp]
    mmsi_col = all_stacks[:, idx_mmsi]
    # MMSI as the most frequent value (robust in case of noise)
    vals, counts = np.unique(mmsi_col.astype(np.int64), return_counts=True)
    mmsi = int(vals[np.argmax(counts)])

    t_start = float(np.nanmin(ts))
    t_end   = float(np.nanmax(ts))
    return mmsi, t_start, t_end

2) Put identification into title, subtitle, filename, and a CSV
	•	Add new CLI flags (defaults shown):

--stamp_titles true
--stamp_filename true
--save_meta true
--meta_path data/figures/traj_eval_meta.csv
--timefmt "%Y-%m-%d %H:%M:%S UTC"

	•	Build a descriptive title and filename:
	•	Title example:
Trajectory sample 3 (tptrans) — MMSI 219012345 — 2024-04-03 11:20:00 UTC → 2024-04-03 12:10:00 UTC
	•	Filename example:
traj_tptrans_mmsi-219012345_2024-04-03T112000Z_2024-04-03T121000Z_idx-3.png
	•	Please sanitize characters for filenames (: → nothing, space → _, append Z for UTC).

mmsi, t0, t1 = extract_id_and_span([past_arr, true_future_arr], idx_timestamp=7, idx_mmsi=8)
t0_iso, t1_iso = to_iso(t0), to_iso(t1)

if args.stamp_titles:
    ax.set_title(
        f"Trajectory sample {sample_idx} ({args.model}) — MMSI {mmsi} — {t0_iso} → {t1_iso}"
    )

if args.stamp_filename:
    def fname_ts(ts):
        import datetime as dt
        return dt.datetime.utcfromtimestamp(float(ts)).strftime("%Y-%m-%dT%H%M%SZ")
    out_name = (
        f"traj_{args.model}_mmsi-{mmsi}_{fname_ts(t0)}_{fname_ts(t1)}_idx-{sample_idx}.png"
    )
else:
    out_name = f"traj_{args.model}_idx-{sample_idx}.png"

fig.savefig(os.path.join(args.out_dir, out_name), dpi=200, bbox_inches="tight")

	•	Append a row to a CSV metadata file for each saved plot (create header if file doesn’t exist):

# meta columns: sample_idx, model, mmsi, t_start, t_end, t_start_iso, t_end_iso,
#               n_past, n_true_future, n_pred, out_path, lat_lon_bounds
meta_row = {
    "sample_idx": sample_idx,
    "model": args.model,
    "mmsi": mmsi,
    "t_start": t0,
    "t_end": t1,
    "t_start_iso": t0_iso,
    "t_end_iso": t1_iso,
    "n_past": len(lats_past),
    "n_true_future": len(lats_true),
    "n_pred": len(pred_lat),
    "out_path": os.path.join(args.out_dir, out_name),
    "lat_min": float(min(np.nanmin(lats_all), np.nanmin(pred_lat))) if len(pred_lat) else float(np.nanmin(lats_all)),
    "lat_max": float(max(np.nanmax(lats_all), np.nanmax(pred_lat))) if len(pred_lat) else float(np.nanmax(lats_all)),
    "lon_min": float(min(np.nanmin(lons_all), np.nanmin(pred_lon))) if len(pred_lon) else float(np.nanmin(lons_all)),
    "lon_max": float(max(np.nanmax(lons_all), np.nanmax(pred_lon))) if len(pred_lon) else float(np.nanmax(lons_all)),
}

if args.save_meta:
    import csv, os
    os.makedirs(args.out_dir, exist_ok=True)
    meta_path = args.meta_path
    file_exists = os.path.exists(meta_path)
    with open(meta_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(meta_row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(meta_row)

3) Optional on-plot annotations (small text near current point)
	•	Add flag --annotate_id false. When true, draw a small label by the black “current pos” dot:

if args.annotate_id:
    ax.text(
        cur_lon, cur_lat,
        f"MMSI {mmsi}\n{t0_iso} → {t1_iso}",
        transform=proj,
        fontsize=8,
        ha="left", va="bottom",
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2),
        zorder=10,
    )

4) Clarify “why some blue lines are longer”
	•	Please log exactly what’s plotted:
	•	n_past = number of past points (your --past_len)
	•	n_true_future = number of ground-truth future steps
	•	n_pred = number of predicted steps
	•	Add a CLI note to the script’s help:
“Past is a window of length --past_len, not the full MMSI trip. To plot the full trip per MMSI, use --full_trip (see below).”

5) (Optional) Add a mode to plot the full trip for an MMSI

If feasible now, add:

--full_trip false
--mmsi_filter <int> (optional)
--max_hours 24

	•	When --full_trip true, fetch and plot all records for that MMSI within the same day (or within --max_hours) instead of a single window; keep the current window highlighted (e.g., thicker line) so context is visible.

⸻

Acceptance criteria
	•	Each saved PNG includes MMSI and time span in the title (and optionally near the current point).
	•	Each saved PNG filename includes mmsi, t_start, t_end, and sample_idx.
	•	A CSV traj_eval_meta.csv is written/appended with one row per figure (fields listed above).
	•	The console log prints n_past, n_true_future, n_pred for each sample and clearly states whether we plotted a window or a full trip.
	•	(If --full_trip implemented) Full-trip mode shows context lines plus the evaluated window emphasized.

⸻

If repo details differ

If timestamps aren’t Unix seconds or column indices differ, please adapt but preserve the identification outputs (title, filename, CSV) and the acceptance criteria.

Goal: every plot is self-describing (which ship + when), and I can filter/sort results later using the metadata CSV.



# Output

• Plan

  - Add identity metadata (MMSI + time span) per plotted sample.
  - Stamp identity into plot titles, filenames, and export a CSV row for each figure.
  - Provide optional on-plot annotation near the current position.
  - Log counts (n_past, n_true_future, n_pred) and clarify that “past” is a window.
  - Optional: overlay full-trip context for the sample file (and filter by MMSI).

  What I changed

  - Identity + time span
      - Extracts MMSI and [t_start, t_end] from original sample data using timestamp/MMSI columns (7, 8).
      - Adds helpers: to_iso, fname_ts, extract_id_and_span. src/eval/eval_traj_newnewnew.py:28–53
  - Title and filename stamping
      - Title: “Trajectory sample <idx> (tptrans) — MMSI <mmsi> — <t0_iso> → <t1_iso>”
      - Filename: traj_<model>_mmsi-<mmsi>_<t0Z>_<t1Z>_idx-<idx>.png
      - New CLI booleans with “no_” toggles: --stamp_titles/--no_stamp_titles, --stamp_filename/--no_stamp_filename. Defaults on.
  - Metadata CSV
      - Writes/appends a row per figure with fields: sample_idx, model, mmsi, t_start, t_end, t_start_iso, t_end_iso, n_past, n_true_future,
        n_pred, out_path, lat_min, lat_max, lon_min, lon_max.
      - New flags: --save_meta/--no_save_meta (default on), --meta_path, --timefmt.
  - On-plot annotation
      - New flag --annotate_id to draw MMSI and time span near the black current-point dot, with a small styled label.
  - Logging and help note
      - Prints counts per sample: n_past, n_true_future, n_pred, and whether we plotted ‘window’ vs ‘full_trip’.
      - Adds help text: “Past is a window of length --past_len, not the full MMSI trip. Use --full_trip for context.”
  - Full-trip context (optional)
      - New flags: --full_trip (overlay entire sample’s source-track), --mmsi_filter (optional int), --max_hours (advisory).
      - When enabled, plots the entire file’s track (grey) behind window tracks, and the window lines (past/true/pred) remain emphasized.

  Implementation details

  - Tracks dataset indices when selecting samples so the script can load the original *_processed.pkl to retrieve timestamps and MMSI. The
    plotting function now receives (ds_idx, X, Y_true, Y_pred) tuples.
  - Uses the same normalization/de-normalization logic as earlier fixes (safe denorm for actual and pred, delta-only scaling if --pred_is_delta).
  - Robust extent and Cartopy transforms remain intact.

  New/updated CLI options (defaults shown)

  - Identity/metadata:
      - --stamp_titles / --no_stamp_titles (default: true)
      - --stamp_filename / --no_stamp_filename (default: true)
      - --save_meta / --no_save_meta (default: true)
      - --meta_path data/figures/traj_eval_meta.csv
      - --timefmt "%Y-%m-%d %H:%M:%S UTC"
      - --annotate_id
  - Full trip:
      - --full_trip
      - --mmsi_filter <int>
      - --max_hours 24
  - Existing options (unchanged): --auto_extent, --extent_source, --extent_outlier_sigma, --pred_is_delta, --anchor_pred/--no_anchor_pred,
    --lat_idx/--lon_idx, --y_order, normalization bounds, etc.

  Examples

  - Auto-zoom with stamped ID and metadata (Denmark dataset assumptions):
      - Lat/lon normalized to [54–58], [6–16], Y is [lat,lon]
        python -m src.eval.eval_traj_newnewnew \
        --split_dir data/map_reduced/val \
        --ckpt data/checkpoints/traj_tptrans.pt \
        --model tptrans \
        --lat_idx 0 --lon_idx 1 \
        --y_order latlon \
        --past_len 64 --max_plots 8 \
        --out_dir data/figures \
        --auto_extent --extent_source actual --extent_outlier_sigma 3.0 \
        --denorm --lat_min 54 --lat_max 58 --lon_min 6 --lon_max 16 \
        --annotate_id --full_trip
  - Europe view with stamped filename only:
    python -m src.eval.eval_traj_newnewnew \
    --split_dir data/map_reduced/val \
    --ckpt data/checkpoints/traj_tptrans.pt \
    --model tptrans \
    --lat_idx 0 --lon_idx 1 \
    --y_order latlon \
    --past_len 64 --max_plots 8 \
    --out_dir data/figures \
    --denorm --lat_min 54 --lat_max 58 --lon_min 6 --lon_max 16 \
    --no_stamp_titles --stamp_filename

  Notes

  - “Past” is the window X (length --past_len), not the full track; enabling --full_trip overlays the entire sample’s source trajectory (from the
    same file) for context with the window emphasized by the colored lines.
  - If your model produces deltas, add --pred_is_delta so the plot stitches from the last past point correctly.
  - The metadata CSV accumulates all figure entries at --meta_path and includes bounds to help filter/sort later.