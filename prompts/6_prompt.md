🧭 Final Prompt — Full Trip Evaluation (--mmsi all), MMSI Subfolder Output, and Full Documentation

🎯 Objective

Extend src.eval.eval_traj_newnewnew to include:
	1.	✅ Three operation modes
	•	Default (multi-sample) — identical to existing behavior
	•	Single-MMSI full trip — evaluate one vessel trip
	•	Batch (--mmsi all) — evaluate all files in --split_dir
	2.	✅ Full-trip context rendering
	•	The gray path represents the entire trip
	•	Blue (past), green (true future), and red (predicted) are subsets of that same trip
	•	Red prediction starts exactly at the last blue point
	3.	✅ Dry-run mode (--list_only)
	4.	✅ Structured logging and skip reasons
	5.	✅ Metadata CSV output
	6.	✅ Optional per-MMSI output folders via --output_per_mmsi_subdir

⸻

⚙️ CLI Argument Reference

Argument	Type	Default	Description
--split_dir	path	required	Directory with .pkl trip files: <MMSI>_<TRIP_ID>_processed.pkl
--ckpt	path	required	Path to model checkpoint
--model	str	required	Model name (e.g. tptrans)
--lat_idx, --lon_idx	int	0, 1	Column indices for lat/lon
--y_order	str	latlon	Coordinate order expected by model
--past_len	int	64	Past timesteps used for model input
--max_plots	int	8	Max number of samples (multi or batch)
--out_dir	path	data/figures	Root folder for outputs
--auto_extent	flag	off	Fit map extent automatically
--extent_source	str	actual	Source for extent (`actual
--extent_outlier_sigma	float	3.0	Outlier removal for extent
--denorm	flag	off	Apply de-normalization
--lat_min, --lat_max, --lon_min, --lon_max	float	—	Bounds for denorm if set
--annotate_id	flag	off	Annotate MMSI and timestamps on plots
--seed	int	—	RNG seed for reproducibility
--log_skip_reasons	flag	off	Print reason for each skipped trip
--list_only	flag	off	Dry run: list selected files and exit
--save_meta	flag	on	Save metadata to CSV
--meta_path	path	data/figures/traj_eval_meta.csv	Metadata CSV path
--full_trip	flag	off	Draw entire trip in gray and evaluate within it
--mmsi	int / all / None	—	Select MMSI: omit for default mode, use all for batch, or specific ID for one trip
--trip_id	int	0	Trip index when --mmsi is numeric
--pred_cut	float	—	% of trip to treat as past before predicting tail
--cap_future	int	—	Cap predicted horizon
--min_points	int	30	Skip too-short trips
--output_per_mmsi_subdir	flag	off	New: save each MMSI’s outputs to --out_dir/<MMSI>/ (figures + metadata)


⸻

🧩 Mode Selection Logic

files = sorted(glob(os.path.join(args.split_dir, "*.pkl")))

def parse_trip(fname):
    base = os.path.basename(fname).replace("_processed.pkl","")
    mmsi_str, trip_id_str = base.split("_", 1)
    return int(mmsi_str), int(trip_id_str)

if args.mmsi is None:
    mode = "multi"
    rng = np.random.default_rng(args.seed)
    n_select = min(len(files), args.max_plots)
    idx = rng.choice(len(files), size=n_select, replace=False) if len(files) > n_select else range(n_select)
    files_to_eval = [files[i] for i in sorted(idx)]

elif isinstance(args.mmsi, str) and args.mmsi.lower() == "all":
    mode = "batch_all"
    files_to_eval = files[:args.max_plots] if args.max_plots else files

else:
    mode = "single"
    tid = args.trip_id or 0
    files_to_eval = [os.path.join(args.split_dir, f"{int(args.mmsi)}_{int(tid)}_processed.pkl")]

print(f"[mode] {mode}  found={len(files)}  selected={len(files_to_eval)}  max_plots={args.max_plots}")


⸻

🧾 List-Only Dry Run

if args.list_only:
    print("\n[files selected for evaluation]\n")
    for f in files_to_eval:
        m, t = parse_trip(f)
        print(f" - {os.path.basename(f)}  (MMSI={m}, trip_id={t})")
    print("\n[done: list_only mode, no plots generated]\n")
    sys.exit(0)


⸻

📦 Trip Loading + Preprocessing

def load_trip(path):
    with open(path, "rb") as f:
        trip = pickle.load(f)
    trip = np.asarray(trip)
    if len(trip) < args.min_points:
        raise ValueError(f"too short: {len(trip)} points")
    ts = trip[:, 7]  # TIMESTAMP
    if not np.all(ts[:-1] <= ts[1:]):
        trip = trip[np.argsort(ts)]
    return trip


⸻

✂️ Splitting (if --pred_cut)

def split_by_percent(trip, pct):
    n = len(trip)
    cut = max(1, min(n-2, int(round(n * pct / 100.0))))
    past = trip[:cut]
    future_true = trip[cut:]
    return past, future_true, cut

When --full_trip:
	•	Always plot the entire trip in gray first.
	•	Blue past = first <pred_cut>% (or sliding window if none).
	•	Green = true future segment.
	•	Red = predicted continuation (anchored to past[-1]).

⸻

🗺️ Full-Trip Context Rules

When --full_trip is active:

Segment	Meaning	Color / Style
Full context	Entire trip	gray, alpha=0.3, linewidth=1
Past (input)	Before cut or past_len	blue solid
True future	After cut	green solid
Predicted future	Model output	red dashed
Current pos	End of past	black dot

	•	All coordinates come from the same denormalized track.
	•	The red predicted path starts exactly at past[-1] and continues smoothly.
	•	Auto extent should cover all four segments.

⸻

🗃️ Per-MMSI Subfolder Output

When --output_per_mmsi_subdir is set:

if args.output_per_mmsi_subdir:
    mmsi_dir = os.path.join(args.out_dir, str(mmsi))
    os.makedirs(mmsi_dir, exist_ok=True)
    fig_path = os.path.join(mmsi_dir, f"traj_{args.model}_mmsi-{mmsi}_trip-{trip_id}_cut-{args.pred_cut or 'none'}_idx-{i}.png")
    meta_path = os.path.join(mmsi_dir, "traj_eval_meta.csv")
else:
    fig_path = os.path.join(args.out_dir, f"traj_{args.model}_mmsi-{mmsi}_trip-{trip_id}_cut-{args.pred_cut or 'none'}_idx-{i}.png")
    meta_path = args.meta_path

Each subfolder should contain:
	•	All generated .png plots for that MMSI
	•	A local metadata CSV (plus optionally appended to global meta_path)

⸻

🪵 Logging

ok, skipped = 0, 0
for i, f in enumerate(files_to_eval):
    try:
        trip = load_trip(f)
        mmsi, tid = parse_trip(f)
        if args.pred_cut:
            past, fut, cut = split_by_percent(trip, args.pred_cut)
        # evaluate_and_plot(...) includes full gray line rendering
        evaluate_and_plot(trip, past, fut, mmsi, tid, args)
        ok += 1
        print(f"[ok] {os.path.basename(f)} plotted ({len(trip)} pts)")
    except Exception as e:
        skipped += 1
        if args.log_skip_reasons:
            print(f"[skip] {os.path.basename(f)} reason={e}")
print(f"[summary] plotted={ok}, skipped={skipped}, total={len(files_to_eval)}")


⸻

📈 Metadata Fields

Column	Description
sample_idx	Plot index
model	Model name
mmsi	Vessel ID
trip_id	Trip index
mode	multi / single / batch_all
pred_cut	% of past used
n_total, n_past, n_true_future, n_pred	counts
t_start_iso, t_end_iso	Time range
lat_min, lat_max, lon_min, lon_max	Plot bounds
out_path	Saved figure path


⸻

🧪 Example Commands

1️⃣ Default multi-sample with full-trip context

python -m src.eval.eval_traj_newnewnew \
  --split_dir data/map_reduced/val \
  --ckpt data/checkpoints/traj_tptrans.pt \
  --model tptrans \
  --lat_idx 0 --lon_idx 1 --y_order latlon \
  --past_len 64 --max_plots 8 \
  --auto_extent --extent_source actual --extent_outlier_sigma 3.0 \
  --denorm --lat_min 54 --lat_max 58 --lon_min 6 --lon_max 16 \
  --annotate_id --full_trip \
  --out_dir data/figures \
  --log_skip_reasons --seed 0

2️⃣ Dry run (no plotting)

python -m src.eval.eval_traj_newnewnew \
  --split_dir data/map_reduced/val \
  --max_plots 10 \
  --list_only

3️⃣ Single MMSI, predict last 10%

python -m src.eval.eval_traj_newnewnew \
  --split_dir data/map_reduced/val \
  --ckpt data/checkpoints/traj_tptrans.pt \
  --model tptrans \
  --full_trip \
  --mmsi 209867000 --trip_id 0 \
  --pred_cut 90 --cap_future 60 \
  --auto_extent --extent_source actual \
  --out_dir data/figures \
  --annotate_id --log_skip_reasons

4️⃣ Batch all MMSIs (save to per-MMSI folders)

python -m src.eval.eval_traj_newnewnew \
  --split_dir data/map_reduced/val \
  --ckpt data/checkpoints/traj_tptrans.pt \
  --model tptrans \
  --lat_idx 0 --lon_idx 1 --y_order latlon \
  --past_len 64 --full_trip --mmsi all \
  --pred_cut 85 \
  --auto_extent --extent_source actual \
  --out_dir data/figures \
  --output_per_mmsi_subdir \
  --save_meta --meta_path data/figures/traj_eval_meta.csv \
  --log_skip_reasons --seed 42


⸻

✅ Acceptance Criteria

Requirement	Status
Default multi-sample works with --max_plots	✅
Single-MMSI mode works with optional --trip_id	✅
--mmsi all batch mode iterates all .pkl files	✅
--output_per_mmsi_subdir creates out_dir/<MMSI>/ folders	✅
Each folder contains plots + metadata	✅
Gray context path always covers full trip	✅
Blue/green/red lines are true subsets and aligned	✅
Red prediction starts at end of blue path	✅
--list_only prints selected files then exits	✅
Clear skip logging and summary	✅
One global metadata CSV (plus optional per-MMSI CSVs)	✅



After this is implemented fully and correctly, explain how to run it fully. 



# output:
• Planned the extension in three parts:

  - Modes and selection: add single-MMSI and batch-all modes alongside the default “multi-sample” window mode, plus dry-run listing, seedable
    selection, and skip-reason logging.
  - Full-trip evaluation + plotting: use trip-level arrays to split past/future (via pred_cut or past_len), run model once, anchor/denorm
    predictions, and render context + segments with robust extent.
  - Identity + outputs: stamp title/filename with MMSI and times, write metadata CSV (global and/or per-MMSI subfolder), and optional on-plot
    annotation.

  What I implemented

  - New modes
      - Default “multi” (kept exactly as before): uses AISDataset windows, computes ADE/FDE, and plots selected windows.
      - Single trip: --mmsi <id> --trip_id <n> evaluates one <MMSI>_<TRIP>_processed.pkl.
      - Batch all: --mmsi all evaluates all trips (or first --max_plots) in --split_dir.
      - Dry-run: --list_only lists selected files and exits without requiring --ckpt.
      - Structured logging: prints selected mode, file counts, OK/skip summary, and per-file skip reasons with --log_skip_reasons.
  - Trip-level evaluation and full context
      - Loads trips: [lat, lon, sog, cog, heading, rot, nav, timestamp, mmsi], sorted by timestamp, enforcing --min_points.
      - Splitting:
          - With --pred_cut: past is first X% of trip; future is the remainder.
          - Otherwise: past is first --past_len points; future is the remainder.
          - Caps number of predicted steps via --cap_future; always respects model’s horizon.
      - Model inference: feeds normalized [lat, lon, sog, cog]; handles model outputs as deltas or absolutes via --pred_is_delta.
      - Anchoring: ensures the red prediction starts exactly at the last blue point; applies anchoring for absolute predictions if the first
        predicted point is > 5 km from current; logs “[anchor] …”.
      - De-normalization:
          - Full trip and predictions convert back to degrees using preprocessing de_normalize_track and range-only scaling for deltas.
      - Plotting:
          - Gray full-trip context, blue past, green true future, red predicted continuation, black dot on current.
          - Robust extent (sigma-clipped) over all segments; Cartopy PlateCarree when available; graceful fallback otherwise.
  - Identity stamping + metadata CSV
      - Title and filename stamping include MMSI and [t_start, t_end] in UTC.
      - New flags: --stamp_titles/--no_stamp_titles, --stamp_filename/--no_stamp_filename (default on).
      - On-plot annotation near current point with --annotate_id.
      - Metadata CSV for every plot with: sample_idx, model, mmsi, trip_id, mode, pred_cut, n_total, n_past, n_true_future, n_pred, t_start_iso,
        t_end_iso, lat/lon bounds, out_path.
      - --save_meta/--no_save_meta (default on), --meta_path.
      - Optional per-MMSI subfolders with --output_per_mmsi_subdir; writes a local per-MMSI CSV plus optionally appends to the global
        --meta_path.
  - CLI additions (see python -m src.eval.eval_traj_newnewnew --help)
      - Modes/selection: --mmsi {all|<int>}, --trip_id <int>, --list_only, --seed, --log_skip_reasons, --min_points.
      - Full trip: --full_trip, --pred_cut <pct>, --cap_future <int>, --output_per_mmsi_subdir.
      - Identity/meta: --stamp_titles, --stamp_filename, --save_meta, --meta_path, --timefmt, --annotate_id.

  Usage examples

  - Default windows + full-trip overlay and titles/IDs:
      - Same as now for windows; add --full_trip to overlay context; --annotate_id to label current point.
  - Dry run: lists selected files without a checkpoint:
    python -m src.eval.eval_traj_newnewnew \
    --split_dir data/map_reduced/val \
    --max_plots 10 \
    --list_only --seed 0
  - Single trip, predict last 10% (anchored):
    python -m src.eval.eval_traj_newnewnew \
    --split_dir data/map_reduced/val \
    --ckpt data/checkpoints/traj_tptrans.pt \
    --model tptrans \
    --full_trip \
    --mmsi 209867000 --trip_id 0 \
    --pred_cut 90 --cap_future 60 \
    --auto_extent --extent_source actual \
    --lat_idx 0 --lon_idx 1 --y_order latlon \
    --denorm --lat_min 54 --lat_max 58 --lon_min 6 --lon_max 16 \
    --annotate_id --log_skip_reasons
  - Batch all, per-MMSI subfolders:
    python -m src.eval.eval_traj_newnewnew \
    --split_dir data/map_reduced/val \
    --ckpt data/checkpoints/traj_tptrans.pt \
    --model tptrans \
    --lat_idx 0 --lon_idx 1 --y_order latlon \
    --full_trip --mmsi all --pred_cut 85 \
    --auto_extent --extent_source actual \
    --out_dir data/figures \
    --output_per_mmsi_subdir \
    --save_meta --meta_path data/figures/traj_eval_meta.csv \
    --log_skip_reasons --seed 42

  Notes

  - In default “multi” mode, ADE/FDE are computed on windowed samples as before. Plots use windows; --full_trip only overlays context, not full
    MMSI path for windows.
  - For trip modes, the prediction horizon is min(cap_future, model.horizon, true future length). If your checkpoint was trained with a specific
    horizon, the model uses its internal horizon; cap it if needed.
  - If you want me to reflect these new modes in README’s visualization section, I can add a concise subsection with the above example commands.
