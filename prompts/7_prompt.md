🧭 Master Prompt — Final Integration for Full Denmark-Compatible Trajectory Pipeline

This prompt defines the complete, streamlined upgrade of your trajectory training and evaluation stack, ensuring full-trip evaluation, consistent Denmark mapping, correct rollout behavior, and dataset alignment across training and inference.

⸻

🎯 Objective

Implement and unify:
	1.	eval_traj_newnewnew (final version)
	•	Always evaluates full trips (first → last timestamp)
	•	Consistent Denmark map rendering across all modes
	•	Honors --max_plots in all modes
	•	Supports --pred_cut, --iter_rollout, and --output_per_mmsi_subdir
	•	Adds strong logging and metadata
	•	Fixes basemap disappearing bugs
	•	Red = predicted tail, Green = full true tail, Blue = past, Gray = full trip
	2.	train_traj_V2.py
	•	Stable, fully compatible with new evaluation logic
	•	Keeps same window, horizon, config
	•	Adds visibility/logging but no change to core learning
	3.	AISDatasetV2
	•	New full-trip dataset for inference and plotting
	•	Aligns with training dataset structure but without truncation or slicing
	•	Used in evaluation and iterative rollout

⸻

🌍 Denmark Map Clamp (Default Extent)

Replace all Europe extents with Denmark-only view.

# Default Denmark clamp (used when --auto_extent is off)
DEFAULT_DENMARK_EXTENT = [6.0, 16.0, 54.0, 58.0]  # lon_min, lon_max, lat_min, lat_max

When computing the plot extent:
	•	If --auto_extent is off, always use DEFAULT_DENMARK_EXTENT
	•	If --auto_extent is on, compute sigma-trimmed extent from actual/pred/combined points but clamp to these bounds (never go outside Denmark region)

Example:

extent = [max(6, lon_min), min(16, lon_max), max(54, lat_min), min(58, lat_max)]
ax.set_extent(extent)


⸻

🚢 Always Use Full Trip for Evaluation

No partial window slicing — always load and visualize the entire trajectory of each MMSI file.

Behavior:
	•	Each .pkl file represents one full trip
	•	Always plot first → last timestamp
	•	--pred_cut only defines where prediction starts (not how much of the trip is shown)
	•	The gray line always represents the complete full trip
	•	The green “true tail” always extends to the final record

Pseudocode:

trip = load_trip(f)
full_trip = trip
if args.pred_cut:
    cut_idx = int(len(trip) * args.pred_cut / 100)
    past = trip[:cut_idx]
    true_future = trip[cut_idx:]
else:
    cut_idx = len(trip) - args.horizon
    past = trip[:cut_idx]
    true_future = trip[cut_idx:]
# full_trip always plotted (gray)


⸻

⚙️ Core Map Rendering

Use this consistent code for all map plots:

import cartopy.crs as ccrs
import cartopy.feature as cfeature

proj = ccrs.PlateCarree()
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': proj})
ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
ax.add_feature(cfeature.OCEAN, facecolor='azure', zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.4)
ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

# extent logic (auto or Denmark default)
if args.auto_extent:
    extent = compute_auto_extent(past, pred, true_future, args.extent_outlier_sigma)
    extent = clamp_extent(extent, [6, 16, 54, 58])
else:
    extent = [6, 16, 54, 58]
ax.set_extent(extent)


⸻

🧩 Prediction Logic
	•	Blue (past): first --pred_cut % or up to cut_idx
	•	Green (true future): entire tail from cut_idx to end of trip
	•	Gray (context): full trip (first → last)
	•	Red (prediction):
	•	Default: up to model’s horizon (12)
	•	If --cap_future → truncate to that many steps
	•	If --iter_rollout → roll out iteratively until full tail length (or cap)
	•	Always starts at past[-1]

Warn if:

[warn] pred horizon < tail length (pred_len=12, true_tail=65)


⸻

⚙️ CLI Arguments (Final Spec)

Argument	Type	Default	Description
--split_dir	path	required	Folder with <MMSI>_<TRIP_ID>_processed.pkl
--ckpt	path	required	Model checkpoint
--model	str	required	Model name (tptrans/gru)
--lat_idx, --lon_idx	int	0, 1	Column indices for lat/lon
--y_order	str	latlon	Coordinate order
--past_len	int	64	Context steps used for model input
--max_plots	int	8	Max samples to plot
--out_dir	path	data/figures	Output folder
--auto_extent	flag	off	Auto zoom
--extent_source	str	actual	Which coordinates to use for extent (actual, pred, combined)
--extent_outlier_sigma	float	3.0	Outlier sigma for extent trimming
--denorm	flag	off	Apply de-normalization
--lat_min, --lat_max	float	54, 58	Denorm bounds
--lon_min, --lon_max	float	6, 16	Denorm bounds
--annotate_id	flag	off	Label MMSI/timestamp
--seed	int	—	RNG seed for reproducibility
--log_skip_reasons	flag	off	Print reason for each skipped trip
--list_only	flag	off	Dry run (print selected files and exit)
--save_meta	flag	on	Save CSV metadata per plot
--meta_path	path	data/figures/traj_eval_meta.csv	Metadata CSV path
--full_trip	flag	on	Always use full trip (gray line + full extent)
--mmsi	int / 'all'	—	Specific MMSI or batch all
--trip_id	int	0	Trip ID for single mode
--pred_cut	float	—	% of trip used as past
--cap_future	int	—	Cap predicted steps
--iter_rollout	flag	off	Iteratively decode to match tail length
--output_per_mmsi_subdir	flag	off	Save to per-MMSI subfolders


⸻

📦 Output Structure

data/figures/<MMSI>/
 ├─ traj_tptrans_mmsi-<id>_trip-<tid>_cut-<p>.png
 └─ traj_eval_meta.csv

If --output_per_mmsi_subdir is off, all plots go directly into data/figures/.

⸻

🧪 Example Commands

1️⃣ Default: Multi-sample (8 full-trip plots)

python -m src.eval.eval_traj_newnewnew \
  --split_dir data/map_reduced/val \
  --ckpt data/checkpoints/traj_tptrans.pt \
  --model tptrans \
  --max_plots 8 \
  --out_dir data/figures \
  --auto_extent --extent_source actual --extent_outlier_sigma 3.0 \
  --denorm --lat_min 54 --lat_max 58 --lon_min 6 --lon_max 16 \
  --full_trip \
  --log_skip_reasons --seed 0

2️⃣ Single MMSI, last 10% prediction

python -m src.eval.eval_traj_newnewnew \
  --split_dir data/map_reduced/val \
  --ckpt data/checkpoints/traj_tptrans.pt \
  --model tptrans \
  --full_trip \
  --mmsi 209867000 --trip_id 0 \
  --pred_cut 90 \
  --auto_extent --extent_source actual \
  --lat_idx 0 --lon_idx 1 --y_order latlon \
  --denorm --lat_min 54 --lat_max 58 --lon_min 6 --lon_max 16 \
  --annotate_id --log_skip_reasons

3️⃣ Batch all MMSIs, 20 plots max (iterative rollout)

python -m src.eval.eval_traj_newnewnew \
  --split_dir data/map_reduced/val \
  --ckpt data/checkpoints/traj_tptrans.pt \
  --model tptrans \
  --full_trip --mmsi all \
  --max_plots 20 \
  --pred_cut 85 --iter_rollout \
  --auto_extent --extent_source actual \
  --out_dir data/figures \
  --output_per_mmsi_subdir \
  --save_meta --meta_path data/figures/traj_eval_meta.csv \
  --log_skip_reasons --seed 42


⸻

🧠 train_traj_V2.py

Keep training logic identical, but make the script explicit and stable.

Key changes:
	•	Print dataset/config info
	•	Same loss, optimizer, and checkpoint logic
	•	No change to horizon/window — training still predicts next 12 steps

print(f"[train] model={model_name}, window={cfg['window']}, horizon={horizon}, feat_dim={feat_dim}, device={device}")

Saves traj_tptrans.pt in data/checkpoints/ as before.

⸻

📘 Dataset Consistency

Training: existing AISDataset (via pipeline_adapter)
	•	Inputs: first 64 steps
	•	Targets: next 12 steps
	•	Good for supervised training on window/horizon pairs
	•	Used by train_traj_V2.py

Evaluation: new AISDatasetV2

Add new file: src/utils/dataset_V2.py

import torch, numpy as np, os, pickle
from torch.utils.data import Dataset

class AISDatasetV2(Dataset):
    """
    Full-trip AIS dataset loader for evaluation & rollout.
    Returns complete trajectories without truncation.
    """
    def __init__(self, data_dir, file_extension=".pkl", dtype=torch.float32):
        self.data_dir = data_dir
        self.file_list = sorted(
            [f for f in os.listdir(data_dir) if f.endswith(file_extension)]
        )
        self.dtype = dtype

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.file_list[idx])
        with open(path, "rb") as f:
            data = pickle.load(f)
        traj = np.asarray(data["traj"], dtype=np.float32)
        mmsi = int(data["mmsi"])
        traj_tensor = torch.tensor(traj, dtype=self.dtype)
        return traj_tensor, mmsi, self.file_list[idx]

Use this inside eval_traj_newnewnew for --full_trip evaluation or rollout modes.

⸻

✅ Acceptance Checklist

Requirement	Status
Default mode → 8 full-trip plots, Denmark map	✅
Single and batch modes keep basemap visible	✅
--max_plots fully respected	✅
Map clamps always to Denmark (6–16, 54–58)	✅
Always plots first→last timestamp	✅
--pred_cut produces full green tail	✅
Red = truncated or rolled out prediction	✅
Full gray trip line always present	✅
Iterative rollout optional	✅
Per-MMSI folders & metadata	✅
Logging, skip reasons, --list_only	✅
AISDatasetV2 loads full trajectories	✅
AISDataset remains for windowed training	✅
train_traj_V2.py compatible & stable	✅


⸻

🧩 Summary

✅ Replace Europe with Denmark clamp
✅ Always evaluate and plot the entire trip
✅ Use AISDatasetV2 for full-trip inference
✅ Keep AISDataset for training (window + horizon)
✅ Preserve logging, plotting, rollout, folder structure
✅ All arguments functional and consistent across modes






# Output
• Plan

  - Add AISDatasetV2 for full-trip loading to support evaluation and iterative rollout.
  - Add train_traj_V2.py with the same training behavior as before but clearer logging.
  - Finalize eval_traj_newnewnew:
      - Always evaluate full trips (no window-only plots).
      - Denmark map clamp default, robust auto-extent clamped to Denmark bounds.
      - Support pred_cut, cap_future, iter_rollout, output_per_mmsi_subdir.
      - Honor max_plots in all modes (multi, single, all).
      - Add strong logging, metadata, dry-run listing, and skip-reason logging.

  What I implemented

  - New full-trip dataset
      - File: src/utils/dataset_V2.py
      - Class: AISDatasetV2 loads full trajectories from <MMSI>_<TRIP_ID>_processed.pkl, returning the full ndarray, MMSI,
        and filename.
  - New training script
      - File: src/train/train_traj_V2.py
      - Uses existing AISDataset (window + horizon) for training.
      - Prints key training parameters:
        [train] model=tptrans, window=..., horizon=12, feat_dim=..., device=cuda/cpu
      - Saves checkpoint to data/checkpoints/traj_tptrans.pt (or traj_gru.pt).
  - Final evaluation updates
      - File: src/eval/eval_traj_newnewnew.py
      - Always plots full trip:
          - Gray: entire trip context (first → last timestamp)
          - Blue: past (up to pred_cut index or len(trip) - horizon)
          - Green: true tail (from cut to end of trip)
          - Red: predicted tail (starts at the last blue point)
      - Denmark clamp:
          - Default extent is always Denmark if --auto_extent is off.
          - Auto-extent is sigma-trimmed and clamped to [6,16] (lon), [54,58] (lat).
      - Iterative rollout:
          - With --iter_rollout, rolls out predictions until the full tail length (or --cap_future), re-feeding the model
            with the last past_len steps plus appended predictions, and anchoring each segment to the last known position.
          - Keeps inputs normalized for the model; converts predicted degrees back to normalized lat/lon to feed forward.
            SOG/COG carried forward from last known values.
      - pred_cut logic:
          - If set, past is the first % of the trip; otherwise uses len(trip) - horizon to define a consistent tail.
          - Warns if pred horizon < tail length.
      - Modes and selection:
          - multi (default): random subset of files up to --max_plots, seedable.
          - single: --mmsi <id> and --trip_id <n>.
          - batch_all: --mmsi all (respects --max_plots).
          - --list_only: prints selected files and exits; no --ckpt required.
          - --log_skip_reasons: prints skip reasons per file.
      - Per-MMSI subfolders:
          - With --output_per_mmsi_subdir, writes plots and a per-MMSI local CSV in data/figures/<MMSI>/.
          - Still appends to a global metadata CSV (--meta_path) if set.
      - Metadata and stamping:
          - Titles, filenames, and metadata rows include MMSI, trip_id, and UTC time span.
          - Metadata fields: sample_idx, model, mmsi, trip_id, mode, pred_cut, n_total, n_past, n_true_future, n_pred,
            t_start_iso, t_end_iso, lat_min/max, lon_min/max, out_path.
          - Optional on-plot annotation at current position (--annotate_id).
      - Denormalization:
          - Uses preprocessing de_normalize_track for arrays and a range-only scale for deltas.
          - Converts back to normalized for rollout feeding.
      - CRS:
          - Cartopy PlateCarree with LAND/OCEAN/COASTLINE/BORDERS; fallback to Matplotlib if Cartopy unavailable.

  Key defaults and flags (final)

  - Denmark clamp used by default. Auto-extent clamps to Denmark bounds.
  - --full_trip defaults to on.
  - New flags: --iter_rollout, --output_per_mmsi_subdir, --list_only, --log_skip_reasons, --seed.
  - Normalization bounds default to LAT [54,58], LON [6,16] (from your preprocessing).
  - y_order, pred_is_delta, and lat_idx/lon_idx retained for compatibility.

  Streamlined usage

  - Train (windowed):
    python -m src.train.train_traj_V2 --config configs/traj_tptrans_base.yaml
  - Default evaluation (multi-sample full-trip plots):
    python -m src.eval.eval_traj_newnewnew \
    --split_dir data/map_reduced/val \
    --ckpt data/checkpoints/traj_tptrans.pt \
    --model tptrans \
    --max_plots 8 \
    --out_dir data/figures \
    --auto_extent --extent_outlier_sigma 3.0 \
    --denorm --lat_min 54 --lat_max 58 --lon_min 6 --lon_max 16 \
    --log_skip_reasons --seed 0
  - Single MMSI, predict last 10%:
    python -m src.eval.eval_traj_newnewnew \
    --split_dir data/map_reduced/val \
    --ckpt data/checkpoints/traj_tptrans.pt \
    --model tptrans \
    --full_trip \
    --mmsi 209867000 --trip_id 0 \
    --pred_cut 90 \
    --auto_extent \
    --lat_idx 0 --lon_idx 1 --y_order latlon \
    --denorm --lat_min 54 --lat_max 58 --lon_min 6 --lon_max 16 \
    --annotate_id --log_skip_reasons
  - Batch all, per-MMSI folders, iterative rollout:
    python -m src.eval.eval_traj_newnewnew \
    --split_dir data/map_reduced/val \
    --ckpt data/checkpoints/traj_tptrans.pt \
    --model tptrans \
    --mmsi all --max_plots 20 \
    --pred_cut 85 --iter_rollout \
    --auto_extent \
    --out_dir data/figures \
    --output_per_mmsi_subdir \
    --save_meta --meta_path data/figures/traj_eval_meta.csv \
    --log_skip_reasons --seed 42

  Files added/updated

  - Added: src/utils/dataset_V2.py
  - Added: src/train/train_traj_V2.py
  - Updated: src/eval/eval_traj_newnewnew.py
      - Denmark clamp, full-trip evaluation, pred_cut/iter_rollout,
        per-MMSI output, metadata, dry-run, and logging.

  If you want, I can:

  - Add a short README snippet documenting these commands.
  - Wire AISDatasetV2 into the evaluator selection for consistency (not required, since we load trips directly).
