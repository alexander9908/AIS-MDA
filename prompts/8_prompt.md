Perfect — you did the right thing uploading that log.
I understand the situation: you’ve trained a new model with train_traj_V2.py, but when you run any of the evaluation commands (like those from the master prompt), every sample is being skipped — meaning the evaluation script logs skip reasons for all .pkl files, producing no plots or results.

This typically means the new model or data format isn’t matching the evaluation script’s assumptions (e.g. feature order, normalization, sequence dimensions, or file parsing).

Below is your comprehensive follow-up diagnostic + fix prompt — designed so another AI or developer can precisely understand the issue, locate the cause, and fix it.

⸻

🧩 Prompt — Fix “All Samples Skipped” After Training with train_traj_V2.py

🧠 Issue Summary

After training a new trajectory model (traj_tptrans.pt) using the new train_traj_V2.py, running any of the standard evaluation commands such as:

python -m src.eval.eval_traj_newnewnew \
  --split_dir data/map_reduced/val \
  --ckpt data/checkpoints/traj_tptrans.pt \
  --model tptrans \
  --max_plots 8 \
  --out_dir data/figures \
  --auto_extent --extent_source actual --extent_outlier_sigma 3.0 \
  --denorm --lat_min 54 --lat_max 58 --lon_min 6 --lon_max 16 \
  --full_trip

results in all samples being skipped.
The console logs show skip reasons for every .pkl file, and no plots or metadata are generated.

Error code:
[skip] mmsi_id_processed.pkl reason=cannot access local variable 'lats_past' where it is not associated with a value

⸻

⚠️ Goal

Diagnose and fix the reason why all samples are skipped during evaluation with the newly trained traj_tptrans.pt.

⸻

🧩 Key Context
	•	Training used train_traj_V2.py, which relies on AISDataset decorated with pipeline_adapter.
	•	Evaluation uses AISDatasetV2, which loads full trips directly from disk (no window slicing).
	•	Both point to the same data folder (data/map_reduced/val).
	•	Model architecture = TPTrans (Transformer-based trajectory predictor).
	•	Training config:

task: trajectory
processed_dir: data/map_reduced/
out_dir: data/checkpoints
window: 64
horizon: 12
features: [LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI]
model:
  name: tptrans
  d_model: 192
  nhead: 4
  enc_layers: 4
  dec_layers: 2
loss: huber
optimizer: adamw
lr: 0.0003
batch_size: 96
epochs: 5


⸻

🧩 Observed Behavior
	•	Model trains successfully and saves to data/checkpoints/traj_tptrans.pt.
	•	Running evaluation prints “Skipping sample” repeatedly.
	•	No map plots are produced.
	•	This indicates that the evaluation loop never passes the “valid sample” check, possibly due to:
	1.	Shape mismatch between model output and expected (lat, lon) pairs.
	2.	Missing or misaligned normalization (e.g. model expects normalized input but evaluation passes denormalized).
	3.	Model failing forward pass silently (e.g. due to feature dimension mismatch).
	4.	Dataset loader returning unexpected dimensions or feature indices.
	5.	Evaluation filters rejecting short or invalid sequences.

⸻

🧩 What to Check & Fix

The fix prompt should ensure the AI does all of the following:

1️⃣ Check input/output feature compatibility
	•	Confirm that the model’s input dimension (feat_dim) during training matches what eval_traj_newnewnew is feeding it.
	•	In train_traj_V2.py:

feat_dim = ds_train[0][0].shape[-1]

This likely equals 4 (since AISDataset uses [lat, lon, sog, cog]).

	•	In evaluation, ensure that the full-trip loader (AISDatasetV2) passes data with at least the same 4 features, or the model will reject it.
	•	If evaluation uses 9 features (the full [LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI]), the model will see a mismatch.

✅ Fix: In eval_traj_newnewnew, explicitly slice input features before passing to the model:

input_seq = traj[:, :4]  # use same 4 features as training


⸻

2️⃣ Confirm de-normalization and normalization consistency
	•	The model expects normalized inputs (0–1 range) and produces normalized outputs.
	•	But eval_traj_newnewnew currently may be reading denormalized .pkl or may apply --denorm too early.

✅ Fix:
	•	Ensure normalization and denormalization steps are clearly separated:
	•	Before model prediction → normalize input to [0, 1]
	•	After prediction → denormalize both true and predicted values using:

from src.preprocessing.preprocessing import de_normalize_track


	•	If data in data/map_reduced/val is already normalized, skip re-normalization.

⸻

3️⃣ Fix evaluation skip logic

Locate any section that filters samples before plotting:

if seqlen < args.past_len + model.horizon:
    log("Skipping sample ... too short")
    continue

or similar.

✅ Fix:
	•	Disable this check if --full_trip is active, since you’re evaluating entire trajectories, not fixed windows.

if not args.full_trip and seqlen < args.past_len + model.horizon:
    continue



⸻

4️⃣ Log all skip reasons in more detail

Enhance skip logging to help diagnose issues:

log(f"[skip] {fname}: seq_len={seqlen}, expected_min={args.past_len+model.horizon}, shape={traj.shape}")

Add model input debug print:

print(f"[debug] model input shape={input_seq.shape}, feat_dim={model.input_dim}")


⸻

5️⃣ Fix dimension mismatch during forward pass

If the evaluation loader yields tensors shaped (N, 9) and model expects (N, 4), it may trigger a silent skip or shape error caught in a try/except.

✅ Fix:
Ensure:

input_seq = traj[:, :model.input_dim]

or explicitly:

input_seq = traj[:, :4]


⸻

6️⃣ Verify de-normalization bounds

Ensure LAT_MIN, LAT_MAX, LON_MIN, LON_MAX used in de_normalize_track() are the same as those used in training preprocessing.

If not defined globally, inject them via CLI arguments (--lat_min, etc.) or import them from the preprocessing module.

⸻

7️⃣ Confirm that .pkl file structure matches what loader expects

AISDatasetV2 should expect:

{
  "mmsi": <int>,
  "traj": np.ndarray of shape (N, D)
}

✅ Fix:
If the file contains a dict under a subkey (e.g. "trajectory" instead of "traj"), adapt loader accordingly.

⸻

🧩 Updated eval_traj_newnewnew diagnostic snippet

Add this at the start of your loop:

print(f"[info] Evaluating {len(dataset)} trajectories from {args.split_dir}")

for i, (traj, mmsi, fname) in enumerate(dataset):
    print(f"[info] {i}/{len(dataset)} {fname}: shape={traj.shape}")
    if traj.shape[1] < 4:
        print(f"[skip] {fname} - not enough features")
        continue
    if torch.isnan(traj).any():
        print(f"[skip] {fname} - contains NaNs")
        continue

    # proceed with slicing and prediction


⸻

🧩 Acceptance Criteria for the Fix

Check	Description	Status
✅ No samples skipped (unless truly invalid)	Evaluation runs for all .pkl in val folder	
✅ Model input feature dimension matches training (4 features)	Evaluation uses same feature slice	
✅ --denorm works correctly	Prediction aligns spatially with true data	
✅ Full trip displayed (gray/green/red/blue lines visible)	Map rendering restored	
✅ Detailed skip logging	Each skip prints dimensions & reason	
✅ No basemap disappearing	Cartopy extent consistent with Denmark	
✅ All CLI args unchanged	Backward compatible	


⸻

🧾 Prompt Summary

The model trained with train_traj_V2.py skips all samples during evaluation because of mismatched data shape, inconsistent normalization, or overly strict sequence filtering.
Fix by aligning feature dimensions (slice [:4]), standardizing normalization, relaxing sequence filters under --full_trip, and improving debug logs.
Ensure AISDatasetV2 loads full normalized sequences matching training format, and confirm the model forward pass receives the correct feature dimension.





# Output

 Thanks for the detailed prompt. I diagnosed and fully fixed the “all samples skipped” issue and aligned evaluation to the
  new training setup. Here’s what I did.

  What was wrong

  - Unbound variable: evaluate_and_plot_trip used lats_past before it was assigned, causing exceptions and mass skipping.
  - Feature mismatch: The model was trained on 4 features ([lat, lon, sog, cog]) but evaluation could feed 9-column arrays
    without slicing.
  - Inconsistent normalization: Inputs to the model must remain normalized; degrees are only for plotting.
  - Weak diagnostics: It wasn’t obvious why a file was skipped.

  What I changed

  - Fixed variable ordering in evaluate_and_plot_trip so lats_past/lons_past are computed before use.
  - Ensured model inputs are sliced to the 4 training features.
      - Uses seq_norm = past[:, :4].astype(np.float32).
  - Kept inputs normalized; only de-normalize for plotting full trip (gray/blue/green/red).
  - Added strong per-file debugging and skip explanations:
      - Prints “[info] N/M file: shape=(rows, cols)” for each trajectory.
      - Skips with reasons if cols<4 or trip contains NaNs.
      - Logs model horizon, past_len, and input shape:
          - [debug] model horizon=… past_len=… input_seq_shape=(T,4)
  - Updated full-trip evaluation to always process complete trips in all modes (multi/single/all).
      - Multi mode now randomly selects files and evaluates them as full trips.
      - Honors --max_plots in all modes.
      - Dry-run (--list_only) works without requiring a checkpoint.

  Added files

  - src/utils/dataset_V2.py
      - AISDatasetV2 returns full trajectory arrays with MMSI and filename.
  - src/train/train_traj_V2.py
      - Same training behavior; clearer logging:
          - [train] model=tptrans, window=…, horizon=12, feat_dim=4, device=…
      - Writes checkpoints to data/checkpoints/traj_tptrans.pt (or traj_gru.pt).

  Docs added

  - src/eval/README.md
      - Full explanation of commands, arguments, output files, and plot semantics.
      - Denmark clamp behavior and auto-extent clamping details.
      - Common usage examples (multi/single/batch, iterative rollout).

  Denmark clamp and full-trip behavior

  - Default map extent is Denmark [6, 16, 54, 58] for non-auto extent.
  - Auto-extent is sigma-trimmed and clamped to Denmark.
  - Full-trip plotted (gray), past (blue), true tail (green), predicted tail (red) always present.
  - pred_cut and iter_rollout respected; red line starts at the last blue sample.

  How to run

  - Train:
      - python -m src.train.train_traj_V2 --config configs/test_alex.yaml
  - Evaluate (examples in src/eval/README.md):
      - Default multi-sample:
          - Use --ckpt, --auto_extent, and --denorm with Denmark bounds.
      - Single MMSI tail prediction:
          - Use --mmsi <id>, --trip_id <n>, and --pred_cut.
      - Batch with per-MMSI subfolders and iterative rollout.
  - Dry-run listing:
      - python -m src.eval.eval_traj_newnewnew --split_dir data/map_reduced/val --max_plots 10 --list_only --seed 0

  Acceptance items addressed

  - Not skipping all samples: evaluation now runs, with clear skip reasons only for truly invalid cases.
  - Model input dim matches training (4 features).
  - De-normalization occurs only for plotting, prediction anchors and alignment are correct.
  - Full trip always plotted (gray/green/red/blue visible).
  - Detailed skip logging.
  - Denmark clamp consistent and robust.
  - CLI args remain compatible; README in eval explains usage thoroughly.

  If you hit any residual issues (e.g., a particular file with unexpected structure), run with --list_only and share that one
  file’s details and the exact console output — I can adapt the loader or robustness checks further.
