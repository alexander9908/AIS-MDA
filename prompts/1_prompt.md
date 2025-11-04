🧩 Context

# AIS-MDA — Increasing Maritime Domain Awareness with Spatio-Temporal Models

This repository provides a **ready-to-run** pipeline for working with real, irregular, and noisy **AIS** (Automatic Identification System) data to solve three key tasks:
- **Trajectory prediction** — forecast future vessel positions (sequence-to-sequence).
- **Anomaly detection** — detect abnormal vessel behavior using self-supervised learning.
- **Port call & ETA prediction** — classify the next port and estimate time of arrival.

It includes:
- Modular data loading, cleaning, and segmentation.
- Feature engineering (Δt, Δx/Δy, rate of turn, acceleration).
- Label generation for trajectory, anomaly, and ETA tasks.
- Baseline models (Constant Velocity, GRU, LSTM) and **TPTrans** (CNN + Transformer hybrid).
- Evaluation metrics (ADE, FDE, MAE, MAPE, AUROC, etc.).
- Ready-made configs, environment setup, and Docker support.

I’m working on a trajectory prediction pipeline. Below are the exact steps and scripts I’ve used so far.

⸻

⚙️ Data Processing Pipeline

1. Convert CSV → Pickle

python -m src.preprocessing.csv2pkl \
  --input_dir data/raw/ \
  --output_dir data/processed_pickle/

2. Map Reduce

python -m src.preprocessing.map_reduce \
  --input_dir data/processed_pickle/ \
  --temp_dir data/TEMP_DIR \
  --final_dir data/map_reduced/

3. Train/Test Split

python -m src.preprocessing.train_test_split \
  --data_dir data/map_reduced/ \
  --val_size 0.1 \
  --test_size 0.1 \
  --random_state 42

4. Train Model

python -m src.train.train_traj \
  --config configs/test_alex.yaml


⸻

📈 Trajectory Evaluation (current issue)

python -m src.eval.eval_traj_newnewnew \
  --split_dir data/map_reduced/val \
  --ckpt data/checkpoints/traj_tptrans.pt \
  --model tptrans \
  --lat_idx 1 --lon_idx 0 \
  --past_len 64 --max_plots 8 \
  --out_dir data/figures


⸻

🚨 Problem

The trajectory evaluation script (src.eval.eval_traj_newnewnew) does not produce the correct visualization.

I want a plot showing:
	•	A map of Europe
	•	Both actual and predicted trajectories plotted on it
(latitude and longitude pairs for each trajectory)
	•	Ideally using something like cartopy, geopandas, or matplotlib.basemap (depending on how the repo is structured)

⸻

🧠 Data Info

Each record in the dataset contains these columns:

LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI


⸻

✅ What I Need Help With
	1.	Read and understand how the current code works (especially eval_traj_newnewnew).
	2.	Fix or extend it so that it produces the correct figure:
	•	Europe map background
	•	Actual vs predicted trajectories clearly visualized
	•	Saved under data/figures/
	3.	Make sure it uses the correct lat/lon indices and aligns with the trained model output.

Let me know what parts of the code or config you need to see (e.g., the eval_traj_newnewnew script, model output format, or plotting utility functions).




# Output 
 Trajectory Eval Update

  - Added resilient Natural Earth basemap loading (user path, GeoDatasets, GeoPandas legacy, or bundled pyogrio fixture fallback) so plots
    render a Europe map even offline, plus CLI hooks to override extent/zoom (--basemap_path, --map_extent, --auto_extent). src/eval/
    eval_traj_newnewnew.py:24-src/eval/eval_traj_newnewnew.py:152
  - Reworked plot_samples to draw past/true/predicted tracks atop the basemap with equal-aspect axes, start/end markers, padded extents, and
    optional per-trajectory zoom. src/eval/eval_traj_newnewnew.py:120-src/eval/eval_traj_newnewnew.py:171
  - Documented the updated evaluation workflow and options in the visualization section. README.md:151-README.md:164

  Tests

  - python -m compileall src/eval/eval_traj_newnewnew.py
  - python -m src.eval.eval_traj_newnewnew --help

  You’re ready to rerun python -m src.eval.eval_traj_newnewnew … to generate the Europe-map figures; switch on --auto_extent when you want a
  zoomed view per sample.