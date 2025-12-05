# Enhancing Maritime Domain Awareness: Robust Vessel Trajectory Prediction

**02456 Deep Learning, DTU Compute, Fall 2025**

This repository contains the implementation for **TPTrans** (a hybrid CNN-Transformer) and **TrAISformer** (a discrete generative Transformer) for vessel trajectory prediction in the Danish waters. The project introduces a scalable **MapReduce** preprocessing pipeline and a density-based **K-Means sampling strategy** to mitigate open-sea bias and capture complex maneuvering behaviors.

## 📄 Abstract

This paper addresses the challenge of forecasting vessel trajectories using Automatic Identification System (AIS) data from the Danish maritime authority (Søfartsstyrelsen). We introduce a MapReduce pipeline and K-means sampling strategy to mitigate ``open-sea bias'' and capture diverse maneuvers. Validating a kinematic Kalman Filter against a discrete TrAISformer and our custom continuous hybrid CNN-Transformer (TPTrans), we identify a critical trade-off: while TPTrans achieves superior short-term precision (lowest 1-hour ADE), TrAISformer demonstrates greater long-term stability and navigational realism. Ultimately, the navigational realism inherent in discrete generative architectures like TrAISformer establishes the necessary foundation for trustworthy, autonomous maritime surveillance.

-----

## 🚀 Quick Start

### 1\. Installation

Clone the repository and install dependencies.

```bash
git clone https://github.com/alexander9908/AIS-MDA.git
cd AIS-MDA
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2\. Data Pipeline

We utilize a scalable pipeline to handle large-scale AIS logs (tested on 3 months of DMA data).

**Step A: Ingest CSV to Pickle**
Converts raw CSVs into efficient per-vessel pickle files.

```bash
python -m src.preprocessing.csv2pkl \
    --input_dir data/raw/ \
    --output_dir data/processed_pickle_2/ \
    --cargo_tankers_only \
    --run_name ingestion_run
```

**Step B: MapReduce Processing**
Cleans, segments voyages (\>2h gaps), interpolates (5 min), and normalizes data.

```bash
python -m src.preprocessing.map_reduce \
    --input_dir data/processed_pickle_2/ \
    --output_dir data/processed_final/map_reduced/ \
    --num_workers 0 \
    --run_name map_reduce_run
```

**Step C: Train/Val/Test Split**
Splits data by MMSI to prevent leakage.

```bash
python -m src.preprocessing.train_test_split \
    --data_dir data/processed_final/map_reduced/ \
    --val_size 0.1 \
    --test_size 0.1 \
    --random_state 42
```

-----

## 🧠 Model Training

We support two primary deep learning architectures. Configuration is handled via YAML files in `configs/`.

### Train TPTrans (Hybrid CNN-Transformer)

Our custom continuous regression model.

```bash
python -m src.train.train_tptrans_transformer --config configs/traj_tptrans.yaml
```

### Train TrAISformer (Generative Discrete)

The discrete classification model based on Nguyen et al.

```bash
python -m src.train.train_traisformer --config configs/traj_traisformer.yaml
```

*Note: Both training scripts automatically utilize the K-Means sampling strategy defined in the dataloader configuration.*

-----

## 📊 Evaluation & Visualization

We provide a comprehensive evaluation script that calculates ADE/FDE metrics and generates qualitative plots (static images and interactive maps).

**Run Full Evaluation on Test Set:**
This command evaluates TPTrans, TrAISformer, and the Kalman baseline, generating an interactive Folium map and trajectory plots for specific vessels.

```bash
python -m src.eval.evaluate_trajectory \
  --split_dir data/processed/map_reduced/test \
  --ckpt data/checkpoints/traj_tptrans_delta.pt,data/checkpoints/traj_traisformer.pt \
  --model tptrans,traisformer,kalman \
  --out_dir data/figures/final/all_models_test \
  --pred_cut 80 \
  --folium \
  --same_pic \
  --collect \
  --samples 1 --temperature 0 --top_k 20 \
  --mmsi 212801000,215933000,218615000,230617000,244554000,248891000,250005981,255802840,305575000,305643000,352005235,636015943,636022355
```

**Key Arguments:**

  * `--pred_cut 80`: Uses the first 80% of a trip as history to predict the remaining 20%.
  * `--folium`: Generates an interactive HTML map (`map_model.html`).
  * `--mmsi`: (Optional) Comma-separated list of MMSIs to visualize specific vessels.
  * `--no_plots`: (Optional) does not plot, only gives metrics and predictions.

**Full Workflow pipline can be seen in notebooks/workflow.ipynb**
-----

## 📈 Results

### Quantitative Performance (ADE/FDE in km)
$$
\begin{tabular}{l cc cc cc}
    \toprule
    & \multicolumn{2}{c}{\textbf{1 Hour Horizon}} & \multicolumn{2}{c}{\textbf{2 Hour Horizon}} & \multicolumn{2}{c}{\textbf{3 Hour Horizon}} \\
    \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
    \textbf{Model} & \textbf{ADE} & \textbf{FDE} & \textbf{ADE} & \textbf{FDE} & \textbf{ADE} & \textbf{FDE} \\
    & \scriptsize{(Mean/Med)} & \scriptsize{(Mean/Med)} & \scriptsize{(Mean/Med)} & \scriptsize{(Mean/Med)} & \scriptsize{(Mean/Med)} & \scriptsize{(Mean/Med)} \\
    \midrule
    Kalman Filter   & 1.58 / \textbf{0.66} & 3.60 / 1.50 & 3.70 / 1.77 & 8.86 / 4.64 & 6.05 / 3.38 & 14.84 / 9.21 \\
    TrAISformer     & 1.38 / 0.90 & 2.41 / 1.37 & 2.62 / \textbf{1.48} & \textbf{5.42} / \textbf{2.70} & \textbf{4.20} / \textbf{2.24} & \textbf{9.41} / \textbf{4.53} \\
    \textbf{TPTrans (Ours)} & \textbf{1.21} / 0.79 & \textbf{2.07} / \textbf{1.23} & \textbf{2.55} / 1.50 & 5.85 / 2.92 & 4.61 / 2.60 & 11.69 / 6.28 \\
    \midrule
    \multicolumn{7}{c}{\footnotesize \textit{Dataset Statistics: Mean Trip Length = 4.17 h, Median Trip Length = 4.42 h}} \\
    \bottomrule
\end{tabular}
$$

*TPTrans achieves the best short-term precision, while TrAISformer demonstrates superior long-term stability.*

-----

## 👥 Authors 

**Technical University of Denmark (DTU)**

  * **Alexander Schiøtz** (s221221)
  * **Felix Thomsen** (s221710) 
  * **Bertram Hage** (s224918)
  * **Christian Rand** (s224930)

**Supervisor:** Dr. Peder Heiselberg (DTU Space).

-----
