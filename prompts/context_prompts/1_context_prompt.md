I’m working on a machine learning project titled **“Increasing Maritime Domain Awareness using Spatio-Temporal Sequential Data.”**  
The goal is to use **AIS (Automatic Identification System)** vessel data to train deep sequential models (RNNs, CNNs, Transformers) for three maritime challenges:

1. **Trajectory prediction** — forecasting a ship’s next positions.  
2. **Anomaly detection** — identifying unusual vessel behavior using self-supervised learning.  
3. **Port call & ETA prediction** — classifying the next port and estimating time of arrival.

The dataset contains irregular, noisy AIS time-series: MMSI, timestamp, latitude, longitude, speed (SOG), course (COG), heading, nav status, and optional metadata (ship type, draught, destination).  
I’ll preprocess it by cleaning invalid/missing data, segmenting on time gaps, and computing derived kinematic features like Δt, Δx/Δy, rate of turn (ROT), acceleration, and grid-cell encodings (H3 or UTM).

I’m implementing:
- Baselines: Constant-velocity model, Extended Kalman Filter, GRU/LSTM/Bi-LSTM seq2seq.  
- Advanced model: **TPTrans (CNN + Transformer hybrid)** for spatio-temporal trajectory prediction.
- Optional: self-supervised forecaster for anomaly detection.

Evaluation metrics include:
- **Trajectory:** ADE, FDE, Hausdorff distance  
- **ETA:** MAE, MAPE, P95  
- **Anomaly:** AUROC, AUPRC, Time-to-detection

I have four guiding papers:
- *Artificial Intelligence in Ship Trajectory Prediction* (2024) — provides taxonomy and preprocessing guidance.  
- *TPTrans: Vessel Trajectory Prediction Model Based on CNN and Transformer* (2023) — defines hybrid CNN+Transformer model.  
- *Vessel Trajectory Prediction with Deep Learning Techniques* (JMSE, 2025) — empirically compares GRU/LSTM/Transformer.  
- *Prediction of Vessel Arrival Time to Port — A Review of Current Studies* (2025) — summarizing ETA prediction literature and key features.

I want to design a modular codebase structured into `/src` folders for data loading, feature extraction, labeling, modeling, training, and evaluation, and automate experiments with YAML configs.  

I’ll start with real AIS data (e.g., NOAA or Global Fishing Watch) for one region, train baseline GRU models, then extend to TPTrans and ETA prediction.

I want help developing:
- Data preparation scripts
- Model implementations (GRU, TPTrans)
- Evaluation metrics
- Experiment tracking (e.g., YAML, WandB)
- Documentation (README, visuals)

Assume this is a research-grade project aiming for publishable-quality results.
