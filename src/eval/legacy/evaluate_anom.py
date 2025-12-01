from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

from ..models import GRUSeq2Seq
from .metrics_traj import ade, fde  # optional

def make_simple_anomalies(X: np.ndarray, rate: float = 0.15, amp: float = 3.0, seed: int = 42):
    """
    Inject simple spikes into a fraction of windows to simulate anomalies.
    X: [N, T, F]
    Return: X_pert, labels (1=anomaly)
    """
    rng = np.random.default_rng(seed)
    N, T, F = X.shape
    labels = np.zeros((N,), dtype=np.int64)
    n_anom = int(N * rate)
    idx = rng.choice(N, size=n_anom, replace=False)
    X_pert = X.copy()
    for i in idx:
        t0 = rng.integers(low=T//4, high=3*T//4)
        f = rng.integers(low=0, high=min(F, 2))  # bias toward kinematics dims
        X_pert[i, t0:t0+2, f] *= amp
        labels[i] = 1
    return X_pert, labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed/anom_w64_h12")
    ap.add_argument("--ckpt", default="data/checkpoints/anom_gru.pt")
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    proc = Path(args.processed_dir)
    X = np.load(proc/"X.npy")     # [N,T,F]
    Y = np.load(proc/"Y.npy")     # [N,H,2]

    # apply the same scaler used in training
    s_path = proc/"scaler.npz"
    if s_path.exists():
        s = np.load(s_path)
        X = (X - s["mean"]) / (s["std"] + 1e-8)

    # model
    feat_dim, horizon = X.shape[-1], Y.shape[1]
    model = GRUSeq2Seq(feat_dim, d_model=128, layers=2, horizon=horizon)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # create anomalies on the inputs (self-supervised forecasting)
    X_pert, labels = make_simple_anomalies(X, rate=0.2, amp=3.0)
    # prediction errors are anomaly scores
    with torch.no_grad():
        X_t = torch.from_numpy(X).float().to(device)
        Xp_t = torch.from_numpy(X_pert).float().to(device)
        Y_pred = model(X_t).cpu().numpy()
        Yp_pred = model(Xp_t).cpu().numpy()

    # use forecast error (L2 over horizon) as score
    def seq_error(pred, true):
        return np.sqrt(((pred - true)**2).sum(axis=(1,2)))

    scores_clean = seq_error(Y_pred, Y)
    scores_pert  = seq_error(Yp_pred, Y)  # same ground truth, higher error → anomaly

    scores = np.concatenate([scores_clean, scores_pert], axis=0)
    ytrue  = np.concatenate([np.zeros_like(scores_clean), np.ones_like(scores_pert)], axis=0)

    auroc = roc_auc_score(ytrue, scores)
    auprc = average_precision_score(ytrue, scores)
    print(f"ANOM: AUROC={auroc:.3f}  AUPRC={auprc:.3f}")

    # save metrics JSON
    mdir = Path("metrics"); mdir.mkdir(parents=True, exist_ok=True)
    out_json = mdir / f"anom_gru_{Path(args.ckpt).stem}.json"
    out_json.write_text(json.dumps({"task":"anomaly","model":"gru","ckpt":str(args.ckpt),
                                    "auroc": float(auroc), "auprc": float(auprc)}, indent=2))
    print(f"[metrics] wrote {out_json}")

if __name__ == "__main__":
    main()