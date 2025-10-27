from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import GroupKFold

from ..models import GRUSeq2Seq, TPTrans
from ..eval.metrics_traj import ade, fde


def build_model(model_name: str, feat_dim: int, horizon: int, hp: Dict[str, Any]):
    if model_name == "gru":
        return GRUSeq2Seq(
            feat_dim=feat_dim,
            d_model=int(hp.get("d_model", 128)),
            layers=int(hp.get("layers", 2)),
            horizon=horizon,
        )
    elif model_name == "tptrans":
        return TPTrans(
            feat_dim=feat_dim,
            d_model=int(hp.get("d_model", 192)),
            nhead=int(hp.get("nhead", 4)),
            enc_layers=int(hp.get("enc_layers", 4)),
            dec_layers=int(hp.get("dec_layers", 2)),
            horizon=horizon,
        )
    else:
        raise ValueError(f"unknown model {model_name}")


def huber_loss(delta: float = 1.0):
    return nn.SmoothL1Loss(beta=delta)


def standardize_targets(pred: torch.Tensor, target: torch.Tensor,
                        y_mean_t: torch.Tensor | None, y_std_t: torch.Tensor | None):
    if y_mean_t is None or y_std_t is None:
        return pred, target
    return (pred - y_mean_t) / y_std_t, (target - y_mean_t) / y_std_t


def train_one(
    model, ds_train: TensorDataset, ds_val: TensorDataset | None,
    y_mean_t: torch.Tensor | None, y_std_t: torch.Tensor | None,
    hp: Dict[str, Any], device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """Train for a small number of epochs; return best val loss (or train loss if no val)."""
    bs = int(hp.get("batch_size", 128))
    epochs = int(hp.get("epochs", 10))
    lr = float(hp.get("lr", 1e-2))
    momentum = float(hp.get("momentum", 0.9))
    weight_decay = float(hp.get("weight_decay", 0.0))
    nesterov = bool(hp.get("nesterov", True))
    clip = float(hp.get("clip_norm", 1.0))
    delta = float(hp.get("huber_delta", 1.0))

    dl_train = DataLoader(ds_train, batch_size=bs, shuffle=True, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=bs, shuffle=False, pin_memory=True) if ds_val else None

    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    loss_fn = huber_loss(delta)

    best = math.inf
    for _epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in dl_train:
            xb = xb.float().to(device, non_blocking=True)
            yb = yb.float().to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pred = model(xb)
                pred_s, yb_s = standardize_targets(pred, yb, y_mean_t, y_std_t)
                loss = loss_fn(pred_s, yb_s)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler_amp.step(opt)
            scaler_amp.update()
            total += loss.item() * xb.size(0)

        # val
        cur = total / len(ds_train)
        if dl_val:
            model.eval()
            vtot = 0.0
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                for xb, yb in dl_val:
                    xb = xb.float().to(device, non_blocking=True)
                    yb = yb.float().to(device, non_blocking=True)
                    out = model(xb)
                    out_s, yb_s = standardize_targets(out, yb, y_mean_t, y_std_t)
                    vtot += loss_fn(out_s, yb_s).item() * xb.size(0)
            cur = vtot / len(ds_val)
        best = min(best, cur)
    return best, {"best_loss": best}


def predict(model, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(X).float().to(device)).cpu().numpy()
    return out


def nested_cv(
    X: np.ndarray, Y: np.ndarray, groups: np.ndarray,
    model_name: str, y_scaler: dict | None,
    outer_folds: int, inner_folds: int,
    search_space: List[Dict[str, Any]],
    device: torch.device
):
    """
    Outer CV for test estimates; inner CV for hyperparameter selection.
    """
    y_mean_t = y_std_t = None
    if y_scaler is not None:
        y_mean_t = torch.from_numpy(y_scaler["mean"].astype("float32")).view(1, 1, -1).to(device)
        y_std_t = torch.from_numpy(y_scaler["std"].astype("float32")).view(1, 1, -1).to(device)

    feat_dim, horizon = X.shape[-1], Y.shape[1]
    outer = GroupKFold(n_splits=outer_folds)
    results = []

    for k, (train_idx, test_idx) in enumerate(outer.split(X, groups=groups), 1):
        # ----- inner CV for tuning -----
        inner = GroupKFold(n_splits=inner_folds)
        best_hp, best_score = None, math.inf
        g_train = groups[train_idx]

        for hp in search_space:
            scores = []
            for tr_i, val_i in inner.split(train_idx, groups=g_train):
                tr_idx = train_idx[tr_i]
                va_idx = train_idx[val_i]

                ds_tr = TensorDataset(torch.from_numpy(X[tr_idx]).float(),
                                      torch.from_numpy(Y[tr_idx]).float())
                ds_va = TensorDataset(torch.from_numpy(X[va_idx]).float(),
                                      torch.from_numpy(Y[va_idx]).float())

                model = build_model(model_name, feat_dim, horizon, hp)
                score, _ = train_one(model, ds_tr, ds_va, y_mean_t, y_std_t, hp, device)
                scores.append(score)
            mean_score = float(np.mean(scores))
            if mean_score < best_score:
                best_score, best_hp = mean_score, hp

        # ----- retrain on full outer-train with best hp -----
        ds_tr_full = TensorDataset(torch.from_numpy(X[train_idx]).float(),
                                   torch.from_numpy(Y[train_idx]).float())
        model = build_model(model_name, feat_dim, horizon, best_hp)
        _score, _ = train_one(model, ds_tr_full, None, y_mean_t, y_std_t, best_hp, device)

        # ----- evaluate on outer test -----
        pred = predict(model, X[test_idx], device)
        # unstandardize predictions if target scaler exists
        if y_scaler is not None:
            pred = pred * y_scaler["std"].reshape(1, 1, -1) + y_scaler["mean"].reshape(1, 1, -1)
        ade_v = ade(pred, Y[test_idx])
        fde_v = fde(pred, Y[test_idx])
        results.append({"fold": k, "ade": float(ade_v), "fde": float(fde_v), "best_hp": best_hp})

        print(f"[outer {k}/{outer_folds}] ADE={ade_v:.3f} FDE={fde_v:.3f}  best_hp={best_hp}")

    # summary
    ade_mean = float(np.mean([r["ade"] for r in results]))
    ade_std = float(np.std([r["ade"] for r in results]))
    fde_mean = float(np.mean([r["fde"] for r in results]))
    fde_std = float(np.std([r["fde"] for r in results]))
    print(f"\n=== Nested CV ({model_name}) ===")
    print(f"ADE: {ade_mean:.3f} ± {ade_std:.3f}   FDE: {fde_mean:.3f} ± {fde_std:.3f}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed/traj_w64_h12")
    ap.add_argument("--model", choices=["gru", "tptrans"], default="tptrans")
    ap.add_argument("--outer_folds", type=int, default=5)
    ap.add_argument("--inner_folds", type=int, default=3)
    ap.add_argument("--max_trials", type=int, default=8, help="number of hyperparameter configurations to try")
    args = ap.parse_args()

    proc = Path(args.processed_dir)
    X = np.load(proc / "X.npy")
    Y = np.load(proc / "Y.npy")
    groups = np.load(proc / "window_mmsi.npy") if (proc / "window_mmsi.npy").exists() else np.arange(len(X))

    # input scaler (already applied in training script; not needed here)
    # target scaler for unscaling preds
    y_scaler = None
    tsc = proc / "target_scaler.npz"
    if tsc.exists():
        ts = np.load(tsc)
        y_scaler = {"mean": ts["mean"], "std": ts["std"]}

    # simple random search space (tweak as needed)
    rng = np.random.default_rng(42)
    def sample_hps(n: int):
        hps = []
        for _ in range(n):
            hp = {
                "epochs": int(rng.integers(8, 16)),          # short epochs for tuning
                "batch_size": int(rng.choice([64, 96, 128])),
                "lr": float(rng.choice([5e-3, 1e-2, 2e-2])),
                "momentum": float(rng.choice([0.8, 0.9, 0.95])),
                "nesterov": True,
                "weight_decay": float(rng.choice([0.0, 1e-4, 5e-4])),
                "clip_norm": 1.0,
                "huber_delta": 1.0,
            }
            if args.model == "gru":
                hp.update({
                    "d_model": int(rng.choice([128, 192])),
                    "layers": int(rng.choice([2, 3])),
                })
            else:
                hp.update({
                    "d_model": int(rng.choice([192, 256])),
                    "nhead": int(rng.choice([4, 8])),
                    "enc_layers": int(rng.choice([3, 4])),
                    "dec_layers": int(rng.choice([2, 3])),
                })
            hps.append(hp)
        return hps

    search_space = sample_hps(args.max_trials)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nested_cv(X, Y, groups, args.model, y_scaler, args.outer_folds, args.inner_folds, search_space, device)


if __name__ == "__main__":
    main()