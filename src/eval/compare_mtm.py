"""Compare MTM-pretrained vs. baseline trajectory prediction models."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.datasets import AISDataset
from src.models import TPTrans
from src.eval.metrics_traj import ade, fde


def evaluate_model(model, dataloader, device):
    """Evaluate a trajectory prediction model."""
    model.eval()
    all_ade = []
    all_fde = []
    
    with torch.no_grad():
        for xb, yb in tqdm(dataloader, desc="Evaluating", leave=False):
            xb = xb.to(device)
            yb = yb.to(device)
            
            pred = model(xb)  # [B, H, 2]
            
            # Compute metrics
            batch_ade = ade(pred, yb)
            batch_fde = fde(pred, yb)
            
            all_ade.append(batch_ade)
            all_fde.append(batch_fde)
    
    metrics = {
        'ade_mean': float(np.mean(all_ade)),
        'ade_std': float(np.std(all_ade)),
        'fde_mean': float(np.mean(all_fde)),
        'fde_std': float(np.std(all_fde)),
    }
    
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Compare MTM-pretrained vs baseline trajectory models")
    ap.add_argument("--processed_dir", default="data/processed/traj_w64_h12/")
    ap.add_argument("--ckpt_baseline", default="data/checkpoints/traj_tptrans_baseline.pt", 
                   help="Checkpoint for baseline (no pretraining)")
    ap.add_argument("--ckpt_mtm", default="data/checkpoints/traj_tptrans.pt",
                   help="Checkpoint for MTM-pretrained model")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--d_model", type=int, default=192)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--enc_layers", type=int, default=4)
    ap.add_argument("--dec_layers", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--out_dir", default="metrics")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load validation dataset
    print(f"Loading validation dataset from {args.processed_dir}")
    val_dir = Path(args.processed_dir) / "val"
    if not val_dir.exists():
        print(f"Warning: {val_dir} not found, using test set")
        val_dir = Path(args.processed_dir) / "test"
    
    ds_val = AISDataset(str(val_dir), max_seqlen=96)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    feat_dim = ds_val[0][0].shape[-1]
    print(f"Dataset: {len(ds_val)} samples, feature_dim={feat_dim}")
    
    results = {}
    
    # Evaluate baseline model (if exists)
    if Path(args.ckpt_baseline).exists():
        print("\n" + "="*60)
        print("Evaluating BASELINE model (no pretraining)")
        print("="*60)
        model_baseline = TPTrans(
            feat_dim=feat_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            enc_layers=args.enc_layers,
            dec_layers=args.dec_layers,
            horizon=args.horizon
        ).to(device)
        model_baseline.load_state_dict(torch.load(args.ckpt_baseline, map_location=device))
        
        metrics_baseline = evaluate_model(model_baseline, dl_val, device)
        results['baseline'] = metrics_baseline
        
        print(f"\nBaseline Results:")
        print(f"  ADE: {metrics_baseline['ade_mean']:.4f} ± {metrics_baseline['ade_std']:.4f}")
        print(f"  FDE: {metrics_baseline['fde_mean']:.4f} ± {metrics_baseline['fde_std']:.4f}")
    else:
        print(f"\nWarning: Baseline checkpoint not found at {args.ckpt_baseline}")
        print("Skipping baseline evaluation")
    
    # Evaluate MTM-pretrained model
    if Path(args.ckpt_mtm).exists():
        print("\n" + "="*60)
        print("Evaluating MTM-PRETRAINED model")
        print("="*60)
        model_mtm = TPTrans(
            feat_dim=feat_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            enc_layers=args.enc_layers,
            dec_layers=args.dec_layers,
            horizon=args.horizon
        ).to(device)
        model_mtm.load_state_dict(torch.load(args.ckpt_mtm, map_location=device))
        
        metrics_mtm = evaluate_model(model_mtm, dl_val, device)
        results['mtm_pretrained'] = metrics_mtm
        
        print(f"\nMTM-Pretrained Results:")
        print(f"  ADE: {metrics_mtm['ade_mean']:.4f} ± {metrics_mtm['ade_std']:.4f}")
        print(f"  FDE: {metrics_mtm['fde_mean']:.4f} ± {metrics_mtm['fde_std']:.4f}")
    else:
        print(f"\nError: MTM checkpoint not found at {args.ckpt_mtm}")
        return
    
    # Compare results
    if 'baseline' in results and 'mtm_pretrained' in results:
        print("\n" + "="*60)
        print("COMPARISON")
        print("="*60)
        baseline_ade = results['baseline']['ade_mean']
        mtm_ade = results['mtm_pretrained']['ade_mean']
        baseline_fde = results['baseline']['fde_mean']
        mtm_fde = results['mtm_pretrained']['fde_mean']
        
        ade_improvement = ((baseline_ade - mtm_ade) / baseline_ade) * 100
        fde_improvement = ((baseline_fde - mtm_fde) / baseline_fde) * 100
        
        print(f"ADE improvement: {ade_improvement:+.2f}%")
        print(f"FDE improvement: {fde_improvement:+.2f}%")
        
        results['comparison'] = {
            'ade_improvement_pct': float(ade_improvement),
            'fde_improvement_pct': float(fde_improvement)
        }
    
    # Save results
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mtm_comparison.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
