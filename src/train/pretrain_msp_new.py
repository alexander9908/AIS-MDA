# src/train/pretrain_msp_new.py
from __future__ import annotations
import argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from src.utils.datasets import AISDataset
from ..models.legacy.tptrans_unsup_new import TPTransMSPNew

def make_mask(x, p=0.2):
    B,T,F = x.shape
    m = torch.rand(B,T, device=x.device) < p
    x_masked = x.clone()
    x_masked[m] = 0.0  # zero-out masked steps (simple/fast)
    return x_masked, m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True, help="data/map_reduced/train")
    ap.add_argument("--out", default="data/checkpoints/pretrain_msp.pt")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--mask_p", type=float, default=0.2)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    ds = AISDataset(args.split_dir, max_seqlen=96, return_labels=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    x0 = next(iter(dl))
    feat_dim = x0.shape[-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TPTransMSPNew(feat_dim=feat_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.SmoothL1Loss()

    model.train()
    for epoch in range(1, args.epochs+1):
        total=0.0; seen=0
        for xb in dl:
            xb = xb.float().to(device)
            x_masked, mask = make_mask(xb, p=args.mask_p)
            rec = model(x_masked)
            if mask.any():
                loss = loss_fn(rec[mask], xb[mask])
            else:
                loss = (rec.sum()*0.0)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss)*xb.size(0); seen += xb.size(0)
        print(f"epoch {epoch}: recon_loss={total/max(1,seen):.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"[pretrain] saved {args.out}")

if __name__ == "__main__":
    main()
