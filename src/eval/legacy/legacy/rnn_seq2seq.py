from __future__ import annotations
import torch
import torch.nn as nn

class GRUSeq2Seq(nn.Module):
    """Minimal GRU encoder-decoder for trajectory deltas."""
    def __init__(self, feat_dim: int, d_model: int = 128, layers: int = 2, horizon: int = 12, dropout: float = 0.1):
        super().__init__()
        self.horizon = horizon
        self.enc = nn.GRU(feat_dim, d_model, num_layers=layers, batch_first=True, dropout=dropout if layers>1 else 0.0)
        self.dec = nn.GRU(2, d_model, num_layers=layers, batch_first=True, dropout=dropout if layers>1 else 0.0)
        self.proj = nn.Linear(d_model, 2)

    def forward(self, x):  # x: [B,T,F]
        B = x.size(0)
        _, h = self.enc(x)  # h: [L,B,D]
        dec_in = torch.zeros(B, self.horizon, 2, device=x.device)
        out, _ = self.dec(dec_in, h)
        y = self.proj(out)  # [B,H,2]
        return y
