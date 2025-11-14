# src/models/tptrans_unsup_new.py
from __future__ import annotations
import torch
import torch.nn as nn
from .tptrans_new import ConvBlock, PositionalEncoding

class TPTransMSPNew(nn.Module):
    """
    Same encoder stack as TPTransNew, but with a reconstruction head for masked-step prediction (MSP).
    """
    def __init__(self, feat_dim: int, d_model: int = 192, nhead: int = 4, enc_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.conv = ConvBlock(feat_dim, d_model)
        self.posenc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True,
                                               dropout=dropout, dim_feedforward=4*d_model,
                                               activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)
        self.recon = nn.Linear(d_model, feat_dim)

    def forward(self, x):  # [B,T,F]
        h = self.conv(x.transpose(1,2)).transpose(1,2)
        h = self.posenc(h)
        z = self.encoder(h)
        return self.recon(z)  # [B,T,F]
