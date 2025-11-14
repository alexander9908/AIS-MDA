# src/models/tptrans.py
from __future__ import annotations
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k = 3, d = 1):#  k=3, d=1):
        super().__init__()
        pad = (k - 1) // 2 * d
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, dilation=d, padding=pad),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=k, dilation=d, padding=pad),
            nn.ReLU(),
        )

    def forward(self, x):  # x: [B, C, T]
        return self.net(x)


class TPTrans(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        d_model: int = 512, # int = 192,
        nhead: int = 4,
        enc_layers: int = 8, # int = 4,
        dec_layers: int = 4,# int = 2,
        horizon: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizon = horizon
        self.dec_layers = dec_layers
        self.d_model = d_model

        # Local convolutional encoder
        self.conv = ConvBlock(feat_dim, d_model)

        # Global Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers)

        # Temporal decoder (GRU)
        self.dec = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=dec_layers,
            batch_first=True,
            dropout=dropout if dec_layers > 1 else 0.0,
        )

        # Output projection to Δlat, Δlon
        self.proj = nn.Linear(d_model, 2)

    def forward(self, x):  # x: [B, T, F]
        # CNN expects [B, C, T]
        x1 = self.conv(x.transpose(1, 2)).transpose(1, 2)  # -> [B, T, D]
        h = self.encoder(x1)  # -> [B, T, D]

        # Use the last encoder step as context, repeat for each decoder layer
        last = h[:, -1, :].unsqueeze(0)  # [1, B, D]
        context = last.repeat(self.dec_layers, 1, 1)  # [L, B, D]

        # Zero input to decoder (no teacher forcing)
        dec_in = torch.zeros(x.size(0), self.horizon, self.d_model, device=x.device)

        out, _ = self.dec(dec_in, context)  # [B, H, D]
        y = self.proj(out)  # [B, H, 2]
        return y

#class ConvBlock(nn.Module):
#    def __init__(self, in_ch, out_ch, k=3, d=1):
#        super().__init__()
#        pad = (k-1)//2 * d
#        self.net = nn.Sequential(
#            nn.Conv1d(in_ch, out_ch, kernel_size=k, dilation=d, padding=pad),
#            nn.ReLU(),
#            nn.Conv1d(out_ch, out_ch, kernel_size=k, dilation=d, padding=pad),
#            nn.ReLU(),
#        )
#    def forward(self, x):  # x: [B,C,T]
#        return self.net(x)
#
#class TPTrans(nn.Module):
#    """CNN + Transformer encoder + GRU decoder (TPTrans-style).
#    - CNN on features along time (local patterns/turns)
#    - Transformer encoder for long-range context
#    - GRU decoder to emit future deltas
#    """
#    def __init__(self, feat_dim: int, d_model: int = 192, nhead: int = 4, enc_layers: int = 4, dec_layers: int = 2, horizon: int = 12, dropout: float = 0.1):
#        super().__init__()
#        self.horizon = horizon
#        self.conv = ConvBlock(feat_dim, d_model)
#        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
#        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers)
#        self.dec = nn.GRU(d_model, d_model, num_layers=dec_layers, batch_first=True, dropout=dropout if dec_layers>1 else 0.0)
#        self.proj = nn.Linear(d_model, 2)
#
#    def forward(self, x):  # x: [B,T,F]
#        # CNN expects [B,C,T]
#        x1 = self.conv(x.transpose(1,2)).transpose(1,2)  # -> [B,T,D]
#        h = self.encoder(x1)  # [B,T,D]
#        # Use last state as context
#        context = h[:, -1:, :].transpose(0,1)  # [1,B,D]
#        dec_in = torch.zeros(x.size(0), self.horizon, h.size(-1), device=x.device)
#        out, _ = self.dec(dec_in, context)
#        y = self.proj(out)  # [B,H,2]
#        return y


