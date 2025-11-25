# src/models/tptrans_V2.py
from __future__ import annotations
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, d=1):
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
        d_model: int = 512,
        nhead: int = 8,
        enc_layers: int = 4,
        dec_layers: int = 2,
        horizon: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizon = horizon
        self.dec_layers = dec_layers
        self.d_model = d_model

        # Encoder: 1D Conv -> Transformer
        self.conv = ConvBlock(feat_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers)

        # Decoder: GRU predicting DELTAS
        self.dec = nn.GRU(
            input_size=d_model, 
            hidden_size=d_model,
            num_layers=dec_layers,
            batch_first=True,
            dropout=dropout if dec_layers > 1 else 0.0,
        )

        # Output projection: Hidden -> (delta_lat, delta_lon)
        self.proj = nn.Linear(d_model, 2)

    def forward(self, x):  # x: [B, T, F]
        # 1. Encode Past
        # Permute for Conv: [B, T, F] -> [B, F, T]
        x_emb = self.conv(x.transpose(1, 2)).transpose(1, 2)  # [B, T, D]
        enc_out = self.encoder(x_emb)  # [B, T, D]

        # 2. Prepare Context for Decoder
        # Take the last encoded state as the "thought vector"
        last_hidden = enc_out[:, -1, :].unsqueeze(0)  # [1, B, D]
        # Replicate for all GRU layers
        context = last_hidden.repeat(self.dec_layers, 1, 1)  # [Layers, B, D]

        # 3. Decode
        # We feed ZEROS as input, but the model learns that 
        # "Zero Input + Context" = "Predicted Velocity" (Delta)
        dec_in = torch.zeros(x.size(0), self.horizon, self.d_model, device=x.device)
        
        out, _ = self.dec(dec_in, context)  # [B, Horizon, D]
        deltas = self.proj(out)             # [B, Horizon, 2]
        
        return deltas