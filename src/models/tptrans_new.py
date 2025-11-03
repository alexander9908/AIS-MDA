from __future__ import annotations
import math
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, d: int = 1):
        super().__init__()
        pad = (k - 1) // 2 * d
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, dilation=d, padding=pad),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=k, dilation=d, padding=pad),
            nn.ReLU(),
        )
    def forward(self, x):  # [B,C,T]
        return self.net(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1,L,D]
    def forward(self, x):  # [B,T,D]
        return x + self.pe[:, :x.size(1), :]

class TPTransNew(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        d_model: int = 192,
        nhead: int = 4,
        enc_layers: int = 4,
        dec_layers: int = 2,
        horizon: int = 12,
        dropout: float = 0.1,
        use_posenc: bool = True,
        dim_ff: int = 2048,   # <-- make FFN size configurable; default 2048 to match old ckpts
    ):
        super().__init__()
        self.horizon = horizon
        self.d_model = d_model
        self.dec_layers = dec_layers

        self.conv = ConvBlock(feat_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,    # <-- use configured FFN size
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)
        self.posenc = PositionalEncoding(d_model) if use_posenc else nn.Identity()

        self.dec = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=dec_layers,
            batch_first=True,
            dropout=dropout if dec_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(d_model, 2)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):  # [B,T,F]
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)   # [B,T,D]
        h = self.posenc(h)
        enc = self.encoder(h)
        context = enc[:, -1, :].unsqueeze(0).repeat(self.dec_layers, 1, 1)  # [L,B,D]
        dec_in = torch.zeros(x.size(0), self.horizon, self.d_model, device=x.device, dtype=x.dtype)
        out, _ = self.dec(dec_in, context)
        return self.proj(out)  # [B,H,2]
