# src/models/tptrans_V3.py
from __future__ import annotations
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Create constant 'pe' matrix with values dependent on pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # [1, max_len, d_model]

    def forward(self, x):
        # x: [Batch, SeqLen, D_Model]
        # Add positional encoding to the input embeddings
        return x + self.pe[:, :x.size(1), :]

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

    def forward(self, x):  
        return self.net(x)

class TPTrans(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        d_model: int = 512,
        nhead: int = 8,
        enc_layers: int = 4,
        dec_layers: int = 4, # Increased for better reasoning
        horizon: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizon = horizon
        self.d_model = d_model

        # 1. Input Processing
        self.conv = ConvBlock(feat_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # 2. Transformer Encoder (Process History)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)

        # 3. Transformer Decoder (Predict Future)
        # Replaces the GRU. Cross-Attention allows looking at specific parts of history.
        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)

        # 4. Query Generator
        # Instead of "Zeros", we learn a "Query" for each future timestep (t+1, t+2...)
        # This tells the model "Predict what happens at Time Step X"
        self.query_embed = nn.Embedding(horizon, d_model)

        # 5. Output Head
        self.proj = nn.Linear(d_model, 2)

    def forward(self, x):  # x: [B, Past_Len, Feat_Dim]
        B, T, F = x.shape
        
        # --- ENCODE PAST ---
        # 1. Conv Features
        x_emb = self.conv(x.transpose(1, 2)).transpose(1, 2) # [B, T, D]
        
        # 2. Add Time Awareness (Positional Encoding)
        x_emb = self.pos_encoder(x_emb)
        
        # 3. Transformer Memory
        memory = self.encoder(x_emb) # [B, T, D]

        # --- DECODE FUTURE ---
        # 4. Create Queries for the Horizon
        # We ask for positions 0 to Horizon-1
        query_idx = torch.arange(self.horizon, device=x.device).unsqueeze(0).repeat(B, 1) # [B, Horizon]
        tgt = self.query_embed(query_idx) # [B, Horizon, D]
        
        # 5. Add Positional Encoding to Queries too (Crucial for sequence order)
        tgt = self.pos_encoder(tgt)

        # 6. Transformer Decoding
        # The decoder uses 'tgt' to query the 'memory' (Cross-Attention)
        out = self.decoder(tgt, memory) # [B, Horizon, D]
        
        # 7. Project to Deltas
        deltas = self.proj(out) # [B, Horizon, 2]
        
        return deltas