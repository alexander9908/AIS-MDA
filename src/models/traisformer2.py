# src/models/traisformer2.py
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Utilities: bins & four-hot I/O
# -----------------------------

@dataclass
class BinSpec:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    sog_max: float     # knots
    n_lat: int         # e.g., 256
    n_lon: int
    n_sog: int         # e.g., 50 for 0..50 kn
    n_cog: int         # e.g., 72 for 5° bins

    @property
    def d_total(self) -> int:
        return self.n_lat + self.n_lon + self.n_sog + self.n_cog

    def to_dict(self) -> Dict[str, int]:
        return dict(n_lat=self.n_lat, n_lon=self.n_lon, n_sog=self.n_sog, n_cog=self.n_cog)

    # --- binning helpers ---
    def _bin_uniform(self, x: torch.Tensor, lo: float, hi: float, n: int) -> torch.Tensor:
        x = x.clamp(min=lo, max=hi)
        idx = ((x - lo) / (hi - lo + 1e-9) * n).floor().long()
        return idx.clamp(0, n - 1)

    def lat_to_bin(self, lat_deg: torch.Tensor) -> torch.Tensor:
        return self._bin_uniform(lat_deg, self.lat_min, self.lat_max, self.n_lat)

    def lon_to_bin(self, lon_deg: torch.Tensor) -> torch.Tensor:
        return self._bin_uniform(lon_deg, self.lon_min, self.lon_max, self.n_lon)

    def sog_to_bin(self, sog_kn: torch.Tensor) -> torch.Tensor:
        return self._bin_uniform(sog_kn, 0.0, self.sog_max, self.n_sog)

    def cog_to_bin(self, cog_deg: torch.Tensor) -> torch.Tensor:
        c = (cog_deg % 360.0 + 360.0) % 360.0
        return self._bin_uniform(c, 0.0, 360.0, self.n_cog)

    def bin_to_lat_mid(self, idx: torch.Tensor) -> torch.Tensor:
        step = (self.lat_max - self.lat_min) / self.n_lat
        return self.lat_min + (idx.float() + 0.5) * step

    def bin_to_lon_mid(self, idx: torch.Tensor) -> torch.Tensor:
        step = (self.lon_max - self.lon_min) / self.n_lon
        return self.lon_min + (idx.float() + 0.5) * step

    def bin_to_sog_mid(self, idx: torch.Tensor) -> torch.Tensor:
        step = (self.sog_max - 0.0) / self.n_sog
        return 0.0 + (idx.float() + 0.5) * step

    def bin_to_cog_mid(self, idx: torch.Tensor) -> torch.Tensor:
        step = 360.0 / self.n_cog
        return (idx.float() + 0.5) * step


def coarse_indices(idx: torch.Tensor, merge: int) -> torch.Tensor:
    return (idx // merge).long()


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


class TrAISformer(nn.Module):
    def __init__(
        self,
        bins: BinSpec,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 8,
        dropout: float = 0.1,
        emb_lat: int = 256,
        emb_lon: int = 256,
        emb_sog: int = 128,
        emb_cog: int = 128,
        coarse_merge: int = 3,
        coarse_beta: float = 0.2,
        use_water_mask: bool = False, # Default to False to prevent crash
    ):
        super().__init__()
        self.bins = bins
        self.coarse_merge = coarse_merge
        self.coarse_beta = coarse_beta

        # Embeddings
        self.lat_emb = nn.Embedding(bins.n_lat, emb_lat)
        self.lon_emb = nn.Embedding(bins.n_lon, emb_lon)
        self.sog_emb = nn.Embedding(bins.n_sog, emb_sog)
        self.cog_emb = nn.Embedding(bins.n_cog, emb_cog)

        d_in = emb_lat + emb_lon + emb_sog + emb_cog
        self.in_proj = nn.Linear(d_in, d_model)

        # Transformer decoder (causal)
        self.posenc = SinusoidalPositionalEncoding(d_model)
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=4 * d_model,
                                               dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        # 4 heads
        self.head_lat = nn.Linear(d_model, bins.n_lat)
        self.head_lon = nn.Linear(d_model, bins.n_lon)
        self.head_sog = nn.Linear(d_model, bins.n_sog)
        self.head_cog = nn.Linear(d_model, bins.n_cog)

        # Learnable start token
        self.start_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.start_token, std=0.02)

    def _embed_step(self, idxs: Dict[str, torch.Tensor]) -> torch.Tensor:
        dev = self.start_token.device
        def _bt_idx(x: torch.Tensor) -> torch.Tensor:
            x = torch.as_tensor(x, device=dev)
            if x.dim() == 0: x = x.view(1, 1)
            elif x.dim() == 1: x = x.unsqueeze(1)
            elif x.dim() > 2: x = x.squeeze()
            return x.long().contiguous()

        e_lat = self.lat_emb(_bt_idx(idxs["lat"]))
        e_lon = self.lon_emb(_bt_idx(idxs["lon"]))
        e_sog = self.sog_emb(_bt_idx(idxs["sog"]))
        e_cog = self.cog_emb(_bt_idx(idxs["cog"]))
        return torch.cat([e_lat, e_lon, e_sog, e_cog], dim=-1)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

    def forward(self, past_idxs, future_idxs):
        # Training forward pass (Teacher Forcing)
        e_past = self._embed_step(past_idxs)
        z_past = self.in_proj(e_past)
        z_past = self.posenc(z_past)

        e_future_in = self._embed_step(future_idxs)
        z_future_in = self.in_proj(e_future_in)
        z_future_in = torch.cat([self.start_token.expand(z_future_in.size(0), 1, -1),
                                 z_future_in[:, :-1, :]], dim=1)
        z_future_in = self.posenc(z_future_in)

        tgt_mask = self._causal_mask(z_future_in.size(1), next(self.parameters()).device)
        dec = self.decoder(tgt=z_future_in, memory=z_past, tgt_mask=tgt_mask)

        logits = {
            "lat": self.head_lat(dec),
            "lon": self.head_lon(dec),
            "sog": self.head_sog(dec),
            "cog": self.head_cog(dec),
        }
        return logits

    @torch.no_grad()
    def generate(
        self,
        past_idxs: Dict[str, torch.Tensor],
        L: int,
        sampling: str = "sample",
        temperature: float = 1.0,
        top_k: int = 20,
    ) -> Dict[str, torch.Tensor]:
        """
        Simple autoregressive generation loop.
        """
        self.eval()
        device = next(self.parameters()).device

        # Encode past
        e_past = self._embed_step(past_idxs)
        z_past = self.in_proj(e_past)
        z_past = self.posenc(z_past)

        B = z_past.size(0)
        y_seq = self.start_token.expand(B, 1, -1)
        
        # Lists to store predictions
        out = {k: [] for k in ["lat", "lon", "sog", "cog"]}

        def _sample(logits):
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            if sampling == "greedy" or temperature == 0:
                return torch.argmax(logits, dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                return torch.multinomial(probs, 1).squeeze(-1)

        for _ in range(L):
            tgt_mask = self._causal_mask(y_seq.size(1), device)
            dec = self.decoder(tgt=self.posenc(y_seq), memory=z_past, tgt_mask=tgt_mask)
            
            # Predict next token from last hidden state
            h = dec[:, -1, :] 
            
            next_lat = _sample(self.head_lat(h))
            next_lon = _sample(self.head_lon(h))
            next_sog = _sample(self.head_sog(h))
            next_cog = _sample(self.head_cog(h))

            # Store
            out["lat"].append(next_lat)
            out["lon"].append(next_lon)
            out["sog"].append(next_sog)
            out["cog"].append(next_cog)

            # Embed for next step input
            next_step_input = {
                "lat": next_lat.unsqueeze(1),
                "lon": next_lon.unsqueeze(1),
                "sog": next_sog.unsqueeze(1),
                "cog": next_cog.unsqueeze(1)
            }
            
            e_next = self._embed_step(next_step_input)
            z_next = self.in_proj(e_next)
            y_seq = torch.cat([y_seq, z_next], dim=1)

        # Stack lists into tensors [B, L]
        for k in out:
            out[k] = torch.stack(out[k], dim=1)
            
        return out