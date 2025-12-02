# src/models/traisformer1.py
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Literal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import the mask builder
try:
    from src.eval.build_water_mask_V2 import make_water_mask
    _HAS_WATER_MASK = True
except ImportError:
    try:
        from src.eval.build_water_mask import make_water_mask
        _HAS_WATER_MASK = True
    except ImportError:
        _HAS_WATER_MASK = False

@dataclass
class BinSpec:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    sog_max: float
    n_lat: int
    n_lon: int
    n_sog: int
    n_cog: int

    def to_dict(self):
        return self.__dict__

    def _bin(self, x, lo, hi, n):
        x = x.clamp(min=lo, max=hi)
        idx = ((x - lo) / (hi - lo + 1e-9) * n).floor().long()
        return idx.clamp(0, n - 1)

    def lat_to_bin(self, x): return self._bin(x, self.lat_min, self.lat_max, self.n_lat)
    def lon_to_bin(self, x): return self._bin(x, self.lon_min, self.lon_max, self.n_lon)
    def sog_to_bin(self, x): return self._bin(x, 0.0, self.sog_max, self.n_sog)
    def cog_to_bin(self, x): 
        c = (x % 360.0 + 360.0) % 360.0
        return self._bin(c, 0.0, 360.0, self.n_cog)

    def bin_to_lat_mid(self, idx):
        return self.lat_min + (idx.float() + 0.5) * (self.lat_max - self.lat_min) / self.n_lat
    def bin_to_lon_mid(self, idx):
        return self.lon_min + (idx.float() + 0.5) * (self.lon_max - self.lon_min) / self.n_lon

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0)

class TrAISformer(nn.Module):
    def __init__(self, bins: BinSpec, d_model=512, nhead=8, num_layers=6, dropout=0.1,
                 emb_lat=128, emb_lon=128, emb_sog=64, emb_cog=64, use_water_mask=False, **kwargs):
        super().__init__()
        self.bins = bins
        self.use_water_mask = use_water_mask
        
        self.lat_emb = nn.Embedding(bins.n_lat, emb_lat)
        self.lon_emb = nn.Embedding(bins.n_lon, emb_lon)
        self.sog_emb = nn.Embedding(bins.n_sog, emb_sog)
        self.cog_emb = nn.Embedding(bins.n_cog, emb_cog)

        d_in = emb_lat + emb_lon + emb_sog + emb_cog
        self.in_proj = nn.Linear(d_in, d_model)
        self.posenc = SinusoidalPositionalEncoding(d_model)
        
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=4*d_model, 
                                               dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        self.head_lat = nn.Linear(d_model, bins.n_lat)
        self.head_lon = nn.Linear(d_model, bins.n_lon)
        self.head_sog = nn.Linear(d_model, bins.n_sog)
        self.head_cog = nn.Linear(d_model, bins.n_cog)

        self.start_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.start_token, std=0.02)

        # --- WATER MASK INIT ---
        if use_water_mask and _HAS_WATER_MASK:
            wm = make_water_mask(bins.lat_min, bins.lat_max, bins.lon_min, bins.lon_max, bins.n_lat, bins.n_lon)
            self.register_buffer("water_mask", torch.tensor(wm, dtype=torch.bool), persistent=False)
        else:
            self.water_mask = None

    def _embed(self, idxs):
        dev = self.start_token.device
        def _L(x): return x.long().to(dev).contiguous()
        e = torch.cat([
            self.lat_emb(_L(idxs["lat"])),
            self.lon_emb(_L(idxs["lon"])),
            self.sog_emb(_L(idxs["sog"])),
            self.cog_emb(_L(idxs["cog"]))
        ], dim=-1)
        return e

    def forward(self, past_idxs, future_idxs):
        e_past = self._embed(past_idxs)
        z_past = self.posenc(self.in_proj(e_past))

        e_fut = self._embed(future_idxs)
        z_fut = self.in_proj(e_fut)
        
        B = z_fut.size(0)
        start = self.start_token.expand(B, 1, -1)
        decoder_input = torch.cat([start, z_fut[:, :-1, :]], dim=1) 
        decoder_input = self.posenc(decoder_input)

        tgt_mask = torch.triu(torch.ones(decoder_input.size(1), decoder_input.size(1), 
                                         device=decoder_input.device), diagonal=1).bool()
        
        out = self.decoder(tgt=decoder_input, memory=z_past, tgt_mask=tgt_mask)

        return {
            "lat": self.head_lat(out),
            "lon": self.head_lon(out),
            "sog": self.head_sog(out),
            "cog": self.head_cog(out)
        }

    def compute_loss(self, logits, targets):
        loss = 0.0
        for key in ["lat", "lon", "sog", "cog"]:
            l = logits[key].reshape(-1, logits[key].size(-1))
            t = targets[key].reshape(-1).long()
            loss += F.cross_entropy(l, t)
        return loss

    @torch.no_grad()
    def generate(self, past_idxs, L, sampling="sample", temperature=1.0, top_k=20, local_window=None, prevent_stuck=False):
        self.eval()
        device = next(self.parameters()).device
        e_past = self._embed(past_idxs)
        z_past = self.posenc(self.in_proj(e_past))
        curr_seq = self.start_token.expand(z_past.size(0), 1, -1)
        
        out = {k: [] for k in ["lat", "lon", "sog", "cog"]}
        
        # Track previous steps for continuity constraints
        last_lat = past_idxs["lat"][:, -1]
        last_lon = past_idxs["lon"][:, -1]
        last_sog = past_idxs["sog"][:, -1]
        
        # Prepare Water Mask for GPU
        wm_tensor = None
        if self.water_mask is not None:
            wm_tensor = self.water_mask.to(device) # [N_lat, N_lon]

        for _ in range(L):
            curr_in = self.posenc(curr_seq)
            tgt_mask = torch.triu(torch.ones(curr_in.size(1), curr_in.size(1), device=device), diagonal=1).bool()
            dec_out = self.decoder(tgt=curr_in, memory=z_past, tgt_mask=tgt_mask)
            last_step = dec_out[:, -1, :] 
            
            # --- JOINT SAMPLING FOR LAT/LON ---
            logit_lat = self.head_lat(last_step) # [B, N_lat]
            logit_lon = self.head_lon(last_step) # [B, N_lon]
            
            # Joint Prob: [B, N_lat, N_lon]
            joint_logits = logit_lat.unsqueeze(2) + logit_lon.unsqueeze(1)
            
            # --- 1. APPLY WATER MASK (Force Field) ---
            if wm_tensor is not None:
                mask_penalty = torch.zeros_like(wm_tensor, dtype=torch.float)
                mask_penalty[~wm_tensor] = -1e9
                joint_logits = joint_logits + mask_penalty.unsqueeze(0) 

            # --- 2. APPLY LOCAL WINDOW (Continuity Field) ---
            if local_window is not None:
                # Create 2D grid indices: [1, N_lat, 1] and [1, 1, N_lon]
                grid_lat = torch.arange(self.bins.n_lat, device=device).view(1, -1, 1)
                grid_lon = torch.arange(self.bins.n_lon, device=device).view(1, 1, -1)
                
                # Current positions: [B, 1, 1]
                curr_lat_view = last_lat.view(-1, 1, 1)
                curr_lon_view = last_lon.view(-1, 1, 1)
                
                # Chebyshev distance (square window)
                dist_lat = (grid_lat - curr_lat_view).abs()
                dist_lon = (grid_lon - curr_lon_view).abs()
                
                # Mask bins outside window
                window_mask = (dist_lat <= local_window) & (dist_lon <= local_window)
                
                # Apply mask (-Infinity where false)
                joint_logits = torch.where(window_mask, joint_logits, torch.tensor(-float('Inf'), device=device))
                
                # --- 3. PREVENT STUCK (Anti-Stationary) ---
                if prevent_stuck:
                    # If speed > ~5 knots (bin 10), forbid staying in exactly the same cell
                    sog_thresh = 10
                    is_moving = (last_sog > sog_thresh).view(-1, 1, 1)
                    is_center = (dist_lat == 0) & (dist_lon == 0)
                    
                    # Apply penalty to center pixel if moving
                    joint_logits = torch.where(is_moving & is_center, torch.tensor(-float('Inf'), device=device), joint_logits)

            # Flatten to [B, N_lat * N_lon] for sampling
            B_size = joint_logits.size(0)
            flat_logits = joint_logits.view(B_size, -1)
            
            # Top-K Sampling
            if top_k > 0:
                v, _ = torch.topk(flat_logits, min(top_k, flat_logits.size(-1)))
                flat_logits[flat_logits < v[:, [-1]]] = -float('Inf')

            if sampling == "greedy" or temperature < 1e-5:
                flat_token = torch.argmax(flat_logits, dim=-1)
            else:
                probs = F.softmax(flat_logits / temperature, dim=-1)
                flat_token = torch.multinomial(probs, 1).squeeze(-1)
            
            # Unravel
            next_lat = flat_token // self.bins.n_lon
            next_lon = flat_token % self.bins.n_lon
            
            # --- Sample SOG/COG independently ---
            def _sample_1d(l):
                if top_k > 0:
                    v, _ = torch.topk(l, min(top_k, l.size(-1)))
                    l[l < v[:, [-1]]] = -float('Inf')
                if sampling == "greedy" or temperature < 1e-5: return torch.argmax(l, dim=-1)
                return torch.multinomial(F.softmax(l / temperature, dim=-1), 1).squeeze(-1)

            next_sog = _sample_1d(self.head_sog(last_step))
            next_cog = _sample_1d(self.head_cog(last_step))

            # Store
            out["lat"].append(next_lat); out["lon"].append(next_lon)
            out["sog"].append(next_sog); out["cog"].append(next_cog)
            
            # Update trackers
            last_lat, last_lon, last_sog = next_lat, next_lon, next_sog

            # Embed for next step
            next_vals = {"lat": next_lat, "lon": next_lon, "sog": next_sog, "cog": next_cog}
            next_embed = self._embed({k: v.unsqueeze(1) for k, v in next_vals.items()})
            curr_seq = torch.cat([curr_seq, self.in_proj(next_embed)], dim=1)
            
        return {k: torch.stack(v, dim=1) for k, v in out.items()}