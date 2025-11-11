# src/models/traisformer1.py
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional: fast boolean mask of water cells on a lat/lon grid
# (expects signature: make_water_mask(lat_min, lat_max, lon_min, lon_max, n_lat, n_lon) -> np.bool_[n_lat, n_lon])
try:
    from src.eval.build_water_mask_V2 import make_water_mask
    _HAS_WATER_MASK = True
except Exception:
    _HAS_WATER_MASK = False

# -----------------------------
# Utilities: bins & four-hot I/O
# -----------------------------

@dataclass
class BinSpec:
    # Uniform binning for each attribute; edges are inclusive of left edge.
    # For degrees: lat in [lat_min, lat_max], lon in [lon_min, lon_max]
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
        # clamp, map to [0, n-1]
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
        # wrap to [0,360)
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


# -----------------------------
# Positional encoding (standard)
# -----------------------------

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # [max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


# -----------------------------
# TrAISformer model
# -----------------------------

class TrAISformer(nn.Module):
    """
    Paper-inspired architecture:
      - Discretize (lat,lon,SOG,COG) into one-hot (‘four-hot’)
      - Learn embedding per attribute and concatenate -> e_t
      - Causal Transformer decoder over e_{0:T-1}
      - 4 classification heads predict logits for next step: lat/lon/sog/cog
      - Multi-step training via teacher forcing, multi-resolution CE loss

    This implementation also supports an optional water mask (land avoidance) used
    during generation by snapping land samples to the nearest water cell. The snap
    is continuity-aware so it prefers to stay on the same side of a strait.
    """

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
        use_water_mask: bool = True,
    ):
        super().__init__()
        self.bins = bins
        self.coarse_merge = coarse_merge
        self.coarse_beta = coarse_beta
        self.use_water_mask = use_water_mask

        # Embeddings (attribute-specific)
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

        # 4 heads — separate linear classifiers
        self.head_lat = nn.Linear(d_model, bins.n_lat)
        self.head_lon = nn.Linear(d_model, bins.n_lon)
        self.head_sog = nn.Linear(d_model, bins.n_sog)
        self.head_cog = nn.Linear(d_model, bins.n_cog)

        # Learnable start token for decoding
        self.start_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.start_token, std=0.02)

        # Optional water mask buffer (lat_idx, lon_idx) -> True means water
        if use_water_mask and _HAS_WATER_MASK:
            wm = make_water_mask(
                bins.lat_min, bins.lat_max, bins.lon_min, bins.lon_max, bins.n_lat, bins.n_lon
            )  # np.bool_[n_lat, n_lon]
            wm = np.asarray(wm, dtype=bool)
            if wm.shape != (bins.n_lat, bins.n_lon):
                raise ValueError(f"water mask has wrong shape {wm.shape}, expected {(bins.n_lat, bins.n_lon)}")
            self.register_buffer("water_mask", torch.tensor(wm, dtype=torch.bool), persistent=False)
        else:
            # default to all-water (no snapping) if mask not available
            self.register_buffer("water_mask", torch.ones(bins.n_lat, bins.n_lon, dtype=torch.bool), persistent=False)

    # ---- internal helpers ----
    def _embed_step(self, idxs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        idxs[*] shape: [B, T] (or [B,1] for a single-step)
        returns: [B, T, d_emb_concat]
        """
        def _bt(x):
            return x if x.dim() == 2 else x.unsqueeze(1)
        e_lat = self.lat_emb(_bt(idxs["lat"]))
        e_lon = self.lon_emb(_bt(idxs["lon"]))
        e_sog = self.sog_emb(_bt(idxs["sog"]))
        e_cog = self.cog_emb(_bt(idxs["cog"]))
        return torch.cat([e_lat, e_lon, e_sog, e_cog], dim=-1)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        # mask future positions for autoregressive decoding
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

    # ---- training forward (teacher forcing) ----
    def forward(
        self,
        past_idxs: Dict[str, torch.Tensor],     # [B, T]
        future_idxs: Dict[str, torch.Tensor],   # [B, L] teacher-forcing targets
    ) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device

        # Embed past sequence
        e_past = self._embed_step(past_idxs)                        # [B, T, d_emb]
        z_past = self.in_proj(e_past)                               # [B, T, d_model]
        z_past = self.posenc(z_past)

        # Build decoder input by shifting future right (teacher forcing)
        e_future_in = self._embed_step(future_idxs)                 # [B, L, d_emb]
        z_future_in = self.in_proj(e_future_in)                     # [B, L, d_model]
        z_future_in = torch.cat([self.start_token.expand(z_future_in.size(0), 1, -1),
                                 z_future_in[:, :-1, :]], dim=1)    # [B, L, d_model]
        z_future_in = self.posenc(z_future_in)

        tgt_mask = self._causal_mask(z_future_in.size(1), device)
        dec = self.decoder(tgt=z_future_in, memory=z_past, tgt_mask=tgt_mask)  # [B, L, d_model]

        logits = {
            "lat": self.head_lat(dec),
            "lon": self.head_lon(dec),
            "sog": self.head_sog(dec),
            "cog": self.head_cog(dec),
        }
        return logits

    # ---- loss (fine + coarse) ----
    def ce_loss_multi(
        self,
        logits: Dict[str, torch.Tensor],     # [B, L, n_bins_attr]
        target_idxs: Dict[str, torch.Tensor] # [B, L]
    ) -> torch.Tensor:
        B, L = target_idxs["lat"].shape
        loss = 0.0

        for key, head in logits.items():
            # fine
            loss = loss + F.cross_entropy(
                head.reshape(B * L, head.size(-1)),
                target_idxs[key].reshape(B * L),
                reduction="mean"
            )

            # coarse (optional)
            if self.coarse_beta > 0 and self.coarse_merge > 1:
                coarse_tgt = coarse_indices(target_idxs[key], self.coarse_merge)
                n_bins = head.size(-1)
                merge = self.coarse_merge

                if n_bins % merge != 0:
                    pad = merge - (n_bins % merge)
                    head_pad = F.pad(head, (0, pad))
                    n_bins_pad = n_bins + pad
                    head_c = head_pad.view(B * L, n_bins_pad // merge, merge).logsumexp(-1)
                    nb_coarse = n_bins_pad // merge
                else:
                    head_c = head.view(B * L, n_bins // merge, merge).logsumexp(-1)
                    nb_coarse = n_bins // merge

                loss = loss + self.coarse_beta * F.cross_entropy(
                    head_c, coarse_tgt.reshape(B * L).clamp(0, nb_coarse - 1), reduction="mean"
                )

        return loss

    # ---- generation (stochastic sampling or greedy) ----
    @torch.no_grad()
    def generate(
        self,
        past_idxs: Dict[str, torch.Tensor],   # [B, T]
        L: int,
        sampling: Literal["sample", "greedy"] = "sample",
        temperature: float = 1.0,
        top_k: Optional[int] = 20,
    ) -> Dict[str, torch.Tensor]:
        """
        Autoregressively sample L future steps (GLOBAL joint sampler).
        Returns dict of future bin indices [B, L] for lat,lon,sog,cog.
        """
        self.eval()
        device = next(self.parameters()).device

        # cache past encodings
        e_past = self._embed_step(past_idxs)
        z_past = self.in_proj(e_past)
        z_past = self.posenc(z_past)

        B = z_past.size(0)
        y = self.start_token.expand(B, 1, -1)  # [B,1,D]
        out = {k: [] for k in ["lat", "lon", "sog", "cog"]}

        def _mask_topk(logit: torch.Tensor, k: Optional[int]) -> torch.Tensor:
            if k is None or k <= 0:
                return logit
            k = min(k, logit.size(-1))
            v, _ = torch.topk(logit, k=k)
            thresh = v[:, -1].unsqueeze(-1)
            return torch.where(logit < thresh, torch.full_like(logit, -1e9), logit)

        # ---- GLOBAL sampler knobs (sane defaults) ----
        LAMBDA_CONT = 0.04     # continuity; higher = stickier
        ALPHA_DIR   = 1.60     # push along heading from COG/SOG
        BETA_TURN   = 0.55     # discourage sharp reversals
        STEP_SCALE  = 0.70     # how far ahead the heading “aims” (0.5–0.8 good)

        wm = self.water_mask.to(device) if (self.use_water_mask and self.water_mask is not None) else None

        # continuity anchors: last TWO past steps so we can measure last move
        prev_lat = past_idxs["lat"][:, -1].clone()
        prev_lon = past_idxs["lon"][:, -1].clone()
        if past_idxs["lat"].size(1) >= 2:
            prev2_lat = past_idxs["lat"][:, -2].clone()
            prev2_lon = past_idxs["lon"][:, -2].clone()
        else:
            prev2_lat = prev_lat.clone()
            prev2_lon = prev_lon.clone()

        nlat, nlon = self.bins.n_lat, self.bins.n_lon
        ii_full = torch.arange(nlat, device=device).unsqueeze(1).float()  # [nlat,1]
        jj_full = torch.arange(nlon, device=device).unsqueeze(0).float()  # [1,nlon]

        for t in range(L):
            # decode next-step logits
            tgt_mask = self._causal_mask(y.size(1), device)
            dec = self.decoder(tgt=self.posenc(y), memory=z_past, tgt_mask=tgt_mask)
            h = dec[:, -1, :]  # [B, D]

            logit_lat = self.head_lat(h) / max(temperature, 1e-6)
            logit_lon = self.head_lon(h) / max(temperature, 1e-6)
            logit_sog = self.head_sog(h) / max(temperature, 1e-6)
            logit_cog = self.head_cog(h) / max(temperature, 1e-6)

            # truncate tails if requested
            logit_lat = _mask_topk(logit_lat, top_k)
            logit_lon = _mask_topk(logit_lon, top_k)

            lat_idx_list, lon_idx_list = [], []

            for b in range(B):
                # base 2D scores = outer sum of lat/lon logits (GLOBAL grid)
                ls = logit_lat[b].unsqueeze(1)   # [nlat,1]
                os = logit_lon[b].unsqueeze(0)   # [1,nlon]
                score_2d = ls + os               # [nlat,nlon]

                # continuity from previous cell (L1 distance)
                i0 = int(prev_lat[b].item()); j0 = int(prev_lon[b].item())
                cont = (ii_full - i0).abs() + (jj_full - j0).abs()
                score_2d = score_2d - LAMBDA_CONT * cont

                # small heading / motion prior from a quick COG/SOG pick
                if sampling == "sample":
                    sog_idx_b = torch.distributions.Categorical(logits=logit_sog[b]).sample()
                    cog_idx_b = torch.distributions.Categorical(logits=logit_cog[b]).sample()
                else:
                    sog_idx_b = torch.argmax(logit_sog[b])
                    cog_idx_b = torch.argmax(logit_cog[b])

                sog_kn = self.bins.bin_to_sog_mid(sog_idx_b.unsqueeze(0)).item()
                cog_deg = self.bins.bin_to_cog_mid(cog_idx_b.unsqueeze(0)).item()

                # ---- NEW: speed floor for the PRIOR only ----
                sog_eff = max(sog_kn, 3.0)  # knots (2–4 works)

                # ---- NEW: blend COG with geometric track bearing from last two past points ----
                lat_prev2_deg = self.bins.bin_to_lat_mid(prev2_lat[b].unsqueeze(0)).item()
                lon_prev2_deg = self.bins.bin_to_lon_mid(prev2_lon[b].unsqueeze(0)).item()
                lat_prev_deg  = self.bins.bin_to_lat_mid(prev_lat[b].unsqueeze(0)).item()
                lon_prev_deg  = self.bins.bin_to_lon_mid(prev_lon[b].unsqueeze(0)).item()
                
                # bearing (prev2 -> prev)
                dlo = math.radians(lon_prev_deg - lon_prev2_deg)
                la1 = math.radians(lat_prev2_deg); la2 = math.radians(lat_prev_deg)
                y = math.sin(dlo) * math.cos(la2)
                x = math.cos(la1)*math.sin(la2) - math.sin(la1)*math.cos(la2)*math.cos(dlo)
                bearing_track = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
                
                heading_used = (0.5 * bearing_track + 0.5 * cog_deg) % 360.0  # 50/50 blend

                # ---- aim a bit ahead along heading_used; scale by speed and grid size ----
                step_mag_i = STEP_SCALE * self.bins.n_lat * (sog_eff / max(self.bins.sog_max, 1.0)) / 100.0
                step_mag_j = STEP_SCALE * self.bins.n_lon * (sog_eff / max(self.bins.sog_max, 1.0)) / 100.0
                dy = -math.sin(math.radians(heading_used)) * step_mag_i
                dx =  math.cos(math.radians(heading_used)) * step_mag_j
                ic = i0 + dy; jc = j0 + dx

                dir_cost = (ii_full - ic).abs() + (jj_full - jc).abs()
                score_2d = score_2d - ALPHA_DIR * dir_cost

                # AFTER you apply the coastline (land) mask and BEFORE sampling:
                score_2d[i0, j0] = -torch.inf    # don't allow staying in the same cell
                # (use -1e9 if you'd rather avoid infs)

                # discourage staying in the same cell (use this batch's previous cell)
                #score_2d[i0, j0] -= 0.4


                # anti-zigzag vs last move (keep this INSIDE the b-loop)
                di_prev = float(i0 - int(prev2_lat[b].item()))
                dj_prev = float(j0 - int(prev2_lon[b].item()))
                di_c = ii_full - i0; dj_c = jj_full - j0
                denom = (di_c**2 + dj_c**2).sqrt() * (di_prev**2 + dj_prev**2)**0.5 + 1e-6
                cos_sim = (di_c * di_prev + dj_c * dj_prev) / denom
                turn_pen = (1.0 - cos_sim).clamp(min=0.0)
                score_2d = score_2d - BETA_TURN * turn_pen

                # HARD coastline mask on the whole grid
                if wm is not None:
                    score_2d = torch.where(wm, score_2d, torch.full_like(score_2d, -1e9))

                # Forbid staying in the same cell
                score_2d[i0, j0] = -1e9

                # --- First step stabilization ---
                if t == 0:
                    # 8-neighborhood only, then greedy
                    iL = max(0, i0 - 1); iR = min(self.bins.n_lat - 1, i0 + 1)
                    jL = max(0, j0 - 1); jR = min(self.bins.n_lon - 1, j0 + 1)
                    neigh = torch.zeros_like(score_2d, dtype=torch.bool)
                    neigh[iL:iR+1, jL:jR+1] = True
                    score_2d = torch.where(neigh, score_2d, torch.full_like(score_2d, -1e9))
                    flat = torch.argmax(score_2d.reshape(-1))            # Tensor (0-dim)
                else:
                    # usual sampling/greedy for subsequent steps
                    flat = (torch.distributions.Categorical(logits=score_2d.reshape(-1)).sample()
                            if sampling == "sample" else torch.argmax(score_2d.reshape(-1)))

                # convert flat (0-dim tensor) -> (i,j) tensors
                li = (flat // nlon).long().to(device)
                lj = (flat %  nlon).long().to(device)
                lat_idx_list.append(li)
                lon_idx_list.append(lj)
                
            # finalize each step
            lat_idx = torch.stack(lat_idx_list, dim=0)   # [B]
            lon_idx = torch.stack(lon_idx_list, dim=0)

            # sample sog/cog for the actual token (independent)
            if sampling == "sample":
                sog_idx = torch.distributions.Categorical(logits=_mask_topk(logit_sog, top_k)).sample()
                cog_idx = torch.distributions.Categorical(logits=_mask_topk(logit_cog, top_k)).sample()
            else:
                sog_idx = _mask_topk(logit_sog, top_k).argmax(dim=-1)
                cog_idx = _mask_topk(logit_cog, top_k).argmax(dim=-1)

            # update continuity anchors (keep last two)
            prev2_lat, prev2_lon = prev_lat, prev_lon
            prev_lat, prev_lon = lat_idx, lon_idx

            # append & embed for next step
            out["lat"].append(lat_idx.long())
            out["lon"].append(lon_idx.long())
            out["sog"].append(sog_idx.long())
            out["cog"].append(cog_idx.long())

            step_embed = self._embed_step({
                "lat": lat_idx.unsqueeze(1),
                "lon": lon_idx.unsqueeze(1),
                "sog": sog_idx.unsqueeze(1),
                "cog": cog_idx.unsqueeze(1),
            })  # [B,1,emb]
            y = torch.cat([y, self.in_proj(step_embed)], dim=1)

        # finalize
        for k in out:
            out[k] = torch.stack(
                [torch.as_tensor(x, device=device) for x in out[k]],
                dim=1
            ).long()

        # final belt-and-suspenders coastline check
        if self.use_water_mask and self.water_mask is not None:
            wm = self.water_mask
            on_land = ~wm[out["lat"], out["lon"]]
            if bool(on_land.any()):
                n_bad = int(on_land.sum().item())
                raise RuntimeError(f"Land violation in generation ({n_bad} steps). Masking failed.")
        return out

    # ---- convenience: decode bins to midpoints (lat/lon in degrees) ----
    def bins_to_continuous(self, idxs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "lat": self.bins.bin_to_lat_mid(idxs["lat"]),
            "lon": self.bins.bin_to_lon_mid(idxs["lon"]),
            "sog": self.bins.bin_to_sog_mid(idxs["sog"]),
            "cog": self.bins.bin_to_cog_mid(idxs["cog"]),
        }
