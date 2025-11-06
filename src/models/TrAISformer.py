import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    n_lat: int         # e.g., 256 for 0.01° over ~2.5°
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
        return 0.0 + (idx.float() + 0.5) * step


def build_fourhot_indices(
    bins: BinSpec,
    lat: torch.Tensor, lon: torch.Tensor, sog: torch.Tensor, cog: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Return bin indices dict with keys: lat, lon, sog, cog
    Shapes follow input tensors (e.g., [B,T]).
    """
    return {
        "lat": bins.lat_to_bin(lat),
        "lon": bins.lon_to_bin(lon),
        "sog": bins.sog_to_bin(sog),
        "cog": bins.cog_to_bin(cog),
    }


def coarse_indices(idx: torch.Tensor, merge: int) -> torch.Tensor:
    """
    Merge consecutive bins into coarse bins by integer division.
    """
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

    Ref: TrAISformer (Nguyen & Fablet, 2024). :contentReference[oaicite:1]{index=1}
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
        # multi-resolution loss: coarse = merge factor (e.g., 3 → 3× coarser)
        coarse_merge: int = 3,
        coarse_beta: float = 0.2,
    ):
        super().__init__()
        self.bins = bins
        self.coarse_merge = coarse_merge
        self.coarse_beta = coarse_beta

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

        # Learnable start token for decoding (one per batch broadcast)
        self.start_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.start_token, std=0.02)

    # ---- internal helpers ----
    def _embed_step(self, idxs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        idxs[*] shape: [B, T]
        returns e: [B, T, d_emb_concat]
        """
        e_lat = self.lat_emb(idxs["lat"])
        e_lon = self.lon_emb(idxs["lon"])
        e_sog = self.sog_emb(idxs["sog"])
        e_cog = self.cog_emb(idxs["cog"])
        e = torch.cat([e_lat, e_lon, e_sog, e_cog], dim=-1)
        return e

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        # mask future positions for autoregressive decoding
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

    # ---- training forward (teacher forcing) ----
    def forward(
        self,
        past_idxs: Dict[str, torch.Tensor],     # [B, T]
        future_idxs: Dict[str, torch.Tensor],   # [B, L] teacher-forcing targets
    ) -> Dict[str, torch.Tensor]:
        """
        Returns logits dict for the L future steps (classification over bins).

        Shapes:
          past_idxs:   dict of [B, T]
          future_idxs: dict of [B, L]
          outputs: logits dict of [B, L, n_bins_attr]
        """
        device = next(self.parameters()).device

        # Embed past sequence
        e_past = self._embed_step(past_idxs)                        # [B, T, d_emb]
        z_past = self.in_proj(e_past)                               # [B, T, d_model]
        z_past = self.posenc(z_past)

        # Build decoder input by shifting future right (teacher forcing):
        # prepend learnable start token, then project embeddings of future input tokens
        e_future_in = self._embed_step(future_idxs)                 # [B, L, d_emb]
        z_future_in = self.in_proj(e_future_in)                     # [B, L, d_model]
        z_future_in = torch.cat([self.start_token.expand(z_future_in.size(0), -1, -1),
                                 z_future_in[:, :-1, :]], dim=1)    # [B, L, d_model]
        z_future_in = self.posenc(z_future_in)

        # Causal mask for target side (self-attention)
        tgt_mask = self._causal_mask(z_future_in.size(1), device)

        # Decode conditioned on past (encoder-less: use decoder with memory=z_past)
        dec = self.decoder(tgt=z_future_in, memory=z_past, tgt_mask=tgt_mask)  # [B, L, d_model]

        # Heads
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
            loss += F.cross_entropy(
                head.reshape(B * L, head.size(-1)),
                target_idxs[key].reshape(B * L),
                reduction="mean"
            )

            # coarse (optional)
            if self.coarse_beta > 0 and self.coarse_merge > 1:
                coarse_tgt = coarse_indices(target_idxs[key], self.coarse_merge)
                n_bins = head.size(-1)
                # merge bins by summing logits over groups of size merge
                merge = self.coarse_merge
                # reshape [..., n_bins] -> [..., n_bins//merge, merge] -> sum over merge
                if n_bins % merge != 0:
                    # pad to multiple of merge
                    pad = merge - (n_bins % merge)
                    head_pad = F.pad(head, (0, pad))
                    n_bins_pad = n_bins + pad
                    head_c = head_pad.view(B * L, n_bins_pad // merge, merge).logsumexp(-1)
                    nb_coarse = n_bins_pad // merge
                else:
                    head_c = head.view(B * L, n_bins // merge, merge).logsumexp(-1)
                    nb_coarse = n_bins // merge

                loss += self.coarse_beta * F.cross_entropy(
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
        Autoregressively sample L future steps.
        Returns dict of future bin indices [B, L] for lat,lon,sog,cog.
        """
        self.eval()
        device = next(self.parameters()).device

        # cache past
        e_past = self._embed_step(past_idxs)
        z_past = self.in_proj(e_past)
        z_past = self.posenc(z_past)

        B = z_past.size(0)
        # start token
        y = self.start_token.expand(B, 1, -1)  # [B,1,D]

        out = {k: [] for k in ["lat", "lon", "sog", "cog"]}

        for _ in range(L):
            tgt_mask = self._causal_mask(y.size(1), device)
            dec = self.decoder(tgt=self.posenc(y), memory=z_past, tgt_mask=tgt_mask)

            # last step features
            h = dec[:, -1, :]  # [B, D]
            logits = {
                "lat": self.head_lat(h),
                "lon": self.head_lon(h),
                "sog": self.head_sog(h),
                "cog": self.head_cog(h),
            }

            idxs_step = {}
            for key, logit in logits.items():
                logit = logit / max(temperature, 1e-6)
                if top_k is not None:
                    v, _ = torch.topk(logit, k=min(top_k, logit.size(-1)))
                    thresh = v[:, -1].unsqueeze(-1)
                    logit = torch.where(logit < thresh, torch.full_like(logit, -1e9), logit)
                if sampling == "sample":
                    idx = torch.distributions.Categorical(logits=logit).sample()
                else:
                    idx = logit.argmax(dim=-1)
                idxs_step[key] = idx
                out[key].append(idx)

            # append embedded token for next step
            e_next = self._embed_step(idxs_step)
            z_next = self.in_proj(e_next)  # [B,1,D]
            y = torch.cat([y, z_next], dim=1)

        # stack to [B, L]
        for k in out:
            out[k] = torch.stack(out[k], dim=1)
        return out

    # ---- convenience: decode bins to midpoints (lat/lon in degrees) ----
    def bins_to_continuous(self, idxs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "lat": self.bins.bin_to_lat_mid(idxs["lat"]),
            "lon": self.bins.bin_to_lon_mid(idxs["lon"]),
            "sog": self.bins.bin_to_sog_mid(idxs["sog"]),
            "cog": self.bins.bin_to_cog_mid(idxs["cog"]),
        }
