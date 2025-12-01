from __future__ import annotations
import torch


def make_time_mask(batch_size: int, seq_len: int, mask_ratio: float = 0.12, span_len: int | None = None, device: torch.device | None = None) -> torch.Tensor:
    """Create a boolean mask over time steps per sequence.
    True indicates a masked timestep.
    - span_len=None → independent positions
    - span_len>1 → short contiguous spans
    """
    device = device or torch.device("cpu")
    num_mask = max(1, int(round(seq_len * float(mask_ratio))))
    mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    if span_len is None or span_len <= 1:
        # independent positions
        for b in range(batch_size):
            idx = torch.randperm(seq_len, device=device)[:num_mask]
            mask[b, idx] = True
        return mask

    # span masking
    span_len = int(span_len)
    for b in range(batch_size):
        remaining = num_mask
        while remaining > 0:
            start = int(torch.randint(low=0, high=seq_len, size=(1,), device=device).item())
            end = min(seq_len, start + span_len)
            mask[b, start:end] = True
            remaining -= (end - start)
    return mask


def apply_time_mask(x: torch.Tensor, mask_time: torch.Tensor, mask_value: float = 0.0, noise_std: float = 0.0) -> torch.Tensor:
    """Apply a time-step mask to x [B,T,F].
    Masked steps are filled with mask_value or with Gaussian noise around mask_value.
    """
    assert x.dim() == 3, "x must be [B,T,F]"
    assert mask_time.shape[:2] == x.shape[:2], "mask_time shape must match [B,T]"
    x_masked = x.clone()
    if noise_std > 0.0:
        noise = torch.randn_like(x_masked) * noise_std
        fill = mask_value + noise
    else:
        fill = torch.full_like(x_masked, mask_value)
    # expand mask to features
    mt = mask_time.unsqueeze(-1).expand_as(x_masked)
    x_masked = torch.where(mt, fill, x_masked)
    return x_masked


