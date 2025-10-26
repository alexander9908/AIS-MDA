from __future__ import annotations
import torch

def pad_mask(lengths, max_len=None):
    if max_len is None:
        max_len = int(max(lengths))
    mask = torch.zeros(len(lengths), max_len, dtype=torch.bool)
    for i, L in enumerate(lengths):
        mask[i, :L] = 1
    return ~mask  # True on padding positions
