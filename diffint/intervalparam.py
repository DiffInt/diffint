#!/usr/bin/env python3


import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# IntervalParam / FeatureNet / IntervalAttentionHead / IPANAutoencoder
# (adapted from your code with device/dtype safety)
# -------------------------
class IntervalParam(nn.Module):
    """
    Stores centers (m) and width logits (log_w) for K intervals in d dims.
    Returns low/high as (K,d).
    """
    def __init__(self, K, d, init_scale=0.1):
        super().__init__()
        self.K = K
        self.d = d
        # initialize centers and log-widths
        self.m = nn.Parameter(init_scale * torch.randn(K, d))
        self.log_w = nn.Parameter(init_scale * torch.randn(K, d))

    def intervals(self, device=None, dtype=None):
        # half-width (positive) with small floor for stability
        w = F.softplus(self.log_w) + 1e-6
        low = self.m - w
        high = self.m + w
        if device is not None or dtype is not None:
            low = low.to(device=device, dtype=dtype)
            high = high.to(device=device, dtype=dtype)
        return low, high   # (K,d), (K,d)
