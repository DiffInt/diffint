#!/usr/bin/env python3


import torch
import torch.nn as nn
import torch.nn.functional as F
from .intervalparam import IntervalParam

eps = 1e-8

class IntervalPatternNetwork(nn.Module):
    """
    Interval bottleneck: K learnable intervals per feature, soft memberships, and the
    competitive aggregation of the paper (a softmax over units of the summed
    log-memberships), giving one scalar code per unit.
    - Vectorized soft-membership computation
    - Delta-method variance estimation for ICR
    """

    def __init__(self, K, d, tau=0.1, min_count=1.0):
        """
        K: number of intervals
        d: number of feature dimensions
        tau: softness parameter for soft membership
        min_count: minimal effective count for variance stabilization
        """
        super().__init__()
        self.K = K
        self.d = d

        # interval parameters (K, d)
        self.intervals = IntervalParam(K, d)

        # softness parameter -- keep as buffer for possible device moves
        self.register_buffer('tau_tensor', torch.tensor(float(tau)))
        self.tau = float(tau)

        # support counts per (K,d) as buffer (can be updated externally)
        self.register_buffer('support', torch.ones(K, d))

        # minimal effective count to stabilize variance estimates
        self.min_count = float(min_count)

    @property
    def device(self):
        # helper
        params = list(self.parameters())
        if len(params) > 0:
            return params[0].device
        return self.support.device

    def set_tau(self, tau):
        self.tau = float(tau)
        # update buffer in-place so it moves with module
        self.tau_tensor.fill_(self.tau)

    def update_support(self, new_support):
        """
        Replace support buffer with new_support (K,d) tensor or array-like.
        """
        ns = torch.as_tensor(new_support, dtype=self.support.dtype, device=self.support.device)
        if ns.shape != (self.K, self.d):
            raise ValueError(f"new_support must be shape ({self.K},{self.d})")
        self.support.copy_(ns)

    def soft_ind(self, x, low, high):
        """
        Vectorized soft membership:
        x: (B, d)
        low/high: (K, d)
        returns: ind (B, K, d)
        Form: sigmoid((x-low)/tau) * sigmoid((high-x)/tau)
        """
        # ensure tau is on correct device and dtype
        tau = torch.as_tensor(self.tau, dtype=x.dtype, device=x.device)

        # broadcast: x -> (B,1,d) ; low/high -> (1,K,d)
        x_exp = x.unsqueeze(1)       # (B,1,d)
        low = low.unsqueeze(0)       # (1,K,d)
        high = high.unsqueeze(0)     # (1,K,d)

        # compute inside/outside sigmoids
        # use stable ops; this creates (B,K,d) tensors
        ind_low = torch.sigmoid((x_exp - low) / (tau + 1e-12))
        ind_high = torch.sigmoid((high - x_exp) / (tau + 1e-12))
        ind = ind_low * ind_high

        return ind

    def _compute_icr_and_var(self, ind, y):
        """
        Compute ICR, var(ICR) etc. using delta-method (vectorized).
        ind: (B, K, d) soft membership
        y: (B,) labels in [0,1]
        returns dict of tensors (K,d)
        """
        B = ind.shape[0]
        # ensure y dtype matches and shape (B,1,1) for broadcasting
        yf = y.view(B, 1, 1).to(ind.dtype)

        # Soft counts and positives inside intervals
        n_in = ind.sum(dim=0)                    # (K,d) soft support inside
        P_in = (ind * yf).sum(dim=0)             # (K,d) soft positives inside
        R_in = P_in / (n_in + eps)               # (K,d) proportion inside

        # Complement (outside)
        soft_out = (1.0 - ind)
        n_out = soft_out.sum(dim=0)              # (K,d)
        P_out = (soft_out * yf).sum(dim=0)       # (K,d)
        R_out = P_out / (n_out + eps)            # (K,d)

        # Stability: floor counts (but keep original n_in/n_out for diagnostics)
        n_in_clamped = torch.clamp(n_in, min=self.min_count)
        n_out_clamped = torch.clamp(n_out, min=self.min_count)

        # avoid negative or nan
        S = R_in + R_out + eps                   # (K,d)
        icr = R_in / S                           # (K,d)

        # gradient of a/(a+b) w.r.t a and b
        grad_a = R_out / (S * S)                 # (K,d)
        grad_b = - R_in / (S * S)                # (K,d)

        # asymptotic variances for proportions (plug-in)
        var_a = (R_in * (1.0 - R_in)) / n_in_clamped   # (K,d)
        var_b = (R_out * (1.0 - R_out)) / n_out_clamped # (K,d)

        var_icr = (grad_a * grad_a) * var_a + (grad_b * grad_b) * var_b
        var_icr = torch.clamp(var_icr, min=1e-12)  # avoid exact zero

        inv_var_weight = 1.0 / (var_icr + 1e-12)
        zscore = (icr - 0.5) / torch.sqrt(var_icr + 1e-12)

        stats = {
            'icr': icr,
            'var_icr': var_icr,
            'inv_var_weight': inv_var_weight,
            'z_score': zscore,
            'n_in': n_in,
            'n_out': n_out,
            'R_in': R_in,
            'R_out': R_out
        }
        return stats

    def forward(self, x, y=None, compute_stats=False):
        """
        Forward pass.

        Inputs:
          x: (B, d) float tensor of features (one scalar per feature dimension)
          y: optional (B,) labels in [0,1]
          compute_stats: if True and y provided, returns stats dict too.

        Returns:
          h: (B, K, 1) unit codes (competitive aggregation over units)
          alpha: None (kept for the return signature)
          intervals: (low, high) each (K,d)
          stats (optional): dict with (K,d) tensors
        """
        if x.ndim != 2 or x.shape[1] != self.d:
            raise ValueError(f"x must be shape (B, {self.d})")

        B = x.shape[0]
        device = x.device
        dtype = x.dtype

        # get intervals on correct device/dtype
        low, high = self.intervals.intervals(device=device, dtype=dtype)

        # vectorized soft membership (B, K, d)
        ind = self.soft_ind(x, low, high)

        # competitive aggregation: softmax over units of the summed log-memberships
        logi = ind.clamp(min=eps).log()                         # (B, K, d)
        pattern_prob = torch.softmax(logi.sum(dim=2), dim=1)    # (B, K)
        h = pattern_prob.unsqueeze(-1)                          # (B, K, 1)
        alpha = None

        if compute_stats and (y is not None):
            stats = self._compute_icr_and_var(ind, y)
            return (h, alpha, (low, high), stats)

        return (h, alpha, (low, high))
