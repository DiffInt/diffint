#!/usr/bin/env python3

import torch.nn as nn

eps = 1e-8

class FeatureNet(nn.Module):
    """
    Vectorized feature MLP that maps scalars -> value_dim.
    Accepts input x shape (B, d) and returns (B, d, value_dim).
    """
    def __init__(self, value_dim=16, hidden=32):
        super().__init__()
        # this net will be applied to last dim of shape (..., 1)
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, value_dim)
        )

    def forward(self, x):
        # x: (B, d) -> need (...,1) to pass through Linear(1,...)
        # net supports broadcasting: apply to shape (B, d, 1)
        x_in = x.unsqueeze(-1)           # (B, d, 1)
        out = self.net(x_in)             # (B, d, value_dim)
        return out
