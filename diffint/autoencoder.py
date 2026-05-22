#!/usr/bin/env python3

from typing import Optional
import torch
import torch.nn as nn
from .intervalPatternNetwork import IntervalPatternNetwork


class DiffIntAutoencoder(nn.Module):
    """Interval-bottleneck autoencoder: an interval pattern network head feeding a
    small MLP decoder. The per-sample reconstruction error is the anomaly score."""

    def __init__(
        self,
        K: int,
        d: int,
        value_dim: int = 16,
        decoder_hidden: int = 128,
        decoder_depth: int = 2,
        decoder_dropout: float = 0.0,
        use_layernorm: bool = True,
        use_residual_decoder: bool = False,
        min_count_for_variance: float = 1.0,
        fusion_mode: str = 'DA',
        decoder_spectral_norm: bool = False,  # spectral-norm the decoder (-> L-Lipschitz)
        **head_kwargs
    ):
        super().__init__()
        self.K = K
        self.d = d
        self.value_dim = value_dim
        self.use_residual_decoder = bool(use_residual_decoder)

        # Head
        head_defaults = dict(min_count=min_count_for_variance)
        head_defaults.update(head_kwargs)
        self.head = IntervalPatternNetwork(K=K, d=d, value_dim=value_dim,
                                           fusion_mode=fusion_mode, **head_defaults)

        # Decoder MLP. When ``decoder_spectral_norm`` is on we build a *certified*
        # decoder: every linear layer is spectrally normalized and all other layers
        # are 1-Lipschitz (ReLU only; no LayerNorm), so the product of the linear
        # spectral norms is a genuine Lipschitz upper bound L. This is the
        # "Certified-Lipschitz DiffInt" variant assumed by the clipping-margin
        # theorem. With spectral norm off, a LayerNorm is used for conditioning and
        # the bound is read structurally (the unconstrained mechanism).
        self.decoder_spectral_norm = bool(decoder_spectral_norm)
        decoder_layernorm = use_layernorm and not self.decoder_spectral_norm
        layers = []
        in_dim = K * value_dim
        for depth in range(decoder_depth):
            out_dim = decoder_hidden if depth < decoder_depth - 1 else d
            lin = nn.Linear(in_dim, out_dim)
            if self.decoder_spectral_norm:
                lin = nn.utils.spectral_norm(lin)
            layers.append(lin)
            if depth < decoder_depth - 1:
                if decoder_layernorm:
                    layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.ReLU(inplace=True))
                if decoder_dropout > 0:
                    layers.append(nn.Dropout(decoder_dropout))
            in_dim = out_dim
        self.decoder = nn.Sequential(*layers)

        # Identity code-norm in the certified variant keeps the whole decoder path
        # Lipschitz (LayerNorm is not 1-Lipschitz).
        if use_layernorm and not self.decoder_spectral_norm:
            self.encoder_norm = nn.LayerNorm(K * value_dim)
        else:
            self.encoder_norm = nn.Identity()
        # EMA of per-(unit,feature) support, maintained by the trainer for IA attention.
        if fusion_mode != 'IA':
            self.register_buffer("_support_ema", torch.ones(K, d), persistent=False)

    @torch.no_grad()
    def decoder_lipschitz(self):
        """Upper bound on the decoder Lipschitz constant L: the product of the
        spectral norms (largest singular values) of its linear layers (ReLU and
        LayerNorm are 1-Lipschitz). This is the L used in the clipping-margin bound."""
        L = 1.0
        for m in self.decoder:
            if isinstance(m, nn.Linear) or hasattr(m, "weight_orig"):
                L *= float(torch.linalg.matrix_norm(m.weight.detach(), ord=2))
        return L

    def encode(self, x: torch.Tensor, compute_stats: bool = False, y: Optional[torch.Tensor] = None):
        if compute_stats and (y is not None):
            out = self.head(x, y=y, compute_stats=True)
            (h, alpha, intervals), stats = out[:-1], out[-1]
            return h, alpha, intervals, stats
        h, alpha, intervals = self.head(x, y=y, compute_stats=False)
        return h, alpha, intervals, None

    def decode(self, flat: torch.Tensor) -> torch.Tensor:
        return self.decoder(flat)

    def reconstruct(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, compute_stats: bool = False):
        h, alpha, intervals, stats = self.encode(x, compute_stats=compute_stats, y=y)
        B = h.shape[0]
        flat = self.encoder_norm(h.reshape(B, -1))
        x_hat = self.decode(flat)
        if self.use_residual_decoder:
            x_hat = x + x_hat
        return x_hat, alpha, intervals, stats
