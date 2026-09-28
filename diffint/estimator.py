"""scikit-learn / PyOD-style estimator for DiffInt.

Example
-------
>>> from diffint import DiffInt
>>> det = DiffInt(K=200, fusion_mode="PA", epochs=1000).fit(X_inliers)
>>> scores = det.decision_function(X_test)   # higher = more anomalous
"""
from __future__ import annotations
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .autoencoder import DiffIntAutoencoder


def _build_scaler(kind):
    """Return (scaler, clip_range_or_None) for a normalization choice."""
    if kind == "minmax11":
        return MinMaxScaler(feature_range=(-1, 1)), (-1.0, 1.0)
    if kind == "minmax01":
        return MinMaxScaler(feature_range=(0, 1)), (0.0, 1.0)
    if kind == "standard":
        return StandardScaler(), None
    if kind == "none":
        return None, None
    raise ValueError(f"unknown scaler '{kind}'")


class DiffInt:
    """Differentiable interval-bottleneck autoencoder for interpretable anomaly detection.

    Trained semi-supervised on inliers; the per-sample reconstruction error (MAE)
    is the anomaly score. Inputs are min--max scaled to ``[-1, 1]`` internally
    (a required part of the method) and clipped to ``[-1, 1]`` at inference.

    Parameters
    ----------
    K : int
        Number of interval units (bottleneck width / maximum budget).
    value_dim : int
        Per-unit channel width. Forced to 1 for ``fusion_mode="PA"``.
    fusion_mode : {"PA", "IA", "DA"}
        Aggregator: pattern competition (default), feature attention, gated fusion.
    epochs, lr, batch_size, tau, rho : training hyper-parameters.
    scaler : {"minmax11", "minmax01", "standard", "none"}
        Input normalization (the method is designed for "minmax11").
    decoder_spectral_norm : bool
        Spectrally normalize the decoder, making it L-Lipschitz (Prop. 1).
    """

    def __init__(self, K=200, value_dim=1, fusion_mode="PA", epochs=1000, lr=5e-5,
                 batch_size=64, tau=0.1, rho=0.999, scaler="minmax11",
                 decoder_spectral_norm=False, device=None, random_state=None,
                 verbose=False):
        self.scaler = scaler
        self.decoder_spectral_norm = decoder_spectral_norm
        self.K = K
        self.value_dim = 1 if fusion_mode == "PA" else value_dim
        self.fusion_mode = fusion_mode
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.tau = tau
        self.rho = rho
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------ fit
    def fit(self, X, y=None):
        """Fit on inliers. If ``y`` is given, only rows with ``y == 0`` are used."""
        X = np.asarray(X, dtype=np.float32)
        if y is not None:
            X = X[np.asarray(y).ravel() == 0]
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        self.scaler_, self.clip_ = _build_scaler(self.scaler)
        if self.scaler_ is not None:
            self.scaler_.fit(X)
        Xs = self._transform(X)
        d = Xs.shape[1]
        n = Xs.shape[0]
        bs = self.batch_size
        if n > 20000:
            bs = 1024
        elif n > 10000:
            bs = 512

        dev = torch.device(self.device)
        Xt = torch.tensor(Xs, dtype=torch.float32, device=dev)
        model = DiffIntAutoencoder(K=self.K, d=d, value_dim=self.value_dim,
                                   fusion_mode=self.fusion_mode,
                                   decoder_spectral_norm=self.decoder_spectral_norm).to(dev)
        model.head.set_tau(self.tau)
        opt = optim.Adam(model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            model.train()
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, bs):
                xb = Xt[perm[s:s + bs]]
                x_hat, *_ = model.reconstruct(xb)
                loss = torch.sqrt(F.mse_loss(x_hat, xb) + 1e-8)
                opt.zero_grad()
                loss.backward()
                # EMA of per-(unit,feature) support, used by IA attention
                if self.fusion_mode != "IA":
                    with torch.no_grad():
                        low, high = model.head.intervals.intervals(device=xb.device, dtype=xb.dtype)
                        ind = model.head.soft_ind(xb, low, high)
                        bsup = ind.sum(dim=0) / float(max(1, xb.size(0)))
                        model._support_ema.mul_(self.rho).add_(bsup * (1.0 - self.rho))
                        model.head.support.copy_(model._support_ema)
                opt.step()
            if self.verbose and (epoch + 1) % max(1, self.epochs // 10) == 0:
                print(f"epoch {epoch+1}/{self.epochs}  loss {loss.item():.5f}")

        self.model_ = model
        self.n_features_in_ = d
        self.decoder_lipschitz_ = model.decoder_lipschitz()
        return self

    def _transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xs = self.scaler_.transform(X) if self.scaler_ is not None else X
        if self.clip_ is not None:
            Xs = np.clip(Xs, self.clip_[0], self.clip_[1])
        return Xs.astype(np.float32)

    # ------------------------------------------------------- scoring / predict
    @torch.no_grad()
    def decision_function(self, X):
        """Per-sample anomaly score (mean reconstruction error); higher = more anomalous."""
        Xs = self._transform(X)
        dev = torch.device(self.device)
        self.model_.eval()
        Xt = torch.tensor(Xs, dtype=torch.float32, device=dev)
        scores = []
        for s in range(0, Xt.size(0), 1024):
            xb = Xt[s:s + 1024]
            x_hat, *_ = self.model_.reconstruct(xb)
            scores.append((xb - x_hat).abs().mean(dim=1).cpu().numpy())
        return np.concatenate(scores)

    def predict(self, X, contamination=0.1):
        """Binary labels (1 = anomaly) by thresholding the top ``contamination`` fraction."""
        s = self.decision_function(X)
        thr = np.quantile(s, 1.0 - contamination)
        return (s >= thr).astype(int)

    def interval_rules(self, feature_names=None):
        """Return the learned intervals as ``(unit, feature, low, high)`` tuples in
        the original (un-scaled) feature space. Useful for inspection."""
        low, high = self.model_.head.intervals.intervals()
        low = low.detach().cpu().numpy(); high = high.detach().cpu().numpy()
        # invert the [-1,1] scaling back to data units
        lo = self.scaler_.inverse_transform(np.clip(low, -1, 1))
        hi = self.scaler_.inverse_transform(np.clip(high, -1, 1))
        keep = getattr(self.model_, "selected_idx_", None)
        units = keep if keep is not None else range(self.K)
        rules = []
        for k in units:
            for j in range(self.n_features_in_):
                name = feature_names[j] if feature_names is not None else f"x{j}"
                rules.append((int(k), name, float(lo[k, j]), float(hi[k, j])))
        return rules
