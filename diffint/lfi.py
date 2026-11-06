"""Label-free importance (LFI) of the learned intervals.

For latent unit k and feature j, with s_kj the EMA soft support of the interval on
inliers, w_kj the width of the learned interval and range_j the width of the scaled
feature range (2 on [-1, 1]):

    LFI_kj = s_kj * (1 - w_kj / range_j) * stab_kj   in [0, 1]

The three factors are how often the interval fires on inliers (support), how much
narrower it is than the feature range (tightness) and how repeatable it is across
re-trainings (stability, 1 for a single trained model). The width is the full width
w_kj = 2 * Delta_kj of the interval [m_kj - Delta_kj, m_kj + Delta_kj]; this is the
quantity behind the LFI results of the extended version of the paper.

LFI is a global, model-level score: it ranks the learned constraints and, aggregated
over units, the features. It uses no anomaly label and no model query. It does not
explain a single point.
"""
import numpy as np


def _module(model):
    """Accept a fitted ``DiffInt`` estimator or a ``DiffIntAutoencoder``."""
    return getattr(model, "model_", model)


def label_free_importance(model, stab=None, feature_range=2.0):
    """LFI per (unit, feature), shape (K, d).

    Parameters
    ----------
    model : fitted ``DiffInt`` or ``DiffIntAutoencoder``
    stab : array of shape (K, d), optional
        Cross-seed stability of each interval; 1 (no effect) when omitted.
    feature_range : float
        Width of the scaled feature range; 2 for the default ``[-1, 1]`` scaling.
    """
    m = _module(model)
    low, high = m.head.intervals.intervals()
    width = (high - low).detach().cpu().numpy()
    support = m.head.support.detach().cpu().numpy()
    lfi = support * (1.0 - np.clip(width, 0.0, feature_range) / feature_range)
    if stab is not None:
        lfi = lfi * np.asarray(stab)
    return lfi


def feature_importance(lfi):
    """Global per-feature ranking: the largest LFI of each feature over units, shape (d,)."""
    return np.asarray(lfi).max(axis=0)


def top_units(lfi, n_features=4, n_units=1):
    """Units ranked by the sum of their ``n_features`` largest LFI values.

    Returns a list of ``(unit, features, score)``, where ``features`` are the indices of
    the ``n_features`` features with the largest LFI in that unit.
    """
    lfi = np.asarray(lfi)
    feats = np.argsort(-lfi, axis=1)[:, :n_features]
    score = np.take_along_axis(lfi, feats, axis=1).sum(axis=1)
    return [(int(k), feats[k].tolist(), float(score[k])) for k in np.argsort(-score)[:n_units]]
