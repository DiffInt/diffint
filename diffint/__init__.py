"""DiffInt: differentiable interval bottlenecks for interpretable anomaly detection.

Reference implementation for the paper "Differentiable Interval Bottlenecks for
Interpretable Anomaly Detection in Numerical Data" (IEEE ICDM 2026).
"""
from .estimator import DiffInt
from .autoencoder import DiffIntAutoencoder
from .intervalPatternNetwork import IntervalPatternNetwork

__all__ = ["DiffInt", "DiffIntAutoencoder", "IntervalPatternNetwork"]
__version__ = "0.1.0"
__author__ = "Lamine Diop and Marc Plantevit"
__license__ = "MIT"
