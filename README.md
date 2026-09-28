# DiffInt

**Differentiable Interval Bottlenecks for Interpretable Anomaly Detection in Numerical Data**
Lamine Diop and Marc Plantevit — EPITA Research Laboratory (LRE).
Reference implementation for the paper published at IEEE ICDM 2026.

DiffInt is a reconstruction autoencoder whose latent bottleneck is a set of
soft, axis-aligned **interval memberships** learned end-to-end from raw numerical
data (no discretization). Each latent unit is a learnable hyper-rectangle
(`feature j ∈ [a, b]`); an instance is encoded by how strongly it falls inside
each interval, and the per-sample reconstruction error is the anomaly score. The
interval structure makes the bottleneck directly inspectable, so a flagged point
comes with feature-range explanations rather than a single opaque score.

Key properties:
- **Interpretable by design**: the bottleneck *is* a set of readable intervals.
- **A deterministic guarantee**: out-of-support points are clipped to a fixed
  reconstruction (the clipping-margin property), which is why MAE is a valid score.
- **Required `[-1,1]` normalization**: calibrates initialization, the sigmoid
  temperature, and the guarantee; one configuration transfers across datasets.
- **Label-free importance (LFI)**: a closed-form score per (unit, feature)
  computed from quantities the trained model already keeps, used to surface
  candidate constraints without anomaly labels.

## Extended version

The extended version of the paper, with all proofs and the supplementary
material, is on arXiv:
[arXiv:2609.03878](https://arxiv.org/abs/2609.03878). It collects
the material that does not fit in the main text: dataset characteristics, the
**complete per-dataset ROC–AUC and AUPRC tables** (48 datasets × 23 methods,
mean ± std with ranks), per-scale critical-difference diagrams on both
metrics, the per-dataset robustness ablations (normalization, contamination,
τ sensitivity, active-coordinate threshold, inference clamping, LFI seed
stability, per-layer spectral norms), the complexity/scalability profile, and
the LFI faithfulness study.

## Install

```bash
git clone https://github.com/DiffInt/diffint
cd diffint
pip install -e .                      # or: pip install -r requirements.txt
```

Requires Python ≥ 3.8 and PyTorch (CPU is fine). Plotting interval-level
explanations additionally needs `seaborn`/`matplotlib`
(`pip install -e ".[explain]"`).

## Quickstart

```python
import numpy as np
from sklearn.metrics import roc_auc_score
from diffint import DiffInt

# X_train: inliers only (semi-supervised). X_test/y_test: mixed.
det = DiffInt(K=200, epochs=1000).fit(X_train)
scores = det.decision_function(X_test)        # higher = more anomalous
print("AUROC", roc_auc_score(y_test, scores))
```

Inputs are min–max scaled to `[-1, 1]` internally (and clipped to `[-1, 1]` at
inference); do not pre-scale. See [`examples/quickstart.py`](examples/quickstart.py).

### Reading the intervals

```python
for unit, feat, lo, hi in det.interval_rules(feature_names=cols)[:10]:
    print(f"unit {unit}: {feat} in [{lo:.2f}, {hi:.2f}]")
```

## Reproduce the ADBench evaluation

Place ADBench CSVs (feature columns + final 0/1 label column) in `csv/`:

```bash
python scripts/run_adbench.py --data_dir csv/ --K 200 \
    --epochs 1000 --lr 5e-5 --nb_runs 3
```

Protocol: semi-supervised (train on inliers), `[-1,1]` normalization for all
methods, 40% test split, mean AUROC/AUPRC over seeds.

## Package layout

```
diffint/
  estimator.py               # DiffInt: fit / decision_function / predict / interval_rules
  autoencoder.py             # interval-bottleneck autoencoder (+ optional spectral norm)
  intervalPatternNetwork.py  # soft memberships + pattern aggregator
  intervalparam.py           # interval centers + softplus half-widths
  utils.py                   # batched scoring helpers
scripts/run_adbench.py       # ADBench reproduction
examples/quickstart.py       # minimal example
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{diop2026diffint,
  title     = {Differentiable Interval Bottlenecks for Interpretable
               Anomaly Detection in Numerical Data},
  author    = {Diop, Lamine and Plantevit, Marc},
  booktitle = {IEEE International Conference on Data Mining (ICDM)},
  year      = {2026}
}
```

## Authors

- Lamine Diop — `lamine.diop@epita.fr`
- Marc Plantevit — `marc.plantevit@epita.fr`

EPITA Research Laboratory (LRE), Le Kremlin-Bicêtre, France.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Lamine Diop and Marc Plantevit.
