# Label-free importance (LFI): reproducing the global-ranking study

`global_ranking.py` reproduces the study of the global LFI ranking in the extended version
of the paper ([arXiv:2609.03878](https://arxiv.org/abs/2609.03878)): the table that compares
the zero-query LFI feature ranking with the same ranking obtained by aggregating post-hoc
attributions, and the agreement figure built from it.

```bash
pip install -e ".[experiments]"          # adds shap
python experiments/lfi/global_ranking.py  # about 30 min on 3 CPU cores
python experiments/lfi/global_ranking.py --quick   # one run, about 4 min
```

The script prints the table and writes every run (per-method metrics and attribution
vectors) to `results/lfi_global_ranking.json`.

## Setting

- Three controlled families of synthetic data: d = 28, 20 and 36, with 14, 6 and 10
  coordinates tightly distributed on inliers among broad ones. Every anomaly is a marginal
  violation of the tight coordinates, so the relevant coordinates are known.
- Seven re-trainings: seeds 0, 1 and 2 for the first two families, seed 0 for the third.
- DiffInt with K = 200, τ = 0.1, lr = 5·10⁻⁵, batch 64, ρ = 0.999 and 350 epochs, on data
  already scaled to [−1, 1].
- LFI as in `diffint.lfi`, aggregated over units (maximum) into a feature ranking, with zero
  model queries.
- Post-hoc rankings, each aggregated as the mean absolute attribution over 120 anomalies:
  - gradient × input;
  - integrated gradients, 24 steps;
  - KernelSHAP, 40 background points and 120 samples;
  - ECOD tail surprise.

## Metrics

| Column | Meaning |
|---|---|
| sep-AUC | does the ranking put the relevant coordinates first? |
| agree | Spearman ρ with the LFI ranking |
| stab. | mean Spearman ρ between re-trainings of the same family |
| sanity | ρ between the LFI ranking and that of a weight-randomised model whose support is re-estimated the same way (low is good) |
| Gini | concentration of the ranking vector |
| queries | model passes per pooled anomaly |

## Expected output

Reported in the paper:

| Method | sep-AUC | agree | stab. | sanity | Gini | queries |
|---|---|---|---|---|---|---|
| LFI | 1.00 | -- | 0.71 | +0.01 | 0.11 | 0 |
| gradient × input | 1.00 | +0.71 | 0.69 | -- | 0.33 | 1 |
| integrated gradients | 1.00 | +0.69 | 0.66 | -- | 0.45 | 24 |
| KernelSHAP | 1.00 | +0.72 | 0.67 | -- | 0.58 | 120 |
| ECOD tail | 1.00 | +0.74 | 0.67 | -- | 0.47 | 0 |

The model in this package is the one used for the paper: with the same seed it has the same
initial parameters, is trained by the same updates and computes the same function, so the
script reproduces these values. Floating-point results depend on the platform: the values
are reproduced exactly on an x86-64 CPU with PyTorch 2.11; other processors or PyTorch
versions give close but not identical values.
