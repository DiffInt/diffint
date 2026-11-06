"""Global LFI ranking vs. aggregated post-hoc attributions (extended version, Table
"Quantitative study of the global LFI ranking" and its agreement figure).

Three controlled families of synthetic data in DiffInt's regime: d in {20, 28, 36}, with
6 to 14 coordinates tightly distributed on inliers among broad ones, and every anomaly a
marginal violation of the tight coordinates, so the relevant coordinates are known.
Seven re-trainings in total (seeds 0-2, 0-2, 0). For each trained model, the global LFI
feature vector (max over units, zero model queries) is compared with the same global
ranking obtained by aggregating, over a pool of 120 anomalies, the mean absolute
attribution of gradient x input, integrated gradients (24 steps), KernelSHAP (40
background points, 120 samples) and the ECOD tail surprise -log 2 min(F, 1 - F).

Reported per method: sep-AUC (relevant coordinates ranked first), agreement with the LFI
ranking (Spearman rho), cross-seed rank stability, model-randomisation sanity check for
LFI (rho with a weight-randomised model whose support is re-estimated the same way),
Gini concentration and model queries per pooled anomaly.

Training follows the experiment of the paper: K = 200, tau = 0.1, lr = 5e-5, batch 64,
rho = 0.999, 350 epochs, on data already scaled to [-1, 1] (scaler="none").

Usage:
    python experiments/lfi/global_ranking.py [--out results/lfi_global_ranking.json]
    python experiments/lfi/global_ranking.py --quick   # one run (first family, seed 0)
Requires shap (pip install -e ".[experiments]"). About 30 min on 3 CPU cores; --quick
takes about 4 min.
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from diffint import DiffInt, label_free_importance, feature_importance  # noqa: E402

TRAIN = dict(K=200, epochs=350, lr=5e-5, batch_size=64, tau=0.1,
             rho=0.999, scaler="none", device="cpu")
POOL, BACKGROUND, SHAP_SAMPLES, IG_STEPS = 120, 40, 120, 24
FAMILIES = [  # name, d, tight coordinates, inlier sd of the tight ones, anomaly shift, seeds
    ("synth-tight-A", 28, 14, 0.05, (0.12, 0.28), [0, 1, 2]),
    ("synth-tight-B", 20, 6, 0.04, (0.20, 0.40), [0, 1, 2]),
    ("synth-tight-C", 36, 10, 0.06, (0.15, 0.30), [0]),
]
METHODS = ["LFI", "grad x input", "integrated gradients", "KernelSHAP", "ECOD tail"]
QUERIES = {"LFI": 0, "grad x input": 1, "integrated gradients": IG_STEPS,
           "KernelSHAP": SHAP_SAMPLES, "ECOD tail": 0}
# values reported in the extended version of the paper
PAPER = {"LFI": (1.00, None, 0.71, 0.01, 0.11), "grad x input": (1.00, 0.71, 0.69, None, 0.33),
         "integrated gradients": (1.00, 0.69, 0.66, None, 0.45),
         "KernelSHAP": (1.00, 0.72, 0.67, None, 0.58), "ECOD tail": (1.00, 0.74, 0.67, None, 0.47)}


# ---------------------------------------------------------------- data
def family(d, n_tight, tight_sd, shift, seed):
    """n_tight tight coordinates and d - n_tight broad ones; anomalies shift the tight ones."""
    r = np.random.default_rng(1000 + seed)

    def draw(n):
        return np.c_[r.normal(0.0, tight_sd, (n, n_tight)),
                     r.uniform(-1.0, 1.0, (n, d - n_tight))].astype(np.float32)

    Xin, Xte, Xout = draw(3500), draw(900), draw(300)
    Xout[:, :n_tight] += r.choice([-1, 1], (300, n_tight)) * r.uniform(*shift, (300, n_tight))
    sc = MinMaxScaler((-1, 1)).fit(Xin)
    f = lambda A: np.clip(sc.transform(A), -3, 3).astype(np.float32)
    relevant = np.r_[np.ones(n_tight, bool), np.zeros(d - n_tight, bool)]
    return f(Xin), f(Xte), f(Xout), relevant


# ---------------------------------------------------------------- attributions
def score(model, X):
    with torch.no_grad():
        xt = torch.tensor(np.asarray(X), dtype=torch.float32)
        out = []
        for s in range(0, len(xt), 4096):
            xh, *_ = model.reconstruct(xt[s:s + 4096])
            out.append(torch.mean(torch.abs(xt[s:s + 4096] - xh), 1).numpy())
    return np.concatenate(out)


def grad_x_input(model, X, base):
    xt = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    xh, *_ = model.reconstruct(xt)
    torch.mean(torch.abs(xt - xh), 1).sum().backward()
    return np.abs(xt.grad.numpy() * (X - base[None]))


def integrated_gradients(model, X, base, steps=IG_STEPS):
    total = np.zeros_like(X)
    for a in np.linspace(1.0 / steps, 1.0, steps):
        xt = torch.tensor(base[None] + a * (X - base[None]), dtype=torch.float32, requires_grad=True)
        xh, *_ = model.reconstruct(xt)
        torch.mean(torch.abs(xt - xh), 1).sum().backward()
        total += xt.grad.numpy()
    return np.abs((X - base[None]) * total / steps)


def kernel_shap(model, X, background):
    import shap
    ex = shap.KernelExplainer(lambda Z: score(model, Z), background, silent=True)
    return np.abs(np.asarray(ex.shap_values(X, nsamples=SHAP_SAMPLES, silent=True)))


def ecod_tail(Xtr, X):
    n, out = len(Xtr), np.zeros_like(X)
    for j in range(X.shape[1]):
        col = np.sort(Xtr[:, j])
        left = np.searchsorted(col, X[:, j], "right") / (n + 1)
        right = 1.0 - np.searchsorted(col, X[:, j], "left") / (n + 1)
        out[:, j] = -np.log(2.0 * np.minimum(np.clip(left, 1e-6, 1), np.clip(right, 1e-6, 1)))
    return out


def randomised_control(model, Xtr, rho=TRAIN["rho"], passes=3):
    """weight-randomised copy whose support is re-estimated as during training"""
    m = copy.deepcopy(model)
    with torch.no_grad():
        # the model used for the paper also held two unused (K,) parameter vectors, which
        # were randomised first; drawing them keeps the random weights of the paper
        for _ in range(2):
            torch.randn(m.K)
        for p in m.parameters():
            p.copy_(torch.randn_like(p) * p.std().clamp_min(1e-3))
        m.eval()
        xt = torch.tensor(Xtr, dtype=torch.float32)
        low, high = m.head.intervals.intervals()
        ema = torch.ones_like(m.head.support)
        for _ in range(passes):
            for s in range(0, len(xt), 512):
                ema.mul_(rho).add_(m.head.soft_ind(xt[s:s + 512], low, high).mean(0) * (1.0 - rho))
        m.head.support.copy_(ema)
    return m


def gini(v):
    v = np.sort(np.abs(v)) + 1e-12
    n = len(v)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))


# ---------------------------------------------------------------- one run
def run(name, d, n_tight, tight_sd, shift, seed):
    Xtr, Xte, Xout, relevant = family(d, n_tight, tight_sd, shift, seed)
    rng = np.random.default_rng(seed)
    model = DiffInt(random_state=seed, **TRAIN).fit(Xtr).model_
    s = np.r_[score(model, Xte), score(model, Xout)]
    det_auc = float(roc_auc_score(np.r_[np.zeros(len(Xte)), np.ones(len(Xout))], s))
    base, pool = np.median(Xtr, 0), Xout[:POOL]
    background = Xtr[rng.choice(len(Xtr), BACKGROUND, replace=False)]
    vec = {"LFI": feature_importance(label_free_importance(model)),
           "grad x input": grad_x_input(model, pool, base).mean(0),
           "integrated gradients": integrated_gradients(model, pool, base).mean(0),
           "ECOD tail": ecod_tail(Xtr, pool).mean(0),
           "KernelSHAP": kernel_shap(model, pool, background).mean(0)}
    control = feature_importance(label_free_importance(randomised_control(model, Xtr)))
    rows = {m: dict(sep_auc=float(roc_auc_score(relevant.astype(int), v)), gini=gini(v),
                    agree=None if m == "LFI" else float(spearmanr(vec["LFI"], v).correlation))
            for m, v in vec.items()}
    rows["LFI"]["sanity"] = float(spearmanr(vec["LFI"], control).correlation)
    return dict(dataset=name, seed=seed, det_auc=det_auc, rows=rows,
                vectors={m: v.tolist() for m, v in vec.items()})


def summarise(runs):
    table = {}
    for m in METHODS:
        rr = [r["rows"][m] for r in runs]
        pairs = []
        for name in {r["dataset"] for r in runs}:
            vs = [np.array(r["vectors"][m]) for r in runs if r["dataset"] == name]
            pairs += [spearmanr(vs[i], vs[j]).correlation
                      for i in range(len(vs)) for j in range(i + 1, len(vs))]
        table[m] = dict(sep_auc=float(np.mean([x["sep_auc"] for x in rr])),
                        agree=None if m == "LFI" else float(np.mean([x["agree"] for x in rr])),
                        stability=float(np.mean(pairs)) if pairs else None,
                        sanity=float(np.mean([x["sanity"] for x in rr])) if m == "LFI" else None,
                        gini=float(np.mean([x["gini"] for x in rr])), queries=QUERIES[m])
    return table


def markdown(table):
    f = lambda v, fmt: "--" if v is None else format(v, fmt)
    lines = ["| Method | sep-AUC | agree | stab. | sanity | Gini | queries | paper (sep / agree / stab. / sanity / Gini) |",
             "|---|---|---|---|---|---|---|---|"]
    for m, t in table.items():
        p = " / ".join(f(v, ".2f") for v in PAPER[m])
        lines.append(f"| {m} | {t['sep_auc']:.2f} | {f(t['agree'], '+.2f')} | {f(t['stability'], '.2f')} | "
                     f"{f(t['sanity'], '+.2f')} | {t['gini']:.2f} | {t['queries']} | {p} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/lfi_global_ranking.json")
    ap.add_argument("--quick", action="store_true", help="first family, seed 0 only")
    args = ap.parse_args()
    t0, runs = time.time(), []
    families = FAMILIES[:1] if args.quick else FAMILIES
    for name, d, n_tight, tight_sd, shift, seeds in families:
        for seed in (seeds[:1] if args.quick else seeds):
            runs.append(run(name, d, n_tight, tight_sd, shift, seed))
            print(f"[{name} seed {seed}] {time.time() - t0:.0f}s", flush=True)
    table = summarise(runs)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(dict(train=TRAIN, runs=runs, table=table), open(args.out, "w"), indent=1)
    print(markdown(table))


if __name__ == "__main__":
    main()
