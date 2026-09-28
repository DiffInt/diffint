"""Optional ICR-based explanation utilities for DiffInt (intervals, ICR matrices,
scatter-matrix plots). Requires the ``explain`` extra (seaborn, matplotlib);
not imported by the package by default."""
import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib.patches import Rectangle
import logging
import warnings
warnings.filterwarnings(
    "ignore",
    message="Attempting to set identical low and high xlims makes transformation singular"
)

# -------------------------
# Robust interval extraction (handles several model naming conventions)
# -------------------------
def extract_intervals(model, device='cpu'):
    """
    Try several attribute names to extract interval centers and widths.
    Returns (lowers, uppers) numpy arrays (K,d) clipped to [-1,1].
    """
    model.to(device).eval()
    with torch.no_grad():
        # 1) encoder_layer.centers / deltas / r_raw (common)
        if hasattr(model, "encoder_layer"):
            enc = model.encoder_layer
            if hasattr(enc, "centers") and hasattr(enc, "deltas"):
                centers = enc.centers.detach().cpu().numpy()
                deltas = F.softplus(enc.deltas).detach().cpu().numpy()
                if hasattr(enc, "r_raw"):
                    r = float(F.softplus(enc.r_raw).item())
                    widths = r * deltas
                else:
                    widths = deltas
                lowers = centers - widths
                uppers = centers + widths
                return np.clip(lowers, -1, None), np.clip(uppers, None, 1)

            # fallback naming m / log_w
            if hasattr(enc, "m") and hasattr(enc, "log_w"):
                centers = enc.m.detach().cpu().numpy()
                log_w = enc.log_w.detach().cpu().numpy()
                widths = np.log1p(np.exp(log_w)) + 1e-6
                lowers = centers - widths
                uppers = centers + widths
                return np.clip(lowers, -1, None), np.clip(uppers, None, 1)

        # 2) head.intervals (IntervalParam style)
        if hasattr(model, "head") and hasattr(model.head, "intervals"):
            ip = model.head.intervals
            # try attributes m / log_w
            if hasattr(ip, "m") and hasattr(ip, "log_w"):
                centers = ip.m.detach().cpu().numpy()
                log_w = ip.log_w.detach().cpu().numpy()
                widths = np.log1p(np.exp(log_w)) + 1e-6
                lowers = centers - widths
                uppers = centers + widths
                return np.clip(lowers, -1, None), np.clip(uppers, None, 1)

        # 3) head.intervals.intervals() -> (low, high)
        try:
            if hasattr(model, "head") and hasattr(model.head, "intervals") and callable(getattr(model.head.intervals, "intervals", None)):
                low, high = model.head.intervals.intervals(device=device, dtype=torch.float32)
                lowers = low.detach().cpu().numpy()
                uppers = high.detach().cpu().numpy()
                return np.clip(lowers, -1, None), np.clip(uppers, None, 1)
        except Exception:
            pass

        # 4) final fallback: attributes low/high on model
        if hasattr(model, "low") and hasattr(model, "high"):
            lowers = getattr(model, "low")
            uppers = getattr(model, "high")
            if torch.is_tensor(lowers):
                lowers = lowers.detach().cpu().numpy()
            if torch.is_tensor(uppers):
                uppers = uppers.detach().cpu().numpy()
            return np.clip(lowers, -1, None), np.clip(uppers, None, 1)

        raise AttributeError(
            "Could not find interval parameters on model. Checked:"
            " encoder_layer.centers/deltas, encoder_layer.m/log_w, head.intervals.m/log_w,"
            " head.intervals.intervals(), and model.low/model.high."
        )

# -------------------------
# Compute interval counts for coverage / scoring
# -------------------------
def compute_interval_counts(
    lowers: np.ndarray,
    uppers: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[Dict[Tuple[int,int], Dict[str, float]], float, float]:
    """
    For each (k,j) compute:
     - P_in, N_in, P_out, N_out
     - support, in_cover
     - score_best = (1 - ICR) * log(1 + N_in)  (as requested)
    Returns coverage dict and totals total_P,total_N
    """
    n, d = X.shape
    K = lowers.shape[0]
    total_P = float(np.sum(y == 1))
    total_N = float(np.sum(y == 0))
    coverage: Dict[Tuple[int,int], Dict[str, float]] = {}

    for k in range(K):
        for j in range(d):
            a, b = float(lowers[k, j]), float(uppers[k, j])
            mask = (X[:, j] >= a) & (X[:, j] <= b)
            if not np.any(mask):
                P_in = 0.0
                N_in = 0.0
            else:
                P_in = float(np.sum(y[mask] == 1))
                N_in = float(np.sum(y[mask] == 0))
            P_out = max(total_P - P_in, 0.0)
            N_out = max(total_N - N_in, 0.0)
            support = P_in + N_in
            R_in = P_in / (P_in + N_in + 1e-12)
            R_out = P_out / (P_out + N_out + 1e-12)
            ICR = R_in / (R_in + R_out + 1e-12)
            # score_kj as requested:
            score_best = (1.0 - ICR) * math.log(1.0 + N_in + 1e-12)
            in_cover = N_in / (N_in + N_out + 1e-8) if (N_in + N_out) > 0 else 0.0

            coverage[(k, j)] = {
                "a": a, "b": b,
                "P_in": P_in, "N_in": N_in,
                "P_out": P_out, "N_out": N_out,
                "support": support,
                "R_in": R_in, "R_out": R_out,
                "ICR": ICR,
                "in_cover": in_cover,
                "score_best": score_best
            }
    return coverage, total_P, total_N

# -------------------------
# Refine patterns by selecting top intervals per pattern
# -------------------------
def refine_all_patterns_by_icr(
    model=None,
    device: str = "cpu",
    lowers: Optional[np.ndarray] = None,
    uppers: Optional[np.ndarray] = None,
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    *,
    min_support: int = 1,
    per_pattern: bool = True,
    eps: float = 1e-12,
) -> Tuple[List[Dict[str, Any]], Dict[int, str]]:
    """
    Build a list of candidate intervals with metrics for all (k,j).
    Returns selected list (flattened entries) and a pattern_intervals mapping (k -> conjunction string).
    Note: this function does not limit top_k/top_N — selection is done later.
    """
    if X is None or y is None:
        raise ValueError("X and y must be provided.")
    if lowers is None or uppers is None:
        if model is None:
            raise ValueError("Either model or lowers/uppers must be provided.")
        lowers, uppers = extract_intervals(model, device=device)

    coverage, total_P, total_N = compute_interval_counts(lowers, uppers, X, y)
    K, d = lowers.shape

    candidates_by_pattern: Dict[int, List[Dict[str, Any]]] = {}
    for (k, j), entry in coverage.items():
        if entry["support"] < min_support:
            continue
        r = dict(entry)  # copy
        r.update({"pattern": int(k), "dim": int(j)})
        candidates_by_pattern.setdefault(int(k), []).append(r)

    # Flatten into list and also build simple pattern_intervals (all intervals per pattern)
    selected_flat = []
    pattern_intervals_all: Dict[int, str] = {}
    for k, lst in candidates_by_pattern.items():
        # sort intervals for readability by dim
        top_sorted_by_dim = sorted(lst, key=lambda r: r["dim"])
        parts = [f"{r['a']:.2f} ≤ x[{r['dim']}] ≤ {r['b']:.2f}" for r in top_sorted_by_dim]
        pattern_intervals_all[int(k)] = " AND ".join(parts) if parts else ""
        selected_flat.extend(lst)

    return selected_flat, pattern_intervals_all

# -------------------------
# Select top_K patterns and for each pick top_N intervals
# -------------------------
def select_top_patterns_and_intervals(
    candidates_flat: List[Dict[str, Any]],
    top_K: int = 4,
    top_N: int = 3
) -> Tuple[List[int], Dict[int, List[Dict[str, Any]]]]:
    """
    candidates_flat: list of entries with keys including 'pattern','dim','ICR','N_in','score_best' (from compute_interval_counts)
    Returns:
      - top_pattern_ids: list of selected pattern indices (length <= top_K)
      - selected_intervals_per_pattern: dict pattern -> list of top_N interval dicts (sorted by score_best desc)
    """
    if len(candidates_flat) == 0:
        return [], {}

    # group by pattern
    by_pattern: Dict[int, List[Dict[str, Any]]] = {}
    for r in candidates_flat:
        by_pattern.setdefault(int(r["pattern"]), []).append(r)

    # compute average ICR per pattern (higher ICR means better separation of positives)
    avg_icr_per_pattern: Dict[int, float] = {}
    for k, lst in by_pattern.items():
        avg_icr_per_pattern[k] = float(np.mean([x["ICR"] for x in lst])) if len(lst) > 0 else 0.0

    # choose top_K patterns by avg ICR descending
    top_patterns_sorted = sorted(avg_icr_per_pattern.items(), key=lambda x: x[1], reverse=True)[:top_K]
    top_pattern_ids = [k for k, _ in top_patterns_sorted]

    # for each selected pattern pick top_N intervals by score_best descending (score_best = (1-ICR)*log(1+N_in))
    selected_intervals_per_pattern: Dict[int, List[Dict[str, Any]]] = {}
    for k in top_pattern_ids:
        lst = by_pattern.get(k, [])
        # sort by score_best descending (larger score_best means preferred in your scoring)
        top_intervals = sorted(lst, key=lambda r: r["score_best"], reverse=True)[:top_N]
        selected_intervals_per_pattern[k] = top_intervals

    return top_pattern_ids, selected_intervals_per_pattern

# -------------------------
# Plot pattern scatter matrix for a single pattern using its selected intervals
# -------------------------
def plot_single_pattern_scatter_matrix(
    X: np.ndarray,
    y: np.ndarray,
    pattern_id: int,
    intervals: List[Dict[str, Any]],
    output_prefix: str = "dataset",
    inlier_color: str = "purple",
    outlier_color: str = "yellow",
    max_bins: int = 15,
    agg: str = "Lse",
    out_plot: str = "plots/",
    run: int = 0
) -> str:
    """
    Plot scatter matrix showing only dimensions present in the 'intervals' list for the given pattern.
    intervals: list of dicts with keys 'dim','a','b','N_in','ICR','score_best' etc.
    Returns saved filename.
    """
    os.makedirs(out_plot, exist_ok=True)

    if len(intervals) == 0:
        raise ValueError("No intervals provided for plotting.")

    dims = sorted({int(r["dim"]) for r in intervals})
    X_sel = X[:, dims]
    col_names = [f"x[{d}]" for d in dims]
    df_plot = pd.DataFrame(X_sel, columns=col_names)
    labels = np.asarray(y).astype(int)
    colors = [inlier_color if lbl == 0 else outlier_color for lbl in labels]

    axes = scatter_matrix(df_plot, figsize=(8, 8), diagonal='hist', color=colors, alpha=0.9)
    # set ticks
    for ax in axes.flatten():
        ax.set_xticks(np.linspace(-1, 1, 5))
        ax.set_yticks(np.linspace(-1, 1, 5))

    # build mapping dim->(a,b) using chosen intervals
    pattern_map = {int(r["dim"]): (r["a"], r["b"]) for r in intervals}

    num_dims = len(dims)
    for i in range(num_dims):
        for j in range(num_dims):
            ax = axes[j, i] if axes.ndim == 2 else axes
            dim_i = dims[i]
            dim_j = dims[j]
            if (dim_i in pattern_map) and (dim_j in pattern_map) and (i != j):
                lx, hx = pattern_map[dim_i]
                ly, hy = pattern_map[dim_j]
                lx, hx = max(-1, lx), min(1, hx)
                ly, hy = max(-1, ly), min(1, hy)
                rect = Rectangle((lx, ly), hx - lx, hy - ly, fill=False, edgecolor='green', linewidth=1.2)
                ax.add_patch(rect)
            elif i == j:
                ax.clear()
                x_data = df_plot.iloc[:, i]
                x_in = x_data[labels == 0]
                x_out = x_data[labels == 1]
                bins = np.linspace(-1., 1., max_bins)
                ax.hist(x_in, bins=bins, alpha=1., color=inlier_color, label='0')
                ax.hist(x_out, bins=bins, alpha=1., color=outlier_color, label='1')
                ax.set_xlabel(col_names[i])

    # create a friendly filename including pattern id and the intervals' score summaries
    # compute an average combined score for the pattern for filename
    mean_score = np.mean([r["score_best"] for r in intervals]) if len(intervals) > 0 else 0.0
    score_tag = f"score_{mean_score:.6f}"
    fname = os.path.join(out_plot, f"{output_prefix}_run_{run}_pattern{pattern_id}_{score_tag}_{agg}_scatter_matrix.eps")

    plt.suptitle(f"Pattern {pattern_id} (top {len(intervals)} intervals) ({agg})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    return fname


# -------------------------
# Compute ICR matrix for plotting
# -------------------------
def compute_icr(model, X_val, y_val, device='cpu', epsilon=1e-8):
    if torch.is_tensor(X_val):
        X = X_val.cpu().numpy()
    else:
        X = np.array(X_val)
    if torch.is_tensor(y_val):
        y = y_val.cpu().numpy()
    else:
        y = np.array(y_val)

    lowers, uppers = extract_intervals(model, device=device)
    K, d = lowers.shape
    icr = np.zeros((K, d))
    for k in range(K):
        for j in range(d):
            a, b = lowers[k, j], uppers[k, j]
            inside = (X[:, j] >= a) & (X[:, j] <= b)
            outside = ~inside
            P_in = np.sum((y == 1) & inside)
            N_in = np.sum((y == 0) & inside)
            P_out = np.sum((y == 1) & outside)
            N_out = np.sum((y == 0) & outside)
            R_in = P_in / (P_in + N_in + epsilon)
            R_out = P_out / (P_out + N_out + epsilon)
            icr[k, j] = R_in / (R_in + R_out + epsilon)
    return icr

# -------------------------
# Plot ICR heatmap
# -------------------------
def plot_icr_matrix(data, agg, data_name, vmax_percentile=99.5, out_plot: str = "plots/", run: int = 0):
    os.makedirs(out_plot, exist_ok=True)
    data = np.array(data).copy()
    vmax = np.percentile(data, vmax_percentile)
    vmin = np.min(data)
    plt.figure(figsize=(10, 6))
    sns.heatmap(data, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax, annot=False)
    plt.xlabel("Feature Dimension (d)", fontsize=12)
    plt.ylabel("Pattern Index (K)", fontsize=12)
    plt.title(f"ICR Matrix ({data_name} under {agg})", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(out_plot, f"{data_name}_run_{run}_{agg}_icr_matrix.eps"), dpi=300)
    plt.close()

# -------------------------
# Top-level explain function to call from your code
# -------------------------
def explain_DiffInt(
    model,
    X: np.ndarray,
    y: np.ndarray,
    data_name: str = "dataset",
    agg: str = "Lse",
    device: str = "cpu",
    top_K: int = 4,
    top_N: int = 3,
    min_support: int = 1,
    out_plot: str = "plots/",
    run: int = 0
):
    """
    Main explainability pipeline:
     - extract intervals
     - compute coverage and candidate intervals (with score_kj)
     - select top_K patterns
     - for each selected pattern pick top_N intervals and plot scatter matrix (filename contains combined score)
     - plot ICR heatmap for all patterns
    """
    os.makedirs(out_plot, exist_ok=True)
    # extract intervals once (will raise a helpful error if not found)
    lowers, uppers = extract_intervals(model, device=device)

    # compute candidates and counts
    candidates_flat, _ = refine_all_patterns_by_icr(
        model=model, device=device,
        lowers=lowers, uppers=uppers,
        X=X, y=y,
        min_support=min_support,
        per_pattern=True
    )

    if len(candidates_flat) == 0:
        logging.info("No candidate intervals found (maybe min_support too large). Exiting.")
        return

    # select top patterns and intervals
    top_pattern_ids, selected_intervals_per_pattern = select_top_patterns_and_intervals(candidates_flat, top_K=top_K, top_N=top_N)

    logging.info(f"Selected top patterns: {top_pattern_ids}")
    for k in top_pattern_ids:
        intervals = selected_intervals_per_pattern.get(k, [])
        # produce a readable rule string and print some stats
        parts = [f"{r['a']:.2f} ≤ x[{r['dim']}] ≤ {r['b']:.2f} (N_in={int(r['N_in'])}, ICR={r['ICR']:.3f}, score={r['score_best']:.3f})" for r in intervals]
        logging.info(f"Pattern {k} top intervals:\n  " + "\n  ".join(parts))

        # plot scatter matrix for this pattern using its top_N intervals
        fname = plot_single_pattern_scatter_matrix(
            X=X, y=y,
            pattern_id=k,
            intervals=intervals,
            output_prefix=data_name,
            inlier_color="purple",
            outlier_color="yellow",
            agg=agg,
            out_plot=out_plot,
            run=run
        )
        logging.info(f"Saved scatter matrix: {fname}")


    # finally plot ICR matrix for all patterns
    icr = compute_icr(model, X, y, device=device)
    plot_icr_matrix(icr, agg, data_name, out_plot=out_plot,
                    run=run)
    logging.info("Saved ICR heatmap.")
    return top_pattern_ids



def summarize_topK_patterns_stat(model, X: np.ndarray, y: np.ndarray, top_pattern_ids: list, device: str = "cpu"):
    """
    Compute summary statistics for top-K patterns:
        - Mean ICR per pattern
        - Mean variance Var(ICR) per pattern
        - Total support (sum of soft membership over all intervals)
        - Mean z-score (weighted by interval support)

    Args:
        model: trained DiffInt model
        X: np.ndarray of shape (n_samples, D)
        y: np.ndarray of shape (n_samples,)
        top_pattern_ids: list of selected pattern indices
        device: 'cpu' or 'cuda'

    Returns:
        pd.DataFrame with columns ['Pattern', 'Mean ICR', 'Variance', 'Total Support', 'Mean z-score']
    """
    with torch.no_grad():
        # Convert to tensor
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device)

        # 1) Intervals
        low, high = model.head.intervals.intervals(device=device, dtype=X_t.dtype)

        # 2) Soft membership
        ind = model.head.soft_ind(X_t, low, high)  # (n_samples, K, D)
        ind_np = ind.detach().cpu().numpy()        # (n_samples, K, D)

        # 3) Compute ICR, variance, z-score per interval
        stats = model.head._compute_icr_and_var(ind, y_t)
        icr_kd = stats["icr"].detach().cpu().numpy()      # (K,D)
        var_kd = stats["var_icr"].detach().cpu().numpy()  # (K,D)
        z_kd = stats["z_score"].detach().cpu().numpy()     # (K,D)

    rows = []
    n_samples, K, D = ind_np.shape

    for k in top_pattern_ids:
        support_d = ind_np[:, k, :].sum(axis=0)  # (D,)
        total_support = support_d.sum()
        mean_icr = icr_kd[k].mean()
        mean_var = var_kd[k].mean()
        mean_z = np.sum(z_kd[k] * support_d) / (total_support + 1e-9)

        rows.append({
            "Pattern": k,
            "Mean ICR": mean_icr,
            "Variance": mean_var,
            "Total Support": total_support,
            "Mean z-score": mean_z
        })

    df = pd.DataFrame(rows).sort_values("Mean z-score", ascending=False).reset_index(drop=True)
    return df


