#!/usr/bin/env python3
import os
import glob
import time
import logging
import torch
import csv
import pandas as pd

# -------------------------
# Utilities
# -------------------------
@torch.no_grad()
def compute_recon_scores(model, X_tensor, batch=512, device='cpu'):
    """
    Returns: (mae_per_sample (numpy), reconstructions (numpy))
    """
    model = model.to(device)
    model.eval()
    maes = []
    recons = []
    X_tensor = X_tensor.to(device)
    for i in range(0, X_tensor.size(0), batch):
        xb = X_tensor[i:i+batch]
        xhat, _, _, _ = model.reconstruct(xb)
        maes.append(torch.mean(torch.abs(xb - xhat), dim=1).cpu())
        recons.append(xhat.cpu())
    return torch.cat(maes).numpy(), torch.cat(recons).numpy()

def timed_compute_recon_scores(model, X_tensor, batch=512, device='cpu'):
    t0 = time.perf_counter()
    maes, recons = compute_recon_scores(model, X_tensor, batch=batch, device=device)
    t1 = time.perf_counter()
    return maes, recons, (t1 - t0)


def discover_datasets(data_dir):
    datasets = {}
    if not os.path.isdir(data_dir):
        logging.warning("Data directory '%s' does not exist.", data_dir)
        return datasets
    for p in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        name = os.path.splitext(os.path.basename(p))[0]
        datasets[name] = p
    return datasets



def save_run_metrics_csv(run_metrics: dict, dataset_name: str, csv_file: str = "results.csv"):
    """
    Append run metrics to a CSV file.
    Writes header only if the file does not exist.
    Stores raw numeric values (floats), not formatted strings with ±.
    """
    # Ensure all metrics are raw floats
    row = {"dataset": dataset_name}
    for k, v in run_metrics.items():
        row[k] = v
    # Convert row to dataframe for easier CSV writing
    df_row = pd.DataFrame([row])
    # Check if file exists
    file_exists = os.path.isfile(csv_file)

    df_row.to_csv(csv_file, mode='a', header=not file_exists, index=False)
    print(f"Saved metrics for dataset '{dataset_name}' to {csv_file}")




def save_topK_patterns_csv(summary_df: pd.DataFrame, dataset_name: str, output_dir: str = "topK_patterns_stats", run_id: int = 0):
    """
    Save top-K pattern statistics to CSV.
    Adds dataset name column and appends to file if exists.
    Header is added only if file does not exist.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        DataFrame with columns like ['Pattern', 'Mean ICR', 'Variance', 'Total Support', 'Mean z-score']
    dataset_name : str
        Name of the dataset
    csv_file : str
        CSV file path
    """
    # Add dataset name column
    df_to_save = summary_df.copy()
    df_to_save.insert(0, "dataset", dataset_name)
    df_to_save.insert(1, "run_id", run_id)

    csv_file = f"{output_dir}/{dataset_name}_topK_patterns.csv"
    # Check if file exists
    file_exists = os.path.isfile(csv_file)
    # Append to CSV
    df_to_save.to_csv(csv_file, mode='a', header=not file_exists, index=False)
    print(f"Saved top-K pattern stats for dataset '{dataset_name}' to {csv_file}")
