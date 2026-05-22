"""Reproduce the ADBench evaluation: semi-supervised, [-1,1] normalization, AUROC/AUPRC.

Each CSV is expected to have feature columns followed by a final label column
(0 = inlier, !=0 = anomaly), matching the ADBench tabular format.

Usage:
    python scripts/run_adbench.py --data_dir csv/ --K 200 --fusion_mode PA \
        --epochs 1000 --lr 5e-5 --nb_runs 3 [--scaler minmax11]
"""
import os, glob, argparse, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from diffint import DiffInt


def run_one(path, args):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values.astype("float32")
    y = df.iloc[:, -1].values.astype("float32")
    y = (y != 0).astype(int)
    aucs, auprs = [], []
    for run in range(args.nb_runs):
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=args.test_size, random_state=run)
        det = DiffInt(
            K=args.K, fusion_mode=args.fusion_mode, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, scaler=args.scaler, random_state=run,
        ).fit(Xtr, ytr)                       # inliers only (y==0)
        s = det.decision_function(Xte)
        aucs.append(roc_auc_score(yte, s))
        auprs.append(average_precision_score(yte, s))
    return np.mean(aucs), np.std(aucs), np.mean(auprs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="csv/")
    p.add_argument("--data", default=None, help="single CSV filename inside data_dir")
    p.add_argument("--K", type=int, default=200)
    p.add_argument("--fusion_mode", default="PA", choices=["PA", "IA", "DA"])
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--test_size", type=float, default=0.4)
    p.add_argument("--nb_runs", type=int, default=3)
    p.add_argument("--scaler", default="minmax11",
                   choices=["minmax11", "minmax01", "standard", "none"])
    args = p.parse_args()

    files = ([os.path.join(args.data_dir, args.data)] if args.data
             else sorted(glob.glob(os.path.join(args.data_dir, "*.csv"))))
    if not files:
        raise SystemExit(f"No CSVs found in {args.data_dir}")
    print(f"{'dataset':22s} {'AUROC':>14} {'AUPRC':>8}")
    print("-" * 46)
    for f in files:
        try:
            am, asd, pm = run_one(f, args)
            name = os.path.splitext(os.path.basename(f))[0]
            print(f"{name:22s} {am:6.4f}±{asd:5.3f} {pm:8.4f}", flush=True)
        except Exception as e:
            print(f"{os.path.basename(f)}: FAILED ({e})", flush=True)


if __name__ == "__main__":
    main()
