"""Minimal end-to-end example: train DiffInt on inliers, score a test set."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from sklearn.metrics import roc_auc_score
from diffint import DiffInt

rng = np.random.RandomState(0)
# toy data: inliers ~ N(0, I) in 8D; anomalies shifted on a few coordinates
d = 8
X_train = rng.randn(400, d).astype("float32")                  # inliers only
X_in = rng.randn(100, d).astype("float32")
X_out = rng.randn(20, d).astype("float32"); X_out[:, :3] += 6.0  # anomalies
X_test = np.vstack([X_in, X_out]); y_test = np.r_[np.zeros(100), np.ones(20)]

det = DiffInt(K=64, fusion_mode="PA", epochs=300, lr=5e-3, random_state=0).fit(X_train)
scores = det.decision_function(X_test)
print("AUROC:", round(roc_auc_score(y_test, scores), 4))

# inspect a few learned interval rules (original feature units)
for k, feat, lo, hi in det.interval_rules()[:5]:
    print(f"  unit {k}: {feat} in [{lo:.2f}, {hi:.2f}]")
