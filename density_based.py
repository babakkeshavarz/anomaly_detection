# Wine dataset (UCI id=109): anomaly detection with
# 1) k-NN average distance score
# 2) k-NN average squared distance (RMS) score (optional baseline)
# 3) LoOP (Local Outlier Probability)
#
# pip install ucimlrepo scikit-learn matplotlib numpy pandas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


# -------------------------
# Load + preprocess
# -------------------------
wine = fetch_ucirepo(id=109)
X_df = wine.data.features.copy()
y_df = wine.data.targets.copy()

# target only used for coloring/inspection, NOT used in scoring
target = y_df.iloc[:, 0].astype(str)

scaler = StandardScaler()
X = scaler.fit_transform(X_df.values).astype(np.float32)

n, d = X.shape
print("X shape:", X.shape)


# -------------------------
# k-NN helpers
# -------------------------
def knn_distances(X, k=10):
    """
    Returns:
      distances: (n, k) distances to the k nearest neighbors (excluding self)
      indices:   (n, k) neighbor indices (excluding self)
    """
    # +1 to include self, then drop it
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean")
    nbrs.fit(X)
    dist_all, idx_all = nbrs.kneighbors(X, return_distance=True)

    distances = dist_all[:, 1:]  # drop self distance = 0
    indices = idx_all[:, 1:]
    return distances, indices


def knn_avg_distance_score(X, k=10):
    """
    Average distance to k nearest neighbors.
    Higher score => more anomalous.
    """
    dist, _ = knn_distances(X, k=k)
    return dist.mean(axis=1)


def knn_rms_distance_score(X, k=10):
    """
    RMS distance to k nearest neighbors (sqrt(mean(d^2))).
    Higher score => more anomalous.
    """
    dist, _ = knn_distances(X, k=k)
    return np.sqrt((dist ** 2).mean(axis=1))


# -------------------------
# LoOP (Local Outlier Probability)
# -------------------------
def loop_scores(X, k=10, lam=3.0, eps=1e-12):
    """
    LoOP implementation (probabilistic local outlier detection).

    Inputs:
      k: number of neighbors
      lam: "lambda" scaling parameter (often 3.0)
    Output:
      loop: array shape (n,) in [0, 1], higher => more likely outlier
    """
    dist, nbr_idx = knn_distances(X, k=k)

    # probabilistic set distance (pdist) per point i
    # pdist_i = lam * sqrt(mean_j d(i, nn_j)^2)
    pdist = lam * np.sqrt((dist ** 2).mean(axis=1) + eps)  # (n,)

    # Expected pdist of neighbors for each i
    # E_pdist_i = mean_{j in N(i)} pdist_j
    E_pdist = pdist[nbr_idx].mean(axis=1) + eps

    # PLOF_i = pdist_i / E_pdist_i - 1
    plof = (pdist / E_pdist) - 1.0

    # nPLOF: normalization term (global)
    # nPLOF = lam * sqrt(mean(PLOF^2))
    nplof = lam * np.sqrt(np.mean(plof ** 2) + eps) + eps

    # LoOP_i = max(0, erf( PLOF_i / (nPLOF * sqrt(2)) ))
    # Use scipy if available; otherwise approximate erf.
    try:
        from math import erf, sqrt
        loop = np.array([max(0.0, erf(p / (nplof * np.sqrt(2.0)))) for p in plof], dtype=np.float64)
    except Exception:
        # fallback erf approximation (Abramowitz-Stegun style)
        def erf_approx(x):
            # max error ~1.5e-7
            sign = np.sign(x)
            x = np.abs(x)
            t = 1.0 / (1.0 + 0.3275911 * x)
            a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
            y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * np.exp(-x * x)
            return sign * y

        x = plof / (nplof * np.sqrt(2.0))
        loop = np.maximum(0.0, erf_approx(x))

    # clamp for numerical safety
    return np.clip(loop, 0.0, 1.0)


# -------------------------
# Run anomaly detection
# -------------------------
k = 10  # try 5, 10, 20
score_avg = knn_avg_distance_score(X, k=k)
score_loop = loop_scores(X, k=k, lam=3.0)

# Choose a cutoff for "anomalies" (example: top 5%)
contamination = 0.05
thr_avg = np.quantile(score_avg, 1.0 - contamination)
thr_loop = np.quantile(score_loop, 1.0 - contamination)

is_anom_avg = score_avg >= thr_avg
is_anom_loop = score_loop >= thr_loop

print(f"k={k}")
print(f"AvgDist anomalies: {is_anom_avg.sum()} / {n}")
print(f"LoOP anomalies:    {is_anom_loop.sum()} / {n}")


# -------------------------
# Report top anomalies
# -------------------------
def top_anomalies_table(X_df, target, scores, name, top_n=10):
    out = X_df.copy()
    out["target"] = target.values
    out[name] = scores
    out = out.sort_values(name, ascending=False).head(top_n)
    return out[["target", name] + [c for c in X_df.columns[:5]]]  # show a few features

print("\nTop anomalies by AvgDist:")
print(top_anomalies_table(X_df, target, score_avg, "avg_knn_dist", top_n=10).to_string(index=False))

print("\nTop anomalies by LoOP:")
print(top_anomalies_table(X_df, target, score_loop, "loop_prob", top_n=10).to_string(index=False))


# -------------------------
# Visualization (PCA 2D, red = outlier, green = normal)
# -------------------------
Z = PCA(n_components=2, random_state=42).fit_transform(X)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- k-NN average distance ---
axes[0].scatter(
    Z[~is_anom_avg, 0],
    Z[~is_anom_avg, 1],
    s=22,
    alpha=0.8,
    label="Normal",
)
axes[0].scatter(
    Z[is_anom_avg, 0],
    Z[is_anom_avg, 1],
    s=30,
    alpha=0.9,
    label="Outlier",
)
axes[0].set_title(f"k-NN avg distance outliers (k={k})")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
axes[0].legend()


# --- LoOP ---
axes[1].scatter(
    Z[~is_anom_loop, 0],
    Z[~is_anom_loop, 1],
    s=22,
    alpha=0.8,
    label="Normal",
)
axes[1].scatter(
    Z[is_anom_loop, 0],
    Z[is_anom_loop, 1],
    s=30,
    alpha=0.9,
    label="Outlier",
)
axes[1].set_title(f"LoOP outliers (k={k})")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].legend()

plt.tight_layout()
plt.show()