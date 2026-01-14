# Wine dataset (UCI id=109): anomaly detection with
# 1) Distance-based: average distance to k nearest neighbors (k-NN avg dist)
# 2) One-class classification: One-Class SVM
#
# Plots:
# - PCA projection (red = outlier, green = normal)
# - UMAP projection (red = outlier, green = normal)
#
# pip install ucimlrepo scikit-learn matplotlib numpy pandas umap-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
import umap


# -------------------------
# Load + preprocess
# -------------------------
wine = fetch_ucirepo(id=109)
X_df = wine.data.features.copy()
y_df = wine.data.targets.copy()

# Target used only for inspection
target = y_df.iloc[:, 0].astype(str)

scaler = StandardScaler()
X = scaler.fit_transform(X_df.values).astype(np.float32)

n, d = X.shape
print("X shape:", X.shape)


# -------------------------
# k-NN average distance score
# -------------------------
def knn_avg_distance_score(X, k=10, metric="euclidean"):
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nbrs.fit(X)
    dist_all, _ = nbrs.kneighbors(X, return_distance=True)
    dist = dist_all[:, 1:]  # drop self distance
    return dist.mean(axis=1)


# -------------------------
# One-Class SVM score
# -------------------------
def one_class_svm_score(X, nu=0.05, gamma="scale"):
    """
    nu ≈ expected fraction of outliers
    score_samples: higher = more normal → negate for anomaly score
    """
    ocsvm = OneClassSVM(
        kernel="rbf",
        nu=nu,
        gamma=gamma
    )
    ocsvm.fit(X)
    return -ocsvm.score_samples(X)


# -------------------------
# Run anomaly detection
# -------------------------
k = 10
contamination = 0.05

score_knn = knn_avg_distance_score(X, k=k)
score_ocsvm = one_class_svm_score(X, nu=contamination)

thr_knn = np.quantile(score_knn, 1.0 - contamination)
thr_ocsvm = np.quantile(score_ocsvm, 1.0 - contamination)

is_out_knn = score_knn >= thr_knn
is_out_ocsvm = score_ocsvm >= thr_ocsvm

print(f"\nk={k}, contamination={contamination:.2f}")
print(f"kNN avg dist outliers: {is_out_knn.sum()} / {n}")
print(f"OC-SVM outliers:       {is_out_ocsvm.sum()} / {n}")


# -------------------------
# Top anomalies table helper
# -------------------------
def top_anomalies_table(X_df, target, scores, name, top_n=10):
    out = X_df.copy()
    out["target"] = target.values
    out[name] = scores
    out = out.sort_values(name, ascending=False).head(top_n)
    return out[["target", name] + list(X_df.columns[:5])]

print("\nTop anomalies by kNN avg distance:")
print(top_anomalies_table(X_df, target, score_knn, "knn_avg_dist").to_string(index=False))

print("\nTop anomalies by One-Class SVM:")
print(top_anomalies_table(X_df, target, score_ocsvm, "ocsvm_score").to_string(index=False))


# -------------------------
# Plot helper
# -------------------------
def plot_binary_outliers(ax, Z2, is_outlier, title):
    ax.scatter(
        Z2[~is_outlier, 0],
        Z2[~is_outlier, 1],
        c="green",
        s=22,
        alpha=0.7,
        label="Normal",
    )
    ax.scatter(
        Z2[is_outlier, 0],
        Z2[is_outlier, 1],
        c="red",
        s=34,
        alpha=0.9,
        label="Outlier",
    )
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend()


# -------------------------
# Figure 1: PCA projection
# -------------------------
Z_pca = PCA(n_components=2, random_state=42).fit_transform(X)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_binary_outliers(axes[0], Z_pca, is_out_knn, f"PCA: kNN avg dist (k={k})")
plot_binary_outliers(axes[1], Z_pca, is_out_ocsvm, "PCA: One-Class SVM")
plt.tight_layout()
plt.show()


# -------------------------
# Figure 2: UMAP projection
# -------------------------
Z_umap = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
).fit_transform(X)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_binary_outliers(axes[0], Z_umap, is_out_knn, f"UMAP: kNN avg dist (k={k})")
plot_binary_outliers(axes[1], Z_umap, is_out_ocsvm, "UMAP: One-Class SVM")
plt.tight_layout()
plt.show()
