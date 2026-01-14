# Wine dataset (UCI id=109): anomaly detection with
# 1) Distance-based: average distance to k nearest neighbors (k-NN avg dist)
# 2) Isolation-based: Isolation Forest
# 3) Tree/ensemble-based: Random Trees Embedding + k-NN avg dist in leaf-embedding space
#
# Plots:
# - Figure 1: PCA 2D projection (red = outlier, green = normal)
# - Figure 2: UMAP 2D projection (red = outlier, green = normal)
#
# pip install ucimlrepo scikit-learn matplotlib numpy pandas umap-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomTreesEmbedding
import umap


# -------------------------
# Load + preprocess
# -------------------------
wine = fetch_ucirepo(id=109)
X_df = wine.data.features.copy()
y_df = wine.data.targets.copy()

# Target is only for inspection, not used in anomaly scoring
target = y_df.iloc[:, 0].astype(str)

scaler = StandardScaler()
X = scaler.fit_transform(X_df.values).astype(np.float32)

n, d = X.shape
print("X shape:", X.shape)


# -------------------------
# k-NN average distance score
# -------------------------
def knn_avg_distance_score(X, k=10, metric="euclidean"):
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric=metric)
    nbrs.fit(X)
    dist_all, _ = nbrs.kneighbors(X, return_distance=True)
    dist = dist_all[:, 1:]  # drop self distance
    return dist.mean(axis=1)


# -------------------------
# Isolation Forest score
# -------------------------
def isolation_forest_score(X, random_state=42):
    iso = IsolationForest(
        n_estimators=500,
        contamination="auto",
        max_samples="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    iso.fit(X)

    # score_samples: higher is more normal, so negate to make higher = more anomalous
    return -iso.score_samples(X)


# -------------------------
# Random Trees Embedding + k-NN avg distance in embedding space
# -------------------------
def rtrees_embedding_knn_score(X, k=10, random_state=42):
    rte = RandomTreesEmbedding(
        n_estimators=400,
        max_depth=6,
        random_state=random_state,
        n_jobs=-1,
    )
    rte.fit(X)
    X_emb = rte.transform(X)  # sparse matrix

    # cosine distance works well for sparse binary embeddings
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="auto")
    nbrs.fit(X_emb)
    dist_all, _ = nbrs.kneighbors(X_emb, return_distance=True)
    dist = dist_all[:, 1:]
    return dist.mean(axis=1)


# -------------------------
# Run anomaly detection
# -------------------------
k = 10
score_knn = knn_avg_distance_score(X, k=k)
score_iso = isolation_forest_score(X)
score_rte = rtrees_embedding_knn_score(X, k=k)

# Choose cutoffs (example: top 5% are outliers)
contamination = 0.05
thr_knn = np.quantile(score_knn, 1.0 - contamination)
thr_iso = np.quantile(score_iso, 1.0 - contamination)
thr_rte = np.quantile(score_rte, 1.0 - contamination)

is_out_knn = score_knn >= thr_knn
is_out_iso = score_iso >= thr_iso
is_out_rte = score_rte >= thr_rte

print(f"\nk={k}, contamination={contamination:.2f}")
print(f"kNN avg dist outliers:        {is_out_knn.sum()} / {n}")
print(f"Isolation Forest outliers:    {is_out_iso.sum()} / {n}")
print(f"RTE + kNN (cosine) outliers:  {is_out_rte.sum()} / {n}")


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
print(top_anomalies_table(X_df, target, score_knn, "knn_avg_dist", top_n=10).to_string(index=False))

print("\nTop anomalies by Isolation Forest:")
print(top_anomalies_table(X_df, target, score_iso, "iso_score", top_n=10).to_string(index=False))

print("\nTop anomalies by RTE + kNN distance:")
print(top_anomalies_table(X_df, target, score_rte, "rte_knn_dist", top_n=10).to_string(index=False))


# -------------------------
# Plot helper: red outliers, green normal
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

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
plot_binary_outliers(axes[0], Z_pca, is_out_knn, f"PCA: kNN avg distance outliers (k={k})")
plot_binary_outliers(axes[1], Z_pca, is_out_iso, "PCA: Isolation Forest outliers")
plot_binary_outliers(axes[2], Z_pca, is_out_rte, f"PCA: RTE + kNN outliers (k={k})")
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

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
plot_binary_outliers(axes[0], Z_umap, is_out_knn, f"UMAP: kNN avg distance outliers (k={k})")
plot_binary_outliers(axes[1], Z_umap, is_out_iso, "UMAP: Isolation Forest outliers")
plot_binary_outliers(axes[2], Z_umap, is_out_rte, f"UMAP: RTE + kNN outliers (k={k})")
plt.tight_layout()
plt.show()
