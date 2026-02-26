# One figure per clustering method (subplots = datasets)
# Global standard scaling is applied per-dataset before fitting EVERY method.
# Includes: ARI/NMI/Silhouette/runtime + noise fraction + k_found
#
# HDBSCAN: uses a "filled labels" approach:
#   1) hard labels from HDBSCAN
#   2) fill noise using all_points_membership_vectors (soft membership)
#   3) fallback: any remaining noise gets assigned to nearest cluster centroid
#
# pip install numpy matplotlib scikit-learn pandas
# Optional: pip install hdbscan

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ----------------------------
# Data generation
# ----------------------------
def _rng(seed=42):
    return np.random.default_rng(seed)

def rotate(X, angle_deg):
    th = np.deg2rad(angle_deg)
    R = np.array([[np.cos(th), -np.sin(th)],
                  [np.sin(th),  np.cos(th)]])
    return X @ R.T

def scale(X, sx, sy):
    S = np.array([[sx, 0.0],
                  [0.0, sy]])
    return X @ S.T

def four_circular_blobs_different_sizes(
    n_total=1600,
    radii=(0.25, 0.45, 0.7, 1.0),
    centers=((-3, -3), (-3, 3), (3, -3), (3, 3)),
    seed=42
):
    rng = _rng(seed)
    n_each = n_total // 4
    Xs, ys = [], []
    for j, ((cx, cy), r) in enumerate(zip(centers, radii)):
        X = rng.normal(size=(n_each, 2)) * r + np.array([cx, cy])
        y = np.full(n_each, j, dtype=int)
        Xs.append(X); ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)

def ellipsoids(n_total=1600, seed=42):
    centers = np.array([[-4, 0], [0, 0], [4, 0]])
    X, y = make_blobs(
        n_samples=n_total,
        centers=centers,
        cluster_std=[0.8, 0.9, 0.7],
        random_state=seed
    )
    X_out = np.empty_like(X)
    for lab in np.unique(y):
        Xi = X[y == lab]
        if lab == 0:
            Xi = rotate(scale(Xi, 2.2, 0.5), 25)
        elif lab == 1:
            Xi = rotate(scale(Xi, 0.7, 2.0), -35)
        else:
            Xi = rotate(scale(Xi, 1.8, 0.6), 70)
        X_out[y == lab] = Xi
    return X_out, y

def spiral_arms(n=1500, arms=3, noise=0.08, turns=2.5, seed=42):
    rng = _rng(seed)
    n_per = n // arms
    Xs, ys = [], []
    for k in range(arms):
        t = rng.uniform(0.0, turns * 2*np.pi, size=n_per)
        r = t
        phi = (2*np.pi * k) / arms
        x = r * np.cos(t + phi)
        y = r * np.sin(t + phi)
        X = np.column_stack([x, y])
        X += rng.normal(scale=noise, size=X.shape)
        Xs.append(X)
        ys.append(np.full(n_per, k, dtype=int))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    return X, y

def two_moons(n_total=1600, noise=0.08, seed=42):
    X, y = make_moons(n_samples=n_total, noise=noise, random_state=seed)
    return X, y

def concentric_circles(n_total=1600, noise=0.05, factor=0.45, seed=42):
    X, y = make_circles(n_samples=n_total, noise=noise, factor=factor, random_state=seed)
    return X, y

def varying_density_blobs(n_total=1600, seed=42):
    centers = np.array([[-5, -2], [-2, 3], [3, 2], [6, -3]])
    stds = [0.25, 0.6, 1.2, 0.35]
    X, y = make_blobs(n_samples=n_total, centers=centers, cluster_std=stds, random_state=seed)
    return X, y

def anisotropic_blobs(n_total=1600, seed=42):
    X, y = make_blobs(n_samples=n_total, centers=4, cluster_std=1.0, random_state=seed)
    A = np.array([[0.6, -0.6],
                  [0.4,  1.0]])
    X = X @ A
    return X, y

def noisy_line_plus_blob(n_total=1600, seed=42):
    rng = _rng(seed)
    n1 = int(n_total * 0.65)
    n2 = n_total - n1
    blob = rng.normal(size=(n1, 2)) * np.array([0.8, 0.8]) + np.array([2.5, 2.0])
    t = rng.uniform(-6, 6, size=n2)
    line = np.column_stack([t, 0.35 * t + rng.normal(scale=0.25, size=n2)]) + np.array([-1.0, -1.0])
    X = np.vstack([blob, line])
    y = np.concatenate([np.zeros(n1, dtype=int), np.ones(n2, dtype=int)])
    return X, y

def high_overlap_gaussians(n_total=1600, seed=42):
    X, y = make_blobs(
        n_samples=n_total,
        centers=[(-1, 0), (1, 0), (0, 1.5)],
        cluster_std=[1.2, 1.2, 1.1],
        random_state=seed
    )
    return X, y


# ----------------------------
# Complexity strings
# ----------------------------
COMPLEXITY = {
    "KMeans": "O(n·k·i·d)",
    "GMM": "O(n·k·i·d²)",
    "DBSCAN": "O(n log n) to O(n²)",
    "HDBSCAN": "O(n log n) to O(n²)",
    "Agglomerative": "O(n²) mem O(n²)",
    "Spectral": "O(n²) + O(n³)",
    "MeanShift": "O(n²·i)",
}


# ----------------------------
# Utilities
# ----------------------------
def choose_dbscan_eps(X, k=10, factor=1.2):
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    dists, _ = nn.kneighbors(X)
    kth = dists[:, -1]
    return float(np.median(kth) * factor)

def safe_silhouette(X, labels):
    labels = np.asarray(labels)
    mask = labels != -1
    if np.sum(mask) < 3:
        return np.nan
    labs = labels[mask]
    if len(set(labs)) < 2:
        return np.nan
    return float(silhouette_score(X[mask], labs))

def plot_labels(ax, X, labels, title):
    labels = np.asarray(labels)
    noise = labels == -1
    if np.any(~noise):
        ax.scatter(X[~noise, 0], X[~noise, 1], c=labels[~noise], s=8, alpha=0.85, cmap="tab20")
    if np.any(noise):
        ax.scatter(X[noise, 0], X[noise, 1], s=8, alpha=0.65, color="lightgray")
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal", "box")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.set_xticks([])
    ax.set_yticks([])


# ----------------------------
# Datasets
# ----------------------------
seed = 42
datasets = [
    ("4 circular blobs", *four_circular_blobs_different_sizes(seed=seed)),
    ("Ellipsoids", *ellipsoids(seed=seed)),
    ("Spiral arms", *spiral_arms(arms=3, seed=seed)),
    ("Two moons", *two_moons(seed=seed)),
    ("Concentric circles", *concentric_circles(seed=seed)),
    ("Varying density", *varying_density_blobs(seed=seed)),
    ("Anisotropic blobs", *anisotropic_blobs(seed=seed)),
    ("Blob + line", *noisy_line_plus_blob(seed=seed)),
    ("High-overlap Gaussians", *high_overlap_gaussians(seed=seed)),
]

# Optional HDBSCAN
try:
    import hdbscan
    HAS_HDBSCAN = True
except Exception:
    HAS_HDBSCAN = False

algorithms = ["GroundTruth", "KMeans", "GMM", "DBSCAN", "HDBSCAN", "Agglomerative", "Spectral", "MeanShift"]


# ----------------------------
# HDBSCAN helper: fill noise using membership vectors + fallback nearest-centroid
# ----------------------------
'''
cluster_selection_epsilon
0.05 → very mild merging
0.10 → mild merging
0.20 → noticeable merging
0.30 → aggressive merging
'''

def hdbscan_fit_predict_filled(
    X_scaled,
    min_cluster_size=200,
    min_samples=5,
    cluster_selection_method="leaf",  ### eom = more clusters, leaf = fewer clusters
    prob_threshold=0.05,   # 0.0 = as aggressive as possible,
    cluster_selection_epsilon=0.1,
    fallback_nearest_centroid=True,
    debug=False
):
    """
    Returns labels where many noise points are reassigned:
      1) hard labels from HDBSCAN
      2) reassign noise points to best soft-membership cluster (if best_prob >= prob_threshold)
      3) fallback: any remaining noise points assigned to nearest centroid of labeled clusters

    Note: This turns HDBSCAN into a "mostly-partitioning" approach for demos when prob_threshold=0.
    """
    if not HAS_HDBSCAN:
        raise RuntimeError("hdbscan not installed")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        prediction_data=True
    ).fit(X_scaled)

    hard = clusterer.labels_.copy()

    cluster_labels = np.array(sorted([lab for lab in np.unique(hard) if lab != -1]), dtype=int)
    if debug:
        u, c = np.unique(hard, return_counts=True)
        print("hard label counts:", dict(zip(u, c)))
        print("n_clusters_found:", cluster_labels.size)

    # If HDBSCAN found no clusters, nothing to fill
    if cluster_labels.size == 0:
        return hard

    membership = hdbscan.all_points_membership_vectors(clusterer)
    membership = np.nan_to_num(membership, nan=0.0, posinf=0.0, neginf=0.0)

    if membership.ndim != 2 or membership.shape[0] != X_scaled.shape[0]:
        if debug:
            print("membership shape mismatch:", getattr(membership, "shape", None))
        return hard

    best_idx = membership.argmax(axis=1)
    best_prob = membership.max(axis=1)

    filled = hard.copy()

    # Fill noise based on membership (prob_threshold=0.0 assigns all noise points
    # that have ANY non-zero membership)
    mask = (filled == -1) & (best_prob >= prob_threshold)
    filled[mask] = cluster_labels[best_idx[mask]]

    # Fallback: assign remaining noise to nearest centroid of labeled clusters
    if fallback_nearest_centroid:
        remaining = filled == -1
        if np.any(remaining):
            centroids = []
            for lab in cluster_labels:
                pts = X_scaled[filled == lab]
                if pts.shape[0] > 0:
                    centroids.append(pts.mean(axis=0))
                else:
                    centroids.append(np.zeros(X_scaled.shape[1]))
            centroids = np.vstack(centroids)

            diffs = X_scaled[remaining, None, :] - centroids[None, :, :]
            d2 = np.sum(diffs * diffs, axis=2)
            nearest = np.argmin(d2, axis=1)
            filled[remaining] = cluster_labels[nearest]

    if debug:
        print("noise before:", float(np.mean(hard == -1)))
        print("noise after :", float(np.mean(filled == -1)))
        print("rows with all-zero membership:", int(np.sum(best_prob == 0.0)))

    return filled


### Similar approach for DBSCAN (not as good since no soft membership, but better than leaving all noise as -1)
def dbscan_fit_predict_filled(
    X_scaled,
    eps=None,
    min_samples=10,
    k_for_eps=10,
    eps_factor=1.2,
    fallback_nearest_centroid=True,
    debug=False
):
    if eps is None:
        eps = choose_dbscan_eps(X_scaled, k=k_for_eps, factor=eps_factor)

    model = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
    hard = model.labels_.copy()

    cluster_labels = np.array(sorted([lab for lab in np.unique(hard) if lab != -1]), dtype=int)

    if not fallback_nearest_centroid:
        return hard

    if cluster_labels.size == 0:
        return hard

    filled = hard.copy()
    remaining = filled == -1
    if np.any(remaining):
        centroids = []
        for lab in cluster_labels:
            pts = X_scaled[filled == lab]
            centroids.append(pts.mean(axis=0))
        centroids = np.vstack(centroids)

        diffs = X_scaled[remaining, None, :] - centroids[None, :, :]
        d2 = np.sum(diffs * diffs, axis=2)
        nearest = np.argmin(d2, axis=1)
        filled[remaining] = cluster_labels[nearest]

    return filled

# ----------------------------
# Fit-predict wrapper
# ----------------------------
def fit_predict(alg, X_scaled, k_true, seed=42):
    if alg == "GroundTruth":
        raise ValueError("GroundTruth is not a model")

    if alg == "KMeans":
        return KMeans(n_clusters=k_true, n_init=10, random_state=seed).fit_predict(X_scaled)

    if alg == "GMM":
        m = GaussianMixture(n_components=k_true, covariance_type="full", random_state=seed)
        m.fit(X_scaled)
        return m.predict(X_scaled)

    if alg == "DBSCAN":
        eps = choose_dbscan_eps(X_scaled, k=10, factor=1.2)
        return dbscan_fit_predict_filled(
                                            X_scaled,
                                            eps=None,
                                            min_samples=10,
                                            k_for_eps=10,
                                            eps_factor=1.2,
                                            fallback_nearest_centroid=True,
                                            debug=False
                                        )

    if alg == "HDBSCAN":
        return hdbscan_fit_predict_filled(
            X_scaled,
            min_cluster_size=5,
            min_samples=1,
            cluster_selection_method="leaf",
            prob_threshold=0.0,              # set to 0.02/0.05 if you want some noise retained
            fallback_nearest_centroid=True,  # ensures almost no -1
            debug=False
        )

    if alg == "Agglomerative":
        return AgglomerativeClustering(n_clusters=k_true, linkage="ward").fit_predict(X_scaled)

    if alg == "Spectral":
        return SpectralClustering(
            n_clusters=k_true,
            affinity="rbf",  ## nearest_neighbors
            n_neighbors=20, ## Too small: each arm becomes disconnected in sections, Spectral splits weirdly. Too large: the graph connects across arms (shortcuts), Spectral merges arms.
            assign_labels="kmeans",
            random_state=seed
        ).fit_predict(X_scaled)

    if alg == "MeanShift":
        bw = estimate_bandwidth(X_scaled, quantile=0.2, n_samples=min(500, X_scaled.shape[0]), random_state=seed)
        if not np.isfinite(bw) or bw <= 0:
            bw = None
        return MeanShift(bandwidth=bw, bin_seeding=True).fit_predict(X_scaled)

    raise ValueError(f"Unknown algorithm: {alg}")


# ----------------------------
# Benchmark + plots (global scaling per dataset)
# ----------------------------
results = []
n_ds = len(datasets)
cols = 3
rows = int(np.ceil(n_ds / cols))

for alg in algorithms:
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.6, rows * 4.1), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for idx, (ds_name, X, y_true) in enumerate(datasets):
        ax = axes[idx]
        X = np.asarray(X)
        y_true = np.asarray(y_true)
        k_true = len(np.unique(y_true))

        # Global scaling per dataset (applied to all methods)
        X_scaled = StandardScaler().fit_transform(X)

        if alg == "GroundTruth":
            plot_labels(ax, X_scaled, y_true, f"{ds_name}\nGround truth (scaled)")
            continue

        t0 = time.perf_counter()
        status = "ok"
        try:
            y_pred = fit_predict(alg, X_scaled, k_true, seed=seed)
            if alg == "KMeans":
                y_pred = np.asarray(y_pred)
                print(
                    f"[debug] {ds_name} KMeans: shape={y_pred.shape}, dtype={y_pred.dtype}, "
                    f"min={y_pred.min()}, max={y_pred.max()}, unique_count={len(np.unique(y_pred))}"
                )
                print("[debug] first 20 labels:", y_pred[:20])
        except Exception as e:
            status = f"fail: {type(e).__name__}"
            print(f"[{alg}] failed on '{ds_name}': {repr(e)}")
            y_pred = np.full(X_scaled.shape[0], -1, dtype=int)
        runtime_ms = (time.perf_counter() - t0) * 1000.0

        ari = adjusted_rand_score(y_true, y_pred) if status == "ok" else np.nan
        nmi = normalized_mutual_info_score(y_true, y_pred) if status == "ok" else np.nan
        sil = safe_silhouette(X_scaled, y_pred) if status == "ok" else np.nan

        comp = COMPLEXITY.get(alg, "")
        k_found = (len(set(y_pred)) - (1 if -1 in set(y_pred) else 0))
        noise_frac = float(np.mean(y_pred == -1))

        title = (
            f"{ds_name}\n"
            f"{alg} ({comp})\n"
            f"ARI={ari:.3f}  NMI={nmi:.3f}  Sil={sil:.3f}\n"
            f"{runtime_ms:.1f} ms  k={k_found}/{k_true}  noise={noise_frac:.2f}  [{status}]"
        )
        plot_labels(ax, X_scaled, y_pred, title)

        results.append({
            "dataset": ds_name,
            "algorithm": alg,
            "complexity": comp,
            "runtime_ms": runtime_ms,
            "ARI": ari,
            "NMI": nmi,
            "Silhouette": sil,
            "status": status,
            "k_true": k_true,
            "k_found_excl_noise": k_found,
            "noise_frac": noise_frac,
        })

    for j in range(n_ds, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{alg} results across datasets (scaled)", fontsize=14)
    plt.show()


# ----------------------------
# Results table + quick summaries
# ----------------------------
df = pd.DataFrame(results)
df_sorted = df.sort_values(["dataset", "ARI", "runtime_ms"], ascending=[True, False, True])

print("\n=== Top 3 per dataset by ARI ===\n")
print(df_sorted.groupby("dataset", as_index=False).head(3)[
    ["dataset", "algorithm", "ARI", "NMI", "Silhouette", "runtime_ms", "k_true", "k_found_excl_noise", "noise_frac", "status"]
].to_string(index=False))

print("\n=== Worst per dataset by ARI ===\n")
worst = df.sort_values(["dataset", "ARI"], ascending=[True, True]).groupby("dataset", as_index=False).head(1)
print(worst[["dataset", "algorithm", "ARI", "NMI", "Silhouette", "runtime_ms", "status"]].to_string(index=False))

df_sorted.to_csv("clustering_benchmark_results.csv", index=False)
print("\nSaved: clustering_benchmark_results.csv")