# Projection / dimensionality reduction comparison in 6D
# pip install numpy matplotlib scikit-learn
# optional:
# pip install umap-learn

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, MDS
from sklearn.preprocessing import StandardScaler


# ----------------------------
# Optional UMAP
# ----------------------------
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


# ----------------------------
# Random generator
# ----------------------------
def _rng(seed=42):
    return np.random.default_rng(seed)


# ----------------------------
# 6D dataset generators
# ----------------------------
def blobs_6d(n=1800, seed=42):
    X, y = make_blobs(
        n_samples=n,
        n_features=6,
        centers=4,
        cluster_std=[0.7, 1.0, 0.8, 1.2],
        random_state=seed
    )
    return X, y, "6D blobs"

def anisotropic_blobs_6d(n=1800, seed=42):
    X, y = make_blobs(
        n_samples=n,
        n_features=6,
        centers=4,
        cluster_std=1.0,
        random_state=seed
    )
    A = np.array([
        [1.0,  0.8,  0.0,  0.0,  0.0,  0.0],
        [0.0,  0.4,  0.7,  0.0,  0.0,  0.0],
        [0.0,  0.0,  1.2,  0.6,  0.0,  0.0],
        [0.0,  0.0,  0.0,  0.5,  0.8,  0.0],
        [0.0,  0.0,  0.0,  0.0,  1.1,  0.7],
        [0.3,  0.0,  0.0,  0.0,  0.0,  0.9],
    ])
    X = X @ A
    return X, y, "6D anisotropic blobs"

def concentric_hyperspheres_6d(n=1800, seed=42):
    rng = _rng(seed)
    n1 = n // 2
    n2 = n - n1

    X1 = rng.normal(size=(n1, 6))
    X1 /= np.linalg.norm(X1, axis=1, keepdims=True)
    X1 *= 1.0

    X2 = rng.normal(size=(n2, 6))
    X2 /= np.linalg.norm(X2, axis=1, keepdims=True)
    X2 *= 2.0

    X = np.vstack([X1, X2])
    X += rng.normal(scale=0.05, size=X.shape)
    y = np.array([0] * n1 + [1] * n2)
    return X, y, "6D concentric hyperspheres"

def spiral_6d(n=1800, seed=42):
    rng = _rng(seed)
    t = np.linspace(0, 6 * np.pi, n)

    x1 = np.cos(t)
    x2 = np.sin(t)
    x3 = t / (6 * np.pi)
    x4 = np.cos(2 * t)
    x5 = np.sin(2 * t)
    x6 = 0.5 * np.sin(0.5 * t)

    X = np.column_stack([x1, x2, x3, x4, x5, x6])
    X += rng.normal(scale=0.08, size=X.shape)

    # label by thirds along the curve, just for visualization
    y = np.digitize(t, bins=np.quantile(t, [1/3, 2/3]))
    return X, y, "6D spiral manifold"

def two_interleaving_manifolds_6d(n=1800, seed=42):
    rng = _rng(seed)
    n1 = n // 2
    n2 = n - n1

    t1 = rng.uniform(0, 4 * np.pi, size=n1)
    t2 = rng.uniform(0, 4 * np.pi, size=n2)

    X1 = np.column_stack([
        np.cos(t1),
        np.sin(t1),
        t1 / (4 * np.pi),
        np.cos(2 * t1),
        np.sin(2 * t1),
        0.3 * t1 / (4 * np.pi),
    ])

    X2 = np.column_stack([
        np.cos(t2 + np.pi / 2),
        np.sin(t2 + np.pi / 2),
        t2 / (4 * np.pi),
        np.cos(2 * t2 + np.pi / 2),
        np.sin(2 * t2 + np.pi / 2),
        0.3 * t2 / (4 * np.pi) + 0.3,
    ])

    X = np.vstack([X1, X2])
    X += rng.normal(scale=0.08, size=X.shape)
    y = np.array([0] * n1 + [1] * n2)
    return X, y, "6D interleaving manifolds"

def varying_density_blobs_6d(n=1800, seed=42):
    X, y = make_blobs(
        n_samples=n,
        n_features=6,
        centers=4,
        cluster_std=[0.25, 0.6, 1.3, 0.4],
        random_state=seed
    )
    return X, y, "6D varying-density blobs"


# ----------------------------
# Projection methods
# ----------------------------
def project_2d(X, method_name, seed=42):
    if method_name == "PCA":
        return PCA(n_components=2, random_state=seed).fit_transform(X)

    if method_name == "t-SNE":
        return TSNE(
            n_components=2,
            perplexity=30,
            init="pca",
            learning_rate="auto",
            random_state=seed
        ).fit_transform(X)

    if method_name == "Isomap":
        return Isomap(n_components=2, n_neighbors=15).fit_transform(X)

    if method_name == "MDS":
        return MDS(n_components=2, random_state=seed, n_init=1, max_iter=300).fit_transform(X)

    if method_name == "UMAP":
        if not HAS_UMAP:
            raise RuntimeError("umap-learn not installed")
        return umap.UMAP(
            n_components=2,
            n_neighbors=20,
            min_dist=0.1,
            random_state=seed
        ).fit_transform(X)

    raise ValueError(f"Unknown method: {method_name}")


def project_3d(X, method_name, seed=42):
    if method_name == "PCA":
        return PCA(n_components=3, random_state=seed).fit_transform(X)

    if method_name == "t-SNE":
        return TSNE(
            n_components=3,
            perplexity=30,
            init="pca",
            learning_rate="auto",
            random_state=seed
        ).fit_transform(X)

    if method_name == "Isomap":
        return Isomap(n_components=3, n_neighbors=15).fit_transform(X)

    if method_name == "MDS":
        return MDS(n_components=3, random_state=seed, n_init=1, max_iter=300).fit_transform(X)

    if method_name == "UMAP":
        if not HAS_UMAP:
            raise RuntimeError("umap-learn not installed")
        return umap.UMAP(
            n_components=3,
            n_neighbors=20,
            min_dist=0.1,
            random_state=seed
        ).fit_transform(X)

    raise ValueError(f"Unknown method: {method_name}")


# ----------------------------
# Plot helpers
# ----------------------------
def plot_2d_grid(X, y, dataset_name, methods, seed=42):
    fig, axes = plt.subplots(
        1, len(methods),
        figsize=(4.8 * len(methods), 4.2),
        constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    for ax, method in zip(axes, methods):
        try:
            Z = project_2d(X, method, seed=seed)
            ax.scatter(Z[:, 0], Z[:, 1], c=y, s=10, alpha=0.8, cmap="tab10")
            ax.set_title(f"{method} - 2D", fontsize=11)
        except Exception as e:
            ax.text(0.5, 0.5, f"{method}\nfailed:\n{type(e).__name__}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{method} - 2D", fontsize=11)

        ax.grid(True, linewidth=0.3, alpha=0.4)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"{dataset_name}: 6D → 2D projections", fontsize=14)
    plt.show()


def plot_3d_grid(X, y, dataset_name, methods, seed=42):
    fig = plt.figure(figsize=(5.2 * len(methods), 4.6), constrained_layout=True)

    for i, method in enumerate(methods, start=1):
        ax = fig.add_subplot(1, len(methods), i, projection="3d")
        try:
            Z = project_3d(X, method, seed=seed)
            ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=y, s=8, alpha=0.8, cmap="tab10")
            ax.set_title(f"{method} - 3D", fontsize=11)
        except Exception as e:
            ax.text2D(0.5, 0.5, f"{method}\nfailed:\n{type(e).__name__}",
                      ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{method} - 3D", fontsize=11)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    fig.suptitle(f"{dataset_name}: 6D → 3D projections", fontsize=14)
    plt.show()


# ----------------------------
# Main
# ----------------------------
seed = 42

datasets = [
    blobs_6d(seed=seed),
    anisotropic_blobs_6d(seed=seed),
    concentric_hyperspheres_6d(seed=seed),
    spiral_6d(seed=seed),
    two_interleaving_manifolds_6d(seed=seed),
    varying_density_blobs_6d(seed=seed),
]


'''
PROJECTION_COMPLEXITY_SHORT = {
    "PCA": "O(n·d^2)",
    "t-SNE": "O(i·n^2)",
    "Isomap": "O(n^3)",
    "MDS": "O(n^2–n^3)",
    "UMAP": "O(n log n)"
}

'''


methods = ["PCA", "t-SNE", "Isomap", "MDS"]
if HAS_UMAP:
    methods.append("UMAP")

for X, y, name in datasets:
    # standardize before projection so methods are compared fairly
    X_scaled = StandardScaler().fit_transform(X)

    plot_2d_grid(X_scaled, y, name, methods, seed=seed)
    plot_3d_grid(X_scaled, y, name, methods, seed=seed)