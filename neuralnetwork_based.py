# Wine dataset (UCI id=109): anomaly detection with
# 1) Distance-based: average distance to k nearest neighbors (k-NN avg dist)
# 2) Neural network based: Denoising Autoencoder (PyTorch) anomaly score = reconstruction error
#
# Plots:
# - PCA projection (red = outlier, green = normal)
# - UMAP projection (red = outlier, green = normal)
#
# pip install ucimlrepo scikit-learn matplotlib numpy pandas umap-learn torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import umap

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


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

X_t = torch.from_numpy(X)


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
# Neural method: Denoising Autoencoder anomaly score (reconstruction error)
# -------------------------
class DenoisingAE(nn.Module):
    def __init__(self, input_dim, latent_dim=6, p_drop=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p_drop),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p_drop),

            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def dae_reconstruction_error_score(
    X_t,
    latent_dim=6,
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-4,
    noise_std=0.05,
    max_epochs=2000,
    patience=60,
    seed=42,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = TensorDataset(X_t)

    n = len(ds)
    n_val = max(1, int(0.2 * n))
    n_train = n - n_val

    train_ds, val_ds = random_split(
        ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = DenoisingAE(input_dim=X_t.shape[1], latent_dim=latent_dim, p_drop=0.2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for (xb,) in train_loader:
            xb = xb.to(device)

            # denoising input
            if noise_std > 0:
                xb_in = xb + noise_std * torch.randn_like(xb)
            else:
                xb_in = xb

            opt.zero_grad()
            xb_hat = model(xb_in)
            loss = loss_fn(xb_hat, xb)  # reconstruct clean xb
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        count = 0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                xb_hat = model(xb)
                val_loss += loss_fn(xb_hat, xb).item() * xb.size(0)
                count += xb.size(0)
        val_loss /= count

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch % 100 == 0 or epoch == 1:
            print(f"epoch {epoch:4d} | val loss {val_loss:.6f}")

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch} (best val loss {best_val:.6f})")
            break

    model.load_state_dict(best_state)
    model.eval()

    # Reconstruction error for each sample
    X_np = X_t.numpy()
    errs = np.zeros(len(X_np), dtype=np.float64)

    with torch.no_grad():
        X_tensor = X_t.to(device)
        X_hat = model(X_tensor).cpu().numpy()

    errs = np.mean((X_np - X_hat) ** 2, axis=1)  # per-sample MSE
    return errs


# -------------------------
# Run anomaly detection
# -------------------------
k = 10
contamination = 0.05

score_knn = knn_avg_distance_score(X, k=k)

score_ae = dae_reconstruction_error_score(
    X_t,
    latent_dim=6,
    noise_std=0.05,
    weight_decay=1e-4,
    patience=60
)

thr_knn = np.quantile(score_knn, 1.0 - contamination)
thr_ae = np.quantile(score_ae, 1.0 - contamination)

is_out_knn = score_knn >= thr_knn
is_out_ae = score_ae >= thr_ae

print(f"\nk={k}, contamination={contamination:.2f}")
print(f"kNN avg dist outliers: {is_out_knn.sum()} / {n}")
print(f"DAE recon err outliers:{is_out_ae.sum()} / {n}")


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

print("\nTop anomalies by DAE reconstruction error:")
print(top_anomalies_table(X_df, target, score_ae, "dae_recon_err").to_string(index=False))


# -------------------------
# Plot helper (red outliers, green normal)
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
# Figure 1: PCA projection (for visualization only)
# -------------------------
Z_pca = PCA(n_components=2, random_state=42).fit_transform(X)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_binary_outliers(axes[0], Z_pca, is_out_knn, f"PCA proj: kNN avg dist (k={k})")
plot_binary_outliers(axes[1], Z_pca, is_out_ae, "PCA proj: Denoising AE recon error")
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
plot_binary_outliers(axes[0], Z_umap, is_out_knn, f"UMAP proj: kNN avg dist (k={k})")
plot_binary_outliers(axes[1], Z_umap, is_out_ae, "UMAP proj: Denoising AE recon error")
plt.tight_layout()
plt.show()
