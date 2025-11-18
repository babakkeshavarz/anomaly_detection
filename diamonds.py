import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture

from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.mad import MAD

pd.set_option('display.max_columns', None)

# ------------------------------------------------------
# Load Data
# ------------------------------------------------------
diamonds = sns.load_dataset("diamonds")

# Initial EDA
sns.pairplot(diamonds.sample(500), hue="price")
plt.tight_layout()
plt.show()

diamonds.plot(kind='box', figsize=(15,10), subplots=True, layout=(3,3))
plt.show()

# Make a copy for outlier labeling
df = diamonds.copy()

# ------------------------------------------------------
# Encode categoricals
# ------------------------------------------------------
cats = df.select_dtypes(include="category").columns.tolist()
oe = OrdinalEncoder()
df[cats] = oe.fit_transform(df[cats])

# ------------------------------------------------------
# Prepare features
# ------------------------------------------------------
X = df.drop("price", axis=1)
y = df["price"]

# ------------------------------------------------------
# 1. MAD Outliers (univariate on price)
# ------------------------------------------------------
mad = MAD().fit(y.values.reshape(-1,1))
df["outlier_mad"] = mad.labels_

# ------------------------------------------------------
# 2. IQR rule (univariate)
# ------------------------------------------------------
Q1 = y.quantile(0.25)
Q3 = y.quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df["outlier_iqr"] = ((y < lower) | (y > upper)).astype(int)

# ------------------------------------------------------
# 3. Z-score rule (univariate)
# ------------------------------------------------------
z = (y - y.mean()) / y.std()
df["outlier_zscore"] = (z.abs() > 3).astype(int)

# ------------------------------------------------------
# 4. Isolation Forest
# ------------------------------------------------------
iforest = IForest(random_state=42, n_estimators=500, max_samples=1000)
iforest.fit(X)
df["outlier_iforest"] = iforest.labels_

# ------------------------------------------------------
# 5. Local Outlier Factor (LOF)
# ------------------------------------------------------
lof = LocalOutlierFactor(n_neighbors=20, contamination="auto")
df["outlier_lof"] = (lof.fit_predict(X) == -1).astype(int)

# ------------------------------------------------------
# 6. KNN Outliers (PyOD)
# ------------------------------------------------------
knn = KNN(method="largest", n_neighbors=20)
knn.fit(X)
df["outlier_knn"] = knn.labels_

# ------------------------------------------------------
# 7. One-Class SVM
# ------------------------------------------------------
ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
df["outlier_ocsvm"] = (ocsvm.fit_predict(X) == -1).astype(int)

# ------------------------------------------------------
# 8. AutoEncoder (PyOD)
# ------------------------------------------------------
autoenc = AutoEncoder(
    contamination=0.05,          # expected outlier fraction
    preprocessing=True,          # scale data automatically
    lr=0.001,                    # learning rate
    epoch_num=20,                # TRAINING EPOCHS (correct param)
    batch_size=32,
    optimizer_name='adam',
    random_state=42,
    hidden_neuron_list=[64, 32],  # encoder:64 → 32 → decoder:32 → 64
    hidden_activation_name='relu',
    batch_norm=True,
    dropout_rate=0.2,
    verbose=0
)

autoenc.fit(X)
df["outlier_autoenc"] = autoenc.labels_

# ------------------------------------------------------
# 9. Gaussian Mixture (probability-based)
# ------------------------------------------------------
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)
probs = gmm.score_samples(X)
threshold = np.percentile(probs, 3)   # bottom 3% considered outliers
df["outlier_gmm"] = (probs < threshold).astype(int)

# ------------------------------------------------------
# Show summary
# ------------------------------------------------------
print(df.filter(like="outlier").sum())
print(df.describe())

# ------------------------------------------------------
# Visualize one model (e.g., Isolation Forest)
# ------------------------------------------------------
sns.pairplot(df.sample(500), hue="outlier_iforest")
plt.tight_layout()
plt.show()


