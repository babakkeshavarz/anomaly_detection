import pandas as pd
import seaborn as sns
from pyod.models.mad import MAD

diamonds = sns.load_dataset("diamonds")
# Extract the feature we want
X = diamonds[["price"]]


mad = MAD().fit(X)

labels = pd.Series(mad.labels_, name="outlier_label")
diamonds["outlier_label"] = labels
print(diamonds[diamonds['outlier_label']==1])


diamonds.info()


from sklearn.preprocessing import OrdinalEncoder
cats = diamonds.select_dtypes(include="category").columns.tolist()

# Initialize encoder
oe = OrdinalEncoder()

# Encode
cats_encoded = oe.fit_transform(diamonds[cats])



diamonds = diamonds.copy()  # avoid SettingWithCopy issues
diamonds[cats] = diamonds[cats].astype(object)
diamonds.loc[:, cats] = cats_encoded


X = diamonds.drop("price", axis=1)
y = diamonds[["price"]]
from pyod.models.iforest import IForest

iforest = IForest(n_estimators=1000)
iforest.fit(X)  

# Extract the labels
labels = iforest.labels_

X_outlier_free = X[labels == 0]
y_outlier_free = y[labels == 0]

print(y_outlier_free)