import pandas as pd
import seaborn as sns
from pyod.models.mad import MAD
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)

diamonds = sns.load_dataset("diamonds")


# Create the pairplot with a smaller figure size per subplot


sns.pairplot(diamonds.sample(500), hue="price")
plt.tight_layout()
plt.show()

diamonds.plot(kind='box',figsize=(15,10),subplots=True,layout=(3,3))
plt.show()



X = diamonds[["price"]]


mad = MAD().fit(X)

labels = pd.Series(mad.labels_, name="outlier_label")
diamonds["outlier_label"] = labels
# print(diamonds[diamonds['outlier_label']==1])


print(diamonds.describe())


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

iforest = IForest(n_estimators=1000,
                    # contamination=0.05,
                    max_samples=1000,
                    random_state=42)

iforest.fit(X)  

# Extract the labels
labels = iforest.labels_
diamonds["outlier_label"] = pd.Series(labels, name="outlier_label")






## now another eda with the iforest labels
sns.pairplot(diamonds.sample(500), hue="outlier_label")
plt.tight_layout()
plt.show()
