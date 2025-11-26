### Does having house rules affect other data such as popularity, review and price?



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)    # show all columns
pd.set_option('display.width', None)         # don't wrap columns
pd.set_option('display.width', 0)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


df = pd.read_csv('Airbnb_Open_Data.csv')


## Data exploration and cleaning
print(df.head())
print(df.info())
print(df.columns)
print(df.isnull().sum())
print(df.dtypes)
print(df.nunique())
print(df.describe(include='number'))
print(df.describe(include='object'))




## missing values in each column
print('Missing values percentages')
print(f'{100 * df.isnull().mean().round(4)}%')



cols = ['license', 'reviews per month', 'last review' , 'house_rules']
for c in cols:
    print(f"\n=== {c} ===")
    print(df[c].value_counts(dropna=False))
    print(df[c].isnull().sum())
    print(df[c].dtype)
    print(df[c].nunique())
# Based on the above, we can drop 'license', 'reviews per month', 'last review' , 'house_rules' columns
df = df.drop(columns=cols)
print(df.info())
## Data visualization
sns.boxplot(x='room type', y='price', data=df)  # compare numeric by category
plt.show()
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.show()
## Further data cleaning and preprocessing can be done based on the above analysis

print(df.describe(include='number'))
print(df.describe(include='object'))
cols = ['license', 'reviews per month', 'last review' , 'house_rules']
for c in cols:
    print(f"\n=== {c} ===")
    print(df[c].value_counts(dropna=False))
    print(df[c].isnull().sum())
    print(df[c].dtype)
    print(df[c].nunique())

df = df.drop(columns=cols)
print(df.info())



## dealing with missing values  
df.drop(columns=['license'], inplace=True)

# 1. Numeric columns → median
num_cols = df.select_dtypes(include='number').columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# 2. Categorical or ordinal columns → mode
cat_cols = df.select_dtypes(include=['object', 'category']).columns
df[cat_cols] = df[cat_cols].apply(lambda col: col.fillna(col.mode()[0]))



# pairplot
sns.set(style="ticks")
g = sns.pairplot(df.sample(500),
                 # hue="price"
                 )
g.fig.savefig("pairplot.png", dpi=300, bbox_inches="tight")

plt.show()


import seaborn as sns
# sns.histplot(df['price'], kde=True)  # numeric distribution
sns.boxplot(x='room type', y='price', data=df)  # compare numeric by category



corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")



## 3. Categorical Relationships

# pd.crosstab(df['room type'], df['price'], normalize=False)
sorted(df["price"].dropna().unique(), key=str)

# # Define price bins (adjust as needed)
# bins = [0, 500, 1000, 5000]
# labels = ["0-500", "500-1000", "1000-5000"]

# # Create a new binned column
# df["price_bin"] = pd.cut(df["price"].dropna(), bins=bins, labels=labels, include_lowest=True)

# # Crosstab using the binned prices
# ct = pd.crosstab(df["room type"], df["price_bin"], normalize=True)