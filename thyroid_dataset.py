from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

### data was downloaded from https://archive.ics.uci.edu/ml/datasets/Thyroid+Disease ###
BASE_DIR = Path(__file__).resolve().parent  ## this ensures compatibility across different environments
df = pd.read_csv(BASE_DIR / "data" / "hypothyroid.csv")


### eliminate the bad values first ###
df = df.replace(["?", " ?", "? ", "  ?"], np.nan)

for col in df.columns:
    print(f"\n{col}")
    print("Type of the column is:", df.dtypes[col])
    print(df[col].value_counts(dropna=False))
    

#### drop constant columns ####
constant_cols = df.columns[df.nunique(dropna=False) == 1]
print("Dropping columns:", constant_cols.tolist())
df = df.drop(columns=constant_cols)

#### casting the columns to proper values ####
cols_to_numeric = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]
df[cols_to_numeric] = (
    df[cols_to_numeric]
    .apply(pd.to_numeric, errors="coerce")
)



# # 3) now pick categorical columns
non_numeric_cols = df.select_dtypes(include=["object", "string", "category"]).columns
print(non_numeric_cols)



tf_map = {"t": 1, "f": 0, "yes": 1, "no": 0, "M": 1, "F": 0}

for col in non_numeric_cols:
    uniques = set(df[col].dropna().unique())
    if uniques.issubset(tf_map.keys()):
        df[col] = df[col].map(tf_map)



df = pd.get_dummies(df, columns=['referral source'], dummy_na=True)

num_cols = df.select_dtypes(include="number").columns

# add missing indicators
for col in num_cols:
    df[col + "_missing"] = df[col].isna().astype(int)

# impute
df[num_cols] = df[num_cols].fillna(df[num_cols].median())


print(df)
df.to_csv(BASE_DIR / "data" / "hypothyroid_cleaned.csv", index=False)





