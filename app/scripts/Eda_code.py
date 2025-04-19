import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

# Load Data
file_path = r"E:\AutoML\TestDatasets\Input.csv"
df = pd.read_csv(file_path)

# Initial Inspection
print(f"DataFrame Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("Data Types:\n", df.dtypes)
print("Missing Values:\n", df.isnull().sum())
print(f"Duplicate Rows: {df.duplicated().sum()}")

# Preprocessing - Missing Values
numeric_cols = df.select_dtypes(include='number').columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        if df[col].isnull().sum() < 0.3 * len(df):
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
        else:
            print(f"Dropping column {col} due to excessive missing values.")
            df.drop(col, axis=1, inplace=True)

categorical_cols = df.select_dtypes(include=['object', 'category']).columns
imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = imputer.fit_transform(df[categorical_cols])

print("Missing Values After Handling:\n", df.isnull().sum())

# Preprocessing - Duplicates
df.drop_duplicates(inplace=True)
print(f"Shape after dropping duplicates: {df.shape}")

# Preprocessing - Column Handling
columns_to_drop = []

for col in df.columns:
    if ('id' in str(col).lower() or 'num' in str(col).lower()) and col not in ['car_ID']:
        columns_to_drop.append(col)
    if df[col].nunique() > 0.9 * len(df):
        columns_to_drop.append(col)

columns_to_drop = list(set(columns_to_drop))

df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
print("Dropped Columns:", columns_to_drop)

# Preprocessing - Encoding Categorical Features
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
high_cardinality_cols = []

for col in categorical_cols:
    if df[col].nunique() == 2:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    elif df[col].nunique() <= 10:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
        ohe_df = pd.DataFrame(ohe.fit_transform(df[[col]]), index=df.index)
        ohe_df.columns = ohe.get_feature_names_out([col])
        df = pd.concat([df, ohe_df], axis=1)
        df.drop(col, axis=1, inplace=True)
    else:
        high_cardinality_cols.append(col)

df.drop(columns=high_cardinality_cols, inplace=True, errors='ignore')

# Preprocessing - Feature Engineering
if 'enginesize' in df.columns and 'horsepower' in df.columns:
    df['power_per_volume'] = np.where(df['enginesize'] != 0, df['horsepower'] / df['enginesize'], 0)

# Final Checks
print("Processed DataFrame Head:\n", df.head())
print(f"Final Processed Shape: {df.shape}")
print("Final Data Types:\n", df.dtypes)
print("Summary Statistics (Processed Data):\n", df.describe(include='all'))

# Save the dataframe
df.to_csv(file_path, index=False)