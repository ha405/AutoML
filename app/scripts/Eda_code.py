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
        missing_percentage = df[col].isnull().sum() / len(df)
        if missing_percentage < 0.3:
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

# Add columns likely to be identifiers
for col in df.columns:
    if ('id' in col.lower() or 'num' in col.lower()) and col not in ['customer_id', 'car_ID']:
        if col in df.columns:
            columns_to_drop.append(col)

# Add columns with > 90% unique values
for col in df.columns:
    if df[col].nunique() > 0.9 * len(df) and col not in ['CarName']:
        if col in df.columns:
            columns_to_drop.append(col)

columns_to_drop = list(set(columns_to_drop))  # Remove duplicates
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
print(f"Dropped columns: {columns_to_drop}")

# Preprocessing - Encoding Categorical Features
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
high_cardinality_cols = []

for col in categorical_cols:
    if df[col].nunique() == 2:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    elif df[col].nunique() <= 10:
        try:
            oe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
            ohe = oe.fit_transform(df[[col]])
            ohe_df = pd.DataFrame(ohe, index=df.index, columns=oe.get_feature_names_out([col]))
            df = pd.concat([df, ohe_df], axis=1)
            df.drop(col, axis=1, inplace=True)
        except Exception as e:
            print(f"OneHotEncoding failed for {col}: {e}")
    else:
        high_cardinality_cols.append(col)

df.drop(columns=high_cardinality_cols, inplace=True, errors='ignore')

# Preprocessing - Feature Engineering (Business Problem Driven)
if 'enginesize' in df.columns and 'horsepower' in df.columns:
    df['power_per_size'] = np.where(df['enginesize'] != 0, df['horsepower'] / df['enginesize'], 0)

# Final Checks
print("Processed DataFrame Head:\n", df.head())
print(f"Final Processed Shape: {df.shape}")
print("Final Data Types:\n", df.dtypes)
print("Summary Statistics (Processed Data):\n", df.describe(include='all'))

df.to_csv(file_path, index=False)