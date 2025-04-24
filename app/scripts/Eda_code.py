import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder

target_file_path = r'd:\AutoML\TestDatasets\Input.csv'
processed_file_path = r'd:\AutoML\TestDatasets\Input_processed.csv'

df = pd.read_csv(target_file_path)

print(f"DataFrame Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("Data Types:\n", df.dtypes)
print("Missing Values:\n", df.isnull().sum())
print(f"Duplicate Rows: {df.duplicated().sum()}")

numeric_cols = df.select_dtypes(include='number').columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        if df[col].isnull().sum() / len(df) < 0.3:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
        else:
            print(f"Dropping column {col} due to excessive missing values.")
            df.drop(col, axis=1, inplace=True)

categorical_cols = df.select_dtypes(include=['object', 'category']).columns
imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = imputer.fit_transform(df[categorical_cols])

print("Missing Values After Handling:\n", df.isnull().sum())

df.drop_duplicates(inplace=True)
print(f"Shape after dropping duplicates: {df.shape}")

columns_to_drop = []
for col in df.columns:
    if 'id' in col.lower() and 'customer_id' not in col.lower():
        columns_to_drop.append(col)
    if 'num' in col.lower() and 'account_num' not in col.lower():
        columns_to_drop.append(col)
    if df[col].nunique() > 0.9 * len(df):
        columns_to_drop.append(col)
columns_to_drop = list(set(columns_to_drop))

df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
print(f"Dropped Columns: {columns_to_drop}")

categorical_cols = df.select_dtypes(include=['object', 'category']).columns

for col in categorical_cols:
    if df[col].nunique() == 2:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    elif df[col].nunique() <= 10:
        try:
            oe = OrdinalEncoder(categories=[['low', 'medium', 'high']]) # Assumption: Ordinality is low < medium < high
            df[col] = oe.fit_transform(df[[col]])
        except:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
            ohe_df = pd.DataFrame(ohe.fit_transform(df[[col]]), columns=ohe.get_feature_names_out([col]), index=df.index)
            df = pd.concat([df, ohe_df], axis=1)
            df.drop(col, axis=1, inplace=True)
    else:
        columns_to_drop.append(col)

df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

print("Processed DataFrame Head:\n", df.head())
print(f"Final Processed Shape: {df.shape}")
print("Final Data Types:\n", df.dtypes)
print("Summary Statistics (Processed Data):\n", df.describe(include='all'))

os.makedirs(os.path.dirname(processed_file_path), exist_ok=True)
df.to_csv(processed_file_path, index=False)
print(f"Processed data saved to: {processed_file_path}")