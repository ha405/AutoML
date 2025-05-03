import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

target_file_path = "e:\\AI 602 (LLM Systems)\\llmproj\\AutoML\\TestDatasets\\Input.csv"
processed_file_path = "e:\\AI 602 (LLM Systems)\\llmproj\\AutoML\\TestDatasets\\Input_processed.csv"

try:
    df = pd.read_csv(target_file_path)
except FileNotFoundError:
    print(f"Error: File not found at {target_file_path}")
    exit()

# Initial Diagnostics
print("Shape:", df.shape)
print("\nColumns with data types:\n", df.dtypes)
print("\nMissing values per column:\n", df.isnull().sum()) # Check for missing values
print("\nDuplicate rows:", df.duplicated().sum())

# Drop car_ID
df = df.drop('car_ID', axis=1)

# Drop CarName (high cardinality, not useful without feature engineering which is out of scope)
df = df.drop('CarName', axis=1)

# Convert price to numeric (handling potential errors)
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Handle missing values - impute with the mean
for col in df.columns:
    if df[col].isnull().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            imputer = SimpleImputer(strategy='mean')
            df[col] = imputer.fit_transform(df[[col]])[:, 0]  # Fit and transform, then extract the first column
        else:
            imputer = SimpleImputer(strategy='most_frequent')
            df[col] = imputer.fit_transform(df[[col]])[:, 0]

# Remove duplicate rows
df = df.drop_duplicates()

# One-Hot Encode categorical features
categorical_cols = ['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem']
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # handle_unknown to prevent errors
encoded_data = encoder.fit_transform(df[categorical_cols])
encoded_cols = encoder.get_feature_names_out(categorical_cols)
encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols)
df = df.reset_index(drop=True)
encoded_df = encoded_df.reset_index(drop=True)
df = pd.concat([df.drop(categorical_cols, axis=1), encoded_df], axis=1)

# Scale numerical features
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
scaler = StandardScaler()

# Ensure correct number of rows before scaling
if df.shape[0] != len(df):
    raise ValueError("Index misalignment detected.")

df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Final Diagnostics
print("\nTransformed DataFrame Head:\n", df.head())
print("\nTransformed DataFrame Shape:", df.shape)
print("\nTransformed DataFrame Data Types:\n", df.dtypes)
print("\nTransformed DataFrame Summary Statistics:\n", df.describe())

# Check for object columns (should be none)
object_columns = df.select_dtypes(include=['object']).columns
if len(object_columns) > 0:
    print("\nWarning: Object columns remaining:", object_columns)

# Save Output
os.makedirs(os.path.dirname(processed_file_path), exist_ok=True)
df.to_csv(processed_file_path, index=False)
print(f"\nProcessed data saved to {processed_file_path}")