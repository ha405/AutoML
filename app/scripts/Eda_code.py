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
print("Columns with data types:\n", df.dtypes)
print("\nMissing values per column:\n", df.isnull().sum())
print("\nNumber of duplicate rows:", df.duplicated().sum())

# --- Guided Preprocessing ---
# Target Variable Investigation
# Assuming 'price' is the target variable and is already float64

# Feature Exploration & Refinement
# Drop 'car_ID' as it's a unique identifier
df = df.drop('car_ID', axis=1)

# Extract CompanyName from CarName
df['CompanyName'] = df['CarName'].str.split(' ').str.get(0)
df = df.drop('CarName', axis=1)

# Correct typos in CompanyName
df['CompanyName'] = df['CompanyName'].replace({'toyouta': 'toyota', 'vokswagen': 'volkswagen', 'vw': 'volkswagen'})

# Feature Representation
# One-Hot Encoding for low cardinality categorical features
categorical_cols = ['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem', 'CompanyName']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Scaling numerical features
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
numerical_cols.remove('price') # Remove target variable from scaling

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Data Quality & Cleaning
# Check for any remaining missing values (handling potential inf introduced by scaling)
df = df.replace([np.inf, -np.inf], np.nan)
for col in df.columns:
    if df[col].isnull().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0]) #just in case there are any object dtypes

# --- Final Diagnostics ---
print("\nTransformed DataFrame Head:\n", df.head())
print("\nTransformed DataFrame Shape:", df.shape)
print("\nTransformed DataFrame Data Types:\n", df.dtypes)
print("\nTransformed DataFrame Summary Statistics:\n", df.describe())

# Verify no unexpected object-type columns remain.
if df.select_dtypes(include=['object']).shape[1] > 0:
    print("WARNING: Unexpected object-type columns remain. Review encoding steps.")

# --- Save Output ---
os.makedirs(os.path.dirname(processed_file_path), exist_ok=True)
df.to_csv(processed_file_path, index=False)
print(f"\nProcessed data saved to: {processed_file_path}")