import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler

target_file_path = "e:\\AI 602 (LLM Systems)\\llmproj\\AutoML\\TestDatasets\\Input.csv"
processed_file_path = "e:\\AI 602 (LLM Systems)\\llmproj\\AutoML\\TestDatasets\\Input_processed.csv"

try:
    df = pd.read_csv(target_file_path)
except FileNotFoundError:
    print(f"Error: File not found at {target_file_path}")
    exit()

print("Initial Diagnostics:")
print("Shape:", df.shape)
print("Columns with data types:\n", df.dtypes)
print("\nMissing values per column:\n", df.isnull().sum())
print("\nNumber of duplicate rows:", df.duplicated().sum())

# --- Guided Preprocessing ---

# 1. Initial ML Task Assessment: Regression, target variable: price (already float64)

# 2. Target Variable Investigation: No transformation needed at this stage.

# 3. Feature Exploration & Refinement:
# Drop car_ID
df.drop('car_ID', axis=1, inplace=True)

# CompanyName cleanup (assuming this is CarName)
df['CompanyName'] = df['CarName'].str.split(' ').str.get(0) # Extract company name
df['CompanyName'] = df['CompanyName'].replace({'maxda': 'mazda', 'porcshce': 'porsche', 'toyouta': 'toyota', 'vokswagen': 'volkswagen', 'vw': 'volkswagen'})

# Encoding
#Label Encode Company Name
label_encoder = LabelEncoder()
df['CompanyName'] = label_encoder.fit_transform(df['CompanyName'])

# Binary Encoding
df['fueltype'] = df['fueltype'].map({'gas': 0, 'diesel': 1})
df['aspiration'] = df['aspiration'].map({'std': 0, 'turbo': 1})
df['doornumber'] = df['doornumber'].map({'two': 0, 'four': 1})
df['enginelocation'] = df['enginelocation'].map({'front': 0, 'rear': 1})

# One-Hot Encoding remaining categorical features
df = pd.get_dummies(df, columns=['carbody', 'drivewheel', 'enginetype', 'cylindernumber', 'fuelsystem'], drop_first=True)

#Drop CarName
df.drop('CarName', axis=1, inplace=True)

# 4. Data Quality & Cleaning: No missing values, outlier handling not explicitly specified, skipping for now

# 5. Feature Representation:
# Scaling numerical features
numerical_cols = df.select_dtypes(include=np.number).columns
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# --- Final Diagnostics ---
print("\nFinal Diagnostics:")
print("Transformed DataFrame head:\n", df.head())
print("New shape:", df.shape)
print("Data types:\n", df.dtypes)
print("Summary statistics:\n", df.describe())

# --- Save Output ---
os.makedirs(os.path.dirname(processed_file_path), exist_ok=True)
df.to_csv(processed_file_path, index=False)
print(f"\nProcessed DataFrame saved to {processed_file_path}")