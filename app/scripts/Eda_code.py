import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

target_file_path = "d:\\AutoML\\TestDatasets\\Input.csv"
processed_file_path = "d:\\AutoML\\TestDatasets\\Input_processed.csv"

try:
    df = pd.read_csv(target_file_path)
except FileNotFoundError:
    print(f"Error: File not found at {target_file_path}")
    exit()

print("Initial Diagnostics:")
print("Shape:", df.shape)
print("Columns:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())

# --- Guided Preprocessing ---
# No specific guidance provided.  Implementing a basic strategy.

# 1. Target Variable: No explicit transformation needed. Assuming 'price' is the target and already numeric. If there were a target column named differently, it would be specified here to either balance or convert.
# In this case, we assume there's a 'price' column to predict.

# 2. Feature Selection/Dropping:  Dropping 'car_ID' and 'CarName' as they are unlikely to be useful features.
df = df.drop(['car_ID', 'CarName'], axis=1)

# 3. Missing Value Strategy: Impute missing values using the mean for numerical columns and the most frequent value for categorical columns.
numerical_cols = df.select_dtypes(include=np.number).columns
categorical_cols = df.select_dtypes(exclude=np.number).columns

for col in numerical_cols:
    if df[col].isnull().any():
        imputer = SimpleImputer(strategy='mean')
        df[col] = imputer.fit_transform(df[[col]])

for col in categorical_cols:
    if df[col].isnull().any():
        imputer = SimpleImputer(strategy='most_frequent')
        df[col] = imputer.fit_transform(df[[col]])

# 4. Duplicate Removal: Removing duplicate rows.
df = df.drop_duplicates()

# 5. Encoding: Apply OneHotEncoder to categorical features with cardinality < 10 (arbitrary threshold).
for col in categorical_cols:
    if df[col].nunique() < 10:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore') #handle_unknown ensures no errors if new categories appear
        encoded_data = encoder.fit_transform(df[[col]])
        encoded_cols = encoder.get_feature_names_out([col]) #Get names for the new columns
        encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=df.index) #Preserve index for proper join
        df = pd.concat([df, encoded_df], axis=1) #Concatenate along columns
        df = df.drop([col], axis=1) #Drop original categorical column


# 6. Feature Engineering: No specific feature engineering steps prescribed.

# 7. Scaling: Scale numerical features using StandardScaler.
scaler = StandardScaler()
numerical_cols = df.select_dtypes(include=np.number).columns
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])


# --- Final Diagnostics ---
print("\nFinal Diagnostics:")
print("Head:\n", df.head())
print("\nShape:", df.shape)
print("Columns:\n", df.dtypes)
print("\nSummary Statistics:\n", df.describe())

# Ensure no object columns remain unexpectedly (after one-hot encoding)
object_cols = df.select_dtypes(include=['object']).columns
if len(object_cols) > 0:
    print("\nWarning: Unexpected object columns found:", object_cols)

# --- Save Output ---
os.makedirs(os.path.dirname(processed_file_path), exist_ok=True)
df.to_csv(processed_file_path, index=False)

print(f"\nProcessed data saved to: {processed_file_path}")