import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt

# 2. Data Loading
try:
    df = pd.read_csv("d:\\AutoML\\TestDatasets\\Input.csv")
except FileNotFoundError:
    print("Error: Input file not found.")
    exit()

# Rename 'money' column to 'sales_volume' since 'sales_volume' does not exist in the CSV file.
df.rename(columns={'money': 'sales_volume'}, inplace=True)

# 3. Initial Diagnostics
print("DataFrame Shape:", df.shape)
print("\nColumn List with Data Types:")
print(df.dtypes)
print("\nMissing Values per Column:")
print(df.isnull().sum())
print("\nNumber of Duplicate Rows:", df.duplicated().sum())

# 4. Guided Preprocessing
#   - Target Variable: sales_volume exists as float64. No conversion needed.
#   - Feature Selection/Dropping: No columns to drop initially, based on the plan.
#   - Missing Value Strategy: No missing values initially, but let's add a check as precaution
for col in df.columns:
    if df[col].isnull().any():  # Corrected way to check for missing values
        if df[col].dtype == 'float64' or df[col].dtype == 'int64':
            imputer = SimpleImputer(strategy='mean')  # numerical columns, impute with mean
        else:
            imputer = SimpleImputer(strategy='most_frequent')  # categorical, impute with mode
        df[col] = imputer.fit_transform(df[[col]])

#   - Duplicate Removal:
df.drop_duplicates(inplace=True)

# Extract features from the datetime column.
df['datetime'] = pd.to_datetime(df['datetime'])
df['month'] = df['datetime'].dt.month
df['day_of_week'] = df['datetime'].dt.dayofweek
df['day_of_year'] = df['datetime'].dt.dayofyear
df['hour_of_day'] = df['datetime'].dt.hour

#   - Encoding: OneHotEncoding for coffee_name, high cardinality
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)  # Corrected for sparse output
encoded_data = encoder.fit_transform(df[['coffee_name']])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['coffee_name'])) # Include feature names
df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1) # Reset index to avoid index mismatch
df.drop('coffee_name', axis=1, inplace=True) # Drop the original column

# Label encode cash_type
label_encoder = LabelEncoder()
df['cash_type'] = label_encoder.fit_transform(df['cash_type'])

# Drop 'date' and 'datetime' columns
df.drop(['date', 'datetime'], axis=1, inplace=True)

#   - Scaling: StandardScaler for numerical features
numerical_cols = df.select_dtypes(include=['number']).columns
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# 5. Visualizations
# Ensure the directory exists
os.makedirs("d:/AutoML/app/visualizations", exist_ok=True)

# Histogram of sales_volume
plt.figure(figsize=(10, 6))
plt.hist(df['sales_volume'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Sales Volume')
plt.ylabel('Frequency')
plt.title('Distribution of Sales Volume')
plt.margins(0.1, 0.1) # Add margins
plt.savefig("d:/AutoML/app/visualizations/sales_volume_distribution.png")
plt.close()

# Boxplot of sales_volume by cash_type
plt.figure(figsize=(10, 6))
df.boxplot(column='sales_volume', by='cash_type')
plt.xlabel('Cash Type')
plt.ylabel('Sales Volume')
plt.title('Sales Volume by Cash Type')
plt.suptitle('') # Remove the default title
plt.margins(0.1, 0.1) # Add margins
plt.savefig("d:/AutoML/app/visualizations/sales_volume_by_cash_type.png")
plt.close()

# Scatter plot of sales_volume vs. hour_of_day
plt.figure(figsize=(10, 6))
plt.scatter(df['hour_of_day'], df['sales_volume'], color='lightcoral')
plt.xlabel('Hour of Day')
plt.ylabel('Sales Volume')
plt.title('Sales Volume vs. Hour of Day')
plt.margins(0.1, 0.1) # Add margins
plt.savefig("d:/AutoML/app/visualizations/sales_volume_vs_hour_of_day.png")
plt.close()

# Plotting the correlation matrix
correlation_matrix = df.corr()

# Ensure enough space for labels
plt.figure(figsize=(12, 10))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar(shrink=0.8)
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=0)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title('Correlation Matrix of Features')
plt.margins(0.1, 0.1) # Add margins
plt.tight_layout()
plt.savefig("d:/AutoML/app/visualizations/correlation_matrix.png")
plt.close()

# 6. Final Diagnostics
print("\nTransformed DataFrame Head:")
print(df.head())
print("\nTransformed DataFrame Shape:", df.shape)
print("\nTransformed DataFrame Data Types:")
print(df.dtypes)
print("\nTransformed DataFrame Summary Statistics:")
print(df.describe())

# Verify no unexpected object-type columns remain
object_cols = df.select_dtypes(include=['object']).columns
print(f"\nObject type columns remaining: {object_cols}")

# 7. Save Output
os.makedirs(os.path.dirname("d:\\AutoML\\TestDatasets\\Input_processed.csv"), exist_ok=True)
df.to_csv("d:\\AutoML\\TestDatasets\\Input_processed.csv", index=False)
print("\nProcessed DataFrame saved to d:\\AutoML\\TestDatasets\\Input_processed.csv")