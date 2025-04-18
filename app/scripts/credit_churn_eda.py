import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder

try:
    df = pd.read_csv(r"E:\AutoML\TestDatasets\credit_card_churn.csv").copy()
except FileNotFoundError:
    print("Error: File not found. Please ensure the file path is correct.")
    exit()

print(f"DataFrame Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("Data Types:\n", df.dtypes)
print("Missing Values:\n", df.isnull().sum())
print(f"Duplicate Rows: {df.duplicated().sum()}")

numeric_cols = df.select_dtypes(include='number').columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        if df[col].isnull().sum() / len(df) < 0.3:
            df[col].fillna(df[col].median(), inplace=True)
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
    if 'id' in col.lower() or 'num' in col.lower():
        if not ('customer_id' in col.lower()):
            columns_to_drop.append(col)

for col in df.columns:
    if df[col].nunique() > 0.9 * len(df):
        columns_to_drop.append(col)

columns_to_drop = list(set(columns_to_drop))

df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
print(f"Dropped columns: {columns_to_drop}")

categorical_cols = df.select_dtypes(include=['object', 'category']).columns

high_cardinality_cols = []
for col in categorical_cols:
    if df[col].nunique() == 2:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    elif df[col].nunique() <= 10:
        try:
            if col == 'Education_Level':
                oe = OrdinalEncoder(categories=[['Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate']]) #Assumption on Education level ordering
                df['Education_Level'] = oe.fit_transform(df[['Education_Level']])

            elif col == 'Income_Category':
                 oe = OrdinalEncoder(categories=[['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +']])  # Assumption on income category ordering
                 df['Income_Category'] = oe.fit_transform(df[['Income_Category']])
            else:
                ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
                ohe.fit(df[[col]])
                df[ohe.get_feature_names_out([col]).tolist()] = ohe.transform(df[[col]])
                df.drop(col, axis=1, inplace=True)
        except ValueError:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
            ohe.fit(df[[col]])
            df[ohe.get_feature_names_out([col]).tolist()] = ohe.transform(df[[col]])
            df.drop(col, axis=1, inplace=True)

    else:
        high_cardinality_cols.append(col)

df.drop(columns=high_cardinality_cols, inplace=True, errors='ignore')

if 'Total_Trans_Amt' in df.columns and 'Total_Trans_Ct' in df.columns:
    df['Avg_Trans_Value'] = np.where(df['Total_Trans_Ct'] != 0, df['Total_Trans_Amt'] / df['Total_Trans_Ct'], 0)

print("Processed DataFrame Head:\n", df.head())
print(f"Final Processed Shape: {df.shape}")
print("Final Data Types:\n", df.dtypes)
print("Summary Statistics (Processed Data):\n", df.describe(include='all'))