import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

file_path = r"C:\Users\HAMZA IQBAL\Downloads\AutoML\AutoML\TestDatasets\credit_card_churn.csv"

try:
    df = pd.read_csv(file_path).copy()

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Dtypes:\n", df.dtypes)
    print("Missing values:\n", df.isnull().sum())
    print("Describe:\n", df.describe())
    print("Nunique:\n", df.nunique())

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].isnull().sum() / len(df) < 0.3:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df = df.drop(col, axis=1)
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    cols_to_drop = [col for col in df.columns if 'id' in str(col).lower() or 'num' in str(col).lower()]
    cols_to_drop.extend([col for col in df.columns if df[col].nunique() / len(df) > 0.9])
    df = df.drop(columns=cols_to_drop, errors='ignore')

    for col in df.select_dtypes(include='object').columns:
        if df[col].nunique() == 2:
            df[col] = df[col].map({df[col].unique()[0]: 0, df[col].unique()[1]: 1})
        elif df[col].nunique() < 5:
             df = pd.get_dummies(df, columns=[col], drop_first=True)
        else:
            frequency = df[col].value_counts(normalize=True)
            df[col] = df[col].map(frequency)

    if 'EstimatedSalary' in df.columns and 'Balance' in df.columns:
        df['Salary_Balance_Ratio'] = df['EstimatedSalary'] / (df['Balance'] + 1e-6)
    elif 'CreditLimit' in df.columns and 'Income_Category' in df.columns:
        try:
            income_mapping = {
            'Less than $40K': 30000,
            '$40K - $60K': 50000,
            '$60K - $80K': 70000,
            '$80K - $120K': 100000,
            '$120K +': 150000
            }
            df['Income_Numeric'] = df['Income_Category'].map(income_mapping)
            df['CreditLimit_to_Income'] = df['CreditLimit'] / (df['Income_Numeric'] + 1e-6)
            df = df.drop(columns=['Income_Numeric'])

        except Exception as e:
            print(f"Error creating CreditLimit_to_Income: {e}")

    if 'Exited' in df.columns:
        X = df.drop('Exited', axis=1, errors='ignore')
        y = df['Exited'] if 'Exited' in df else None
    else:
         X = df.drop('Attrition_Flag', axis=1, errors='ignore')
         y = df['Attrition_Flag']

    print("Processed DataFrame Shape:", df.shape)
    print("Processed DataFrame Columns:", df.columns.tolist())

except Exception as e:
    print(f"Error processing CSV file: {e}")