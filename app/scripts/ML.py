import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, SimpleImputer, ColumnTransformer, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import shap

try:
    df_original = pd.read_csv(r"E:\\AutoML\\processed_sales_data_sample.csv")
except FileNotFoundError:
    print("Error: Unable to read processed CSV file.")
    exit()

df = df_original.copy()

target_column = 'price'
print("Target column:", target_column)

df.drop(['car_ID'], axis=1, inplace=True)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
df[['horsepower', 'enginesize']] = imputer.fit_transform(df[['horsepower', 'enginesize']])
df['power_per_size'] = df['horsepower'] / df['enginesize']

task_type = 'Regression'
print("Task type:", task_type)

X = df.drop(['price'], axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

numeric_transformer = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
categorical_transformer = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore', sparse_output=False))

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

models = [LinearRegression(), DecisionTreeRegressor(random_state=42), RandomForestRegressor(random_state=42)]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for model in models:
    scores = cross_val_score(model, X_train_processed, y_train, cv=kf, scoring='neg_mean_squared_error')
    print(f"Model: {model.__class__.__name__}, Mean MSE: {np.mean(scores):.2f}, Std MSE: {np.std(scores):.2f}")

best_model = RandomForestRegressor(random_state=42)
best_model.fit(X_train_processed, y_train)

y_pred = best_model.predict(X_test_processed)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Final Model: {best_model.__class__.__name__}, MSE: {mse:.2f}, R2: {r2:.2f}")

importances = best_model.feature_importances_
feature_names = preprocessor.get_feature_names_out()
print("Feature Importances:")
for feature, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{feature}: {importance:.2f}")