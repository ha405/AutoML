Here is the corrected script:

```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, ColumnTransformer, SimpleImputer, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, mean_absolute_error
import shap
from sklearn.pipeline import Pipeline

try:
    df_original = pd.read_csv(r"E:\\AutoML\\processed_sales_data_sample.csv")
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

df = df_original.copy()

target_column = 'price'
print(f"Target column: {target_column}")

non_feature_columns = ['car_ID']
df.drop(non_feature_columns, axis=1, inplace=True)

# Basic missing value imputation (if necessary)
df.fillna(df.median(), inplace=True)

task_type = 'Regression'
print(f"Inferred task type: {task_type}")

X = df.drop(target_column, axis=1)
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

feature_names = preprocessor.named_transformers_['num'].named_steps['scaler'].feature_names_in_

models = [LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor()]

scores = []
for model in models:
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, X_train_processed, y_train, cv=kfold, scoring='neg_mean_squared_error')
    scores.append((model.__class__.__name__, np.mean(score), np.std(score)))

best_model_name, _, _ = max(scores, key=lambda x: x[1])
best_model = None
for model in models:
    if model.__class__.__name__ == best_model_name:
        best_model = model
        break

best_model.fit(X_train_processed, y_train)

y_pred = best_model.predict(X_test_processed)

print(f"Final model: {best_model.__class__.__name__}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.3f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.3f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.3f}")

if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    print("Feature Importances:")
    for feature, importance in zip(feature_names, importances):
        print(f"{feature}: {importance:.3f}")