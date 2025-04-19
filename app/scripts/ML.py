Here is the corrected script:
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, SimpleImputer, OneHotEncoder, StandardScaler, ColumnTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score

try:
    df_original = pd.read_csv(r"E:\\AutoML\\processed_sales_data_sample.csv")
except FileNotFoundError:
    print("Error: Unable to read processed CSV file. Please check the file path.")
    exit()

df = df_original.copy()

# Identify target variable
target_column = df.select_dtypes(include=['int64', 'float64', 'object']).columns[-1]
print("Target column:", target_column)

# Handle non-feature columns
non_feature_columns = [col for col in df.columns if df[col].dtype not in ['int64', 'float64', 'object']]
df.drop(columns=non_feature_columns, inplace=True, errors='ignore')

# Basic missing value imputation
df.fillna(df.median(), inplace=True)

# Infer task type
if df[target_column].dtype == 'object' or len(df[target_column].unique()) <= 10:
    task_type = "Classification"
    le = LabelEncoder()
    df[target_column] = le.fit_transform(df[target_column])
else:
    task_type = "Regression"
print("Task type:", task_type)

X = df.drop(columns=[target_column])
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

numeric_transformer = ColumnTransformer(transformers=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())], remainder='passthrough')
categorical_transformer = ColumnTransformer(transformers=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))], remainder='passthrough')

preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

if task_type == "Classification":
    models = [LogisticRegression(random_state=42), DecisionTreeClassifier(random_state=42), RandomForestClassifier(random_state=42)]
    scoring_metrics = ['accuracy', 'f1_macro']
else:
    models = [LinearRegression(), DecisionTreeRegressor(random_state=42), RandomForestRegressor(random_state=42)]
    scoring_metrics = ['neg_mean_squared_error', 'r2']

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = []
for model in models:
    scores = cross_val_score(model, X_train_processed, y_train, cv=kfold, scoring=scoring_metrics)
    cv_scores.append((model, scores))
    for i, metric in enumerate(scoring_metrics):
        print(f"Model: {model.__class__.__name__}, Metric: {metric}, Mean: {scores[i].mean():.3f}, Std: {scores[i].std():.3f}")

# Choose the best model
best_model = max(cv_scores, key=lambda x: x[1].mean())[0]
print("Best model:", best_model.__class__.__name__)

best_model.fit(X_train_processed, y_train)

y_pred = best_model.predict(X_test_processed)

if task_type == "Classification":
    print("Test accuracy:", accuracy_score(y_test, y_pred))
    print("Test classification report:\n", classification_report(y_test, y_pred))
else:
    print("Test MAE:", mean_absolute_error(y_test, y_pred))
    print("Test MSE:", mean_squared_error(y_test, y_pred))
    print("Test R2:", r2_score(y_test, y_pred))

if hasattr(best_model, 'feature_importances_'):
    feature_importances = best_model.feature_importances_
    feature_names = X.columns
    print("Feature importances:")
    for i in np.argsort(feature_importances)[::-1][:10]:
        print(f"{feature_names[i]}: {feature_importances[i]:.3f}")