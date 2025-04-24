import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

try:
    df_original = pd.read_csv(r"d:\AutoML\TestDatasets\Input_processed.csv")
    df = df_original.copy()
except FileNotFoundError:
    print("Error: File not found at d:\\AutoML\\TestDatasets\\Input_processed.csv")
    exit()

target_column = 'price'
print(f"Target column: {target_column}")

if df[target_column].dtype == 'object' or df[target_column].nunique() <= 10:
    task_type = "Classification"
    le = LabelEncoder()
    df[target_column] = le.fit_transform(df[target_column])
else:
    task_type = "Regression"

print(f"Inferred task type: {task_type}")

X = df.drop(target_column, axis=1)
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

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
    ])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

if hasattr(preprocessor, 'get_feature_names_out'):
    feature_names = preprocessor.get_feature_names_out()
else:
    feature_names = None

if task_type == "Regression":
    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
        "RandomForestRegressor": RandomForestRegressor(random_state=42)
    }
    scoring_metrics = ['neg_mean_squared_error', 'r2']
else:
    models = {
        "LogisticRegression": LogisticRegression(random_state=42, solver='liblinear'),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
        "RandomForestClassifier": RandomForestClassifier(random_state=42)
    }
    scoring_metrics = ['accuracy', 'f1_macro']

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

from sklearn.pipeline import Pipeline

for model_name, model in models.items():
    cv_results[model_name] = {}
    for scoring_metric in scoring_metrics:
        scores = cross_val_score(model, X_train_processed, y_train, cv=kf, scoring=scoring_metric)
        cv_results[model_name][scoring_metric] = scores
        print(f"{model_name} - {scoring_metric}: Mean = {scores.mean():.4f}, Std = {scores.std():.4f}")

best_model_name = None
best_metric_value = -np.inf if task_type == "Classification" else np.inf
best_metric = 'accuracy' if task_type == "Classification" else 'neg_mean_squared_error'

for model_name, results in cv_results.items():
    if task_type == "Classification":
        if results['accuracy'].mean() > best_metric_value:
            best_metric_value = results['accuracy'].mean()
            best_model_name = model_name
    else:
        if results['neg_mean_squared_error'].mean() < best_metric_value:
            best_metric_value = results['neg_mean_squared_error'].mean()
            best_model_name = model_name

print(f"Best model: {best_model_name}")

best_model = models[best_model_name]
best_model.fit(X_train_processed, y_train)
y_pred = best_model.predict(X_test_processed)

if task_type == "Regression":
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R2: {r2:.4f}")
else:
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

if hasattr(best_model, "feature_importances_") and feature_names is not None:
    importances = best_model.feature_importances_
    feature_importances = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    print("\nTop 10 Feature Importances:")
    for feature, importance in feature_importances[:10]:
        print(f"{feature}: {importance:.4f}")