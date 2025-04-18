import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, SimpleImputer, OneHotEncoder, ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score
import shap

try:
    df_original = pd.read_csv(r"E:\AutoML\TestDatasets\CarPrice_Assignment.csv")
except FileNotFoundError:
    print("File not found. Please check the file path.")
    exit()

df = df_original.copy()

# Identify target variable
target_column = 'price'  # Assuming 'price' is the target variable
print("Target column:", target_column)

# Handle obvious non-feature columns
df.drop(columns=['id'], inplace=True, errors='ignore')

# Basic missing value imputation (if necessary)
df.fillna(df.median(), inplace=True)

# Infer task type and prepare target
if df[target_column].dtype == 'object' or len(df[target_column].unique()) <= 10:
    task_type = "Classification"
    le = LabelEncoder()
    df[target_column] = le.fit_transform(df[target_column])
else:
    task_type = "Regression"
print("Task type:", task_type)

X = df.drop(columns=[target_column])
y = df[target_column]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Model selection
if task_type == "Classification":
    models = [LogisticRegression(random_state=42), DecisionTreeClassifier(random_state=42), RandomForestClassifier(random_state=42)]
    scoring_metrics = ['accuracy', 'f1_macro']
else:
    models = [LinearRegression(), DecisionTreeRegressor(random_state=42), RandomForestRegressor(random_state=42)]
    scoring_metrics = ['neg_mean_squared_error', 'r2']

# Model training and evaluation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for model in models:
    scores = cross_val_score(model, X_train_processed, y_train, cv=kfold, scoring=scoring_metrics)
    for i, metric in enumerate(scoring_metrics):
        print(f"Model: {model.__class__.__name__}, Metric: {metric}, Mean: {scores[i].mean():.3f}, Std: {scores[i].std():.3f}")

# Final model training and test set evaluation
best_model = models[0]  # Choose the best model based on cross-validation results
best_model.fit(X_train_processed, y_train)
y_pred = best_model.predict(X_test_processed)

if task_type == "Classification":
    print("Test set accuracy:", accuracy_score(y_test, y_pred))
    print("Test set classification report:\n", classification_report(y_test, y_pred))
else:
    print("Test set mean absolute error:", mean_absolute_error(y_test, y_pred))
    print("Test set mean squared error:", mean_squared_error(y_test, y_pred))
    print("Test set R2 score:", r2_score(y_test, y_pred))

# Feature importance (optional)
if hasattr(best_model, 'feature_importances_'):
    feature_importances = best_model.feature_importances_
    feature_names = preprocessor.get_feature_names_out()
    print("Feature importances:")
    for i in np.argsort(feature_importances)[::-1][:10]:
        print(f"{feature_names[i]}: {feature_importances[i]:.3f}")