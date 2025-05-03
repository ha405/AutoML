import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
import joblib

try:
    df_original = pd.read_csv("e:\\AI 602 (LLM Systems)\\llmproj\\AutoML\\TestDatasets\\Input_processed.csv")
    df = df_original.copy()
except FileNotFoundError:
    print("Error: The file 'e:\\AI 602 (LLM Systems)\\llmproj\\AutoML\\TestDatasets\\Input_processed.csv' was not found.")
    exit()

target_column = 'price'
print(f"Target variable: {target_column}")

y = df[target_column]
X = df.drop(target_column, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model instantiation
linear_regression = LinearRegression()
dummy_regressor = DummyRegressor(strategy="mean")
random_forest = RandomForestRegressor(random_state=42)
gradient_boosting = GradientBoostingRegressor(random_state=42)

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model(model, X, y, scoring='neg_root_mean_squared_error'):
    scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)
    rmse_scores = -scores  # Convert negative RMSE to positive
    print(f"Model: {type(model).__name__}")
    print(f"RMSE: Mean = {rmse_scores.mean():.4f}, Std = {rmse_scores.std():.4f}")
    return rmse_scores.mean()

print("Cross-validation results:")
linear_regression_rmse = evaluate_model(linear_regression, X_train, y_train)
dummy_regressor_rmse = evaluate_model(dummy_regressor, X_train, y_train)
random_forest_rmse = evaluate_model(random_forest, X_train, y_train)
gradient_boosting_rmse = evaluate_model(gradient_boosting, X_train, y_train)

# Choose the best model based on cross-validation RMSE
best_model = None
best_rmse = float('inf')

if linear_regression_rmse < best_rmse:
    best_model = linear_regression
    best_rmse = linear_regression_rmse
if random_forest_rmse < best_rmse:
    best_model = random_forest
    best_rmse = random_forest_rmse
if gradient_boosting_rmse < best_rmse:
    best_model = gradient_boosting
    best_rmse = gradient_boosting_rmse

print(f"\nBest model: {type(best_model).__name__}")

# Train the best model on the entire training set
best_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model on the test set
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nTest set evaluation:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R-squared: {r2:.4f}")

# Feature Importance
if isinstance(best_model, RandomForestRegressor) or isinstance(best_model, GradientBoostingRegressor):
    importances = best_model.feature_importances_
    feature_importances = pd.Series(importances, index=X.columns)
    print("\nTop 10 Feature Importances:")
    print(feature_importances.nlargest(10))

# Save the trained model
model_filename = "trained_model.joblib"
joblib.dump(best_model, model_filename)
print(f"\nTrained model saved to: {model_filename}")