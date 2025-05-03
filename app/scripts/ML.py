import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Load Data
file_path = 'e:\\AI 602 (LLM Systems)\\llmproj\\AutoML\\TestDatasets\\Input_processed.csv'
try:
    df_original = pd.read_csv(file_path)
    df = df_original.copy()
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit()

# Identify Target Variable
target_column = 'price'
print(f"Target column: {target_column}")

# Define Features (X) and Target (y)
y = df[target_column]
X = df.drop(target_column, axis=1)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection & Instantiation
baseline_model = LinearRegression()
dummy_model = DummyRegressor(strategy="mean")
rf_model = RandomForestRegressor(random_state=42)

# Model Training and Evaluation (using Cross-Validation)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Baseline Model
baseline_scores_rmse = cross_val_score(baseline_model, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')
baseline_rmse_mean = -baseline_scores_rmse.mean()
baseline_rmse_std = baseline_scores_rmse.std()
print(f"Baseline Model (LinearRegression) RMSE: Mean = {baseline_rmse_mean:.4f}, Std = {baseline_rmse_std:.4f}")

# Dummy Model
dummy_scores_rmse = cross_val_score(dummy_model, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')
dummy_rmse_mean = -dummy_scores_rmse.mean()
dummy_rmse_std = dummy_scores_rmse.std()
print(f"Dummy Model RMSE: Mean = {dummy_rmse_mean:.4f}, Std = {dummy_rmse_std:.4f}")

# Random Forest Model
rf_scores_rmse = cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')
rf_rmse_mean = -rf_scores_rmse.mean()
rf_rmse_std = rf_scores_rmse.std()
print(f"Random Forest Model RMSE: Mean = {rf_rmse_mean:.4f}, Std = {rf_rmse_std:.4f}")

# Final Model Training and Test Set Evaluation
best_model = rf_model  # Choose Random Forest as the best model based on cross-validation results (assumed)
print("Chosen best model: RandomForestRegressor")

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test Set RMSE: {rmse:.4f}")
print(f"Test Set MAE: {mae:.4f}")
print(f"Test Set R-squared: {r2:.4f}")

# Feature Importance
feature_importances = best_model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print("\nTop Feature Importances:")
print(importance_df.head())

# Save the Trained Model
joblib.dump(best_model, 'trained_model.joblib')
print("Trained model saved as trained_model.joblib")