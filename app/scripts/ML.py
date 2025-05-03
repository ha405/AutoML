import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
import joblib

try:
    df_original = pd.read_csv("e:\\AI 602 (LLM Systems)\\llmproj\\AutoML\\TestDatasets\\Input_processed.csv")
    df = df_original.copy()
except FileNotFoundError:
    print("Error: Input_processed.csv not found. Please ensure the file exists at the specified path.")
    exit()

target_column = 'price'
print(f"Target column: {target_column}")

y = df[target_column]
X = df.drop(target_column, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_lr = LinearRegression()
model_rf = RandomForestRegressor(random_state=42)
model_gbr = GradientBoostingRegressor(random_state=42)
model_dummy = DummyRegressor(strategy="mean")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model(model, X, y, scoring='neg_root_mean_squared_error'):
    scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)
    rmse_scores = -scores
    print(f"Model: {type(model).__name__}")
    print(f"RMSE: Mean = {rmse_scores.mean():.4f}, Std = {rmse_scores.std():.4f}")
    return rmse_scores.mean()

print("Cross-validation results:")
rmse_lr = evaluate_model(model_lr, X_train, y_train)
rmse_rf = evaluate_model(model_rf, X_train, y_train)
rmse_gbr = evaluate_model(model_gbr, X_train, y_train)
rmse_dummy = evaluate_model(model_dummy, X_train, y_train)

best_model = None
best_rmse = float('inf')

if rmse_lr < best_rmse:
    best_rmse = rmse_lr
    best_model = model_lr
if rmse_rf < best_rmse:
    best_rmse = rmse_rf
    best_model = model_rf
if rmse_gbr < best_rmse:
    best_rmse = rmse_gbr
    best_model = model_gbr

print(f"\nBest model: {type(best_model).__name__}")

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nTest set evaluation:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R-squared: {r2:.4f}")

if isinstance(best_model, (RandomForestRegressor, LinearRegression)):
    try:
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_importances = pd.Series(importances, index=X.columns)
            print("\nFeature Importances:")
            print(feature_importances.sort_values(ascending=False).head(10))
        elif hasattr(best_model, 'coef_'):
            coefficients = best_model.coef_
            feature_coefficients = pd.Series(coefficients, index=X.columns)
            print("\nFeature Coefficients:")
            print(feature_coefficients.sort_values(ascending=False).head(10))
    except Exception as e:
        print(f"Error getting feature importances: {e}")

joblib.dump(best_model, 'trained_model.joblib')
print("Trained model saved as trained_model.joblib")