import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

try:
    df_original = pd.read_csv("d:\\AutoML\\TestDatasets\\Input_processed.csv")
    df = df_original.copy()
except FileNotFoundError:
    print("Error: The file 'd:\\AutoML\\TestDatasets\\Input_processed.csv' was not found.")
    exit()

target_column = 'price'
print(f"Target column: {target_column}")

y = df[target_column]
X = df.drop(target_column, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_regression = LinearRegression()
dummy_regressor = DummyRegressor(strategy="mean")
random_forest = RandomForestRegressor(random_state=42)
gradient_boosting = GradientBoostingRegressor(random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model(model, X_train, y_train, scoring='neg_root_mean_squared_error'):
    scores = cross_val_score(model, X_train, y_train, cv=kf, scoring=scoring)
    rmse_scores = -scores  # Convert negative RMSE to positive
    print(f"Model: {type(model).__name__}")
    print(f"RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std():.4f})")
    return rmse_scores.mean()

print("Cross-validation results:")
linear_regression_rmse = evaluate_model(linear_regression, X_train, y_train)
dummy_regressor_rmse = evaluate_model(dummy_regressor, X_train, y_train)
random_forest_rmse = evaluate_model(random_forest, X_train, y_train)
gradient_boosting_rmse = evaluate_model(gradient_boosting, X_train, y_train)

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

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nTest set evaluation:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R-squared: {r2:.4f}")

if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feature_importances = pd.Series(importances, index=X.columns)
    print("\nFeature Importances:")
    print(feature_importances.sort_values(ascending=False).head(10))

joblib.dump(best_model, 'trained_model.joblib')
print("Trained model saved as trained_model.joblib")