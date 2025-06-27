import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

try:
    df_original = pd.read_csv('d:\\AutoML\\TestDatasets\\Input_processed.csv')
    df = df_original.copy()
except FileNotFoundError:
    print("Error: The file 'd:\\AutoML\\TestDatasets\\Input_processed.csv' was not found.")
    exit()

target_column = 'sales_volume'
print(f"Target variable: {target_column}")

y = df[target_column]
X = df.drop(target_column, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_regression = LinearRegression()
random_forest = RandomForestRegressor(random_state=42)
gradient_boosting = GradientBoostingRegressor(random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lr_scores_rmse = cross_val_score(linear_regression, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')
rf_scores_rmse = cross_val_score(random_forest, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')
gb_scores_rmse = cross_val_score(gradient_boosting, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')

print("Linear Regression RMSE: Mean = {:.4f}, Std = {:.4f}".format(-lr_scores_rmse.mean(), lr_scores_rmse.std()))
print("Random Forest RMSE: Mean = {:.4f}, Std = {:.4f}".format(-rf_scores_rmse.mean(), rf_scores_rmse.std()))
print("Gradient Boosting RMSE: Mean = {:.4f}, Std = {:.4f}".format(-gb_scores_rmse.mean(), gb_scores_rmse.std()))

best_model = RandomForestRegressor(random_state=42)
print("Chosen model: Random Forest Regressor")

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Test Set RMSE: {:.4f}".format(rmse))
print("Test Set MAE: {:.4f}".format(mae))
print("Test Set R-squared: {:.4f}".format(r2))

feature_importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)

joblib.dump(best_model, 'trained_model.joblib')
print("Trained model saved to trained_model.joblib")

os.makedirs('d:/AutoML/app/visualizations', exist_ok=True)

top_features = feature_importances.head(15)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_features.values, y=top_features.index)
plt.title('Top Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Feature Name')
plt.tight_layout()
plt.savefig(os.path.join('d:/AutoML/app/visualizations', 'feature_importance.png'))
plt.close()
print("Feature importance plot saved to d:/AutoML/app/visualizations/feature_importance.png")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted Values')
plt.tight_layout()
plt.savefig(os.path.join('d:/AutoML/app/visualizations', 'actual_vs_predicted.png'))
plt.close()
print("Actual vs. Predicted plot saved to d:/AutoML/app/visualizations/actual_vs_predicted.png")