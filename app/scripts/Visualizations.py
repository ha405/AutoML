```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

os.makedirs('d:\AutoML\app\visualizations', exist_ok=True)

try:
    df = pd.read_csv(r"d:\AutoML\TestDatasets\Input.csv")
except FileNotFoundError:
    print("Error: Input.csv not found at d:\AutoML\TestDatasets\Input.csv")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Visualization 1: Bar Chart of Feature Importances
try:
    if 'feature_importance' in df.columns and 'feature_name' in df.columns:
        feature_importance = df[['feature_name', 'feature_importance']].sort_values(by='feature_importance', ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='feature_importance', y='feature_name', data=feature_importance)
        plt.title('Feature Importances in Car Price Prediction')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.savefig(os.path.join('d:\AutoML\app\visualizations', 'visualization_1_feature_importances.png'), bbox_inches='tight')
        plt.close()
    else:
        print("Warning: 'feature_importance' or 'feature_name' column missing. Skipping Feature Importance plot.")
except Exception as e:
    print(f"Error generating Feature Importance plot: {e}")

# Visualization 2: Scatter Plot of Predicted vs. Actual Prices
try:
    if 'predicted_price' in df.columns and 'actual_price' in df.columns:
        plt.figure(figsize=(8, 6))
        plt.scatter(df['actual_price'], df['predicted_price'])
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Predicted vs. Actual Car Prices')
        plt.savefig(os.path.join('d:\AutoML\app\visualizations', 'visualization_2_scatterplot.png'), bbox_inches='tight')
        plt.close()
    else:
        print("Warning: 'predicted_price' or 'actual_price' column missing. Skipping Predicted vs. Actual Prices plot.")
except Exception as e:
    print(f"Error generating Predicted vs. Actual Prices plot: {e}")

# Visualization 3: Heatmap of Correlation Matrix
try:
    numeric_df = df.select_dtypes(include=np.number)
    corr_matrix = numeric_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Car Features')
    plt.savefig(os.path.join('d:\AutoML\app\visualizations', 'visualization_3_heatmap.png'), bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"Error generating Correlation Matrix heatmap: {e}")

# Visualization 4: Line Plot of Evaluation Metrics
try:
    if 'RMSE' in df.columns and 'MAE' in df.columns and 'R-squared' in df.columns:
        metrics = ['RMSE', 'MAE', 'R-squared']
        values = [df['RMSE'].iloc[0], df['MAE'].iloc[0], df['R-squared'].iloc[0]]
        plt.figure(figsize=(8, 6))
        plt.plot(metrics, values, marker='o')
        plt.xlabel('Evaluation Metric')
        plt.ylabel('Value')
        plt.title('Model Evaluation Metrics')
        plt.grid(True)
        plt.savefig(os.path.join('d:\AutoML\app\visualizations', 'visualization_4_lineplot.png'), bbox_inches='tight')
        plt.close()
    else:
        print("Warning: 'RMSE', 'MAE', or 'R-squared' column missing. Skipping Evaluation Metrics plot.")
except Exception as e:
    print(f"Error generating Evaluation Metrics plot: {e}")

# Visualization 5: Histogram of Predicted Prices
try:
    if 'predicted_price' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df['predicted_price'], kde=True)
        plt.xlabel('Predicted Price')
        plt.ylabel('Frequency')
        plt.title('Distribution of Predicted Car Prices')
        plt.savefig(os.path.join('d:\AutoML\app\visualizations', 'visualization_5_histogram.png'), bbox_inches='tight')
        plt.close()
    else:
        print("Warning: 'predicted_price' column missing. Skipping Predicted Prices histogram.")
except Exception as e:
    print(f"Error generating Predicted Prices histogram: {e}")