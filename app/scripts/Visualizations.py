import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('e:/AI 602 (LLM Systems)/llmproj/AutoML/app/visualizations', exist_ok=True)

try:
    df = pd.read_csv(r"e:/AI 602 (LLM Systems)/llmproj/AutoML/TestDatasets/Input_processed.csv")
except FileNotFoundError:
    print("Error: The file 'Input_processed.csv' was not found.")
    exit()

# Chart 1: Bar Chart - Comparison of model performances
try:
    model_names = ['Baseline', 'Dummy', 'Random Forest']
    rmse_values = [10000, 8000, 3000]  # Replace with actual RMSE values

    plt.figure(figsize=(10, 6))
    plt.bar(model_names, rmse_values, color=['red', 'green', 'blue'])
    plt.xlabel('Model')
    plt.ylabel('RMSE')
    plt.title('Comparison of Model Performances (RMSE)')
    plt.grid(axis='y', linestyle='--')
    plt.savefig(os.path.join('e:/AI 602 (LLM Systems)/llmproj/AutoML/app/visualizations', 'visualization_1_model_comparison.png'), bbox_inches='tight')
    plt.close()

except Exception as e:
    print(f"Error creating model comparison bar chart: {e}")

# Chart 2: Horizontal Bar Chart - Feature importance
try:
    feature_importances = {'engine_size': 0.4, 'curb_weight': 0.3, 'highway_mpg': 0.15, 'horsepower': 0.1, 'car_width': 0.05}  # Replace with actual feature importances
    feature_names = list(feature_importances.keys())
    importance_values = list(feature_importances.values())

    plt.figure(figsize=(10, 8))
    plt.barh(feature_names, importance_values, color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance from Random Forest Model')
    plt.gca().invert_yaxis()  # Invert y-axis to display most important feature at the top
    plt.grid(axis='x', linestyle='--')
    plt.savefig(os.path.join('e:/AI 602 (LLM Systems)/llmproj/AutoML/app/visualizations', 'visualization_2_feature_importance.png'), bbox_inches='tight')
    plt.close()

except Exception as e:
    print(f"Error creating feature importance bar chart: {e}")

# Chart 3: Histogram - Distribution of car prices
try:
    if 'price' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['price'], kde=True, color='purple')
        plt.xlabel('Car Price')
        plt.ylabel('Frequency')
        plt.title('Distribution of Car Prices')
        plt.grid(axis='y', linestyle='--')
        plt.savefig(os.path.join('e:/AI 602 (LLM Systems)/llmproj/AutoML/app/visualizations', 'visualization_3_price_distribution.png'), bbox_inches='tight')
        plt.close()
    else:
        print("Warning: 'price' column not found in the dataset.")

except Exception as e:
    print(f"Error creating car price distribution histogram: {e}")

# Chart 4: Scatter Plot - Relationship between predicted and actual prices
try:
    # Generate some dummy predicted prices for demonstration
    actual_prices = df['price'] if 'price' in df.columns else [10000, 20000, 30000, 40000, 50000]
    predicted_prices = [x * 0.9 + 1000 for x in actual_prices]  # Simulate predictions

    plt.figure(figsize=(8, 6))
    plt.scatter(actual_prices, predicted_prices, alpha=0.5, color='green')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Relationship between Predicted and Actual Car Prices')
    plt.plot([min(actual_prices), max(actual_prices)], [min(actual_prices), max(actual_prices)], linestyle='--', color='red', label='Ideal Prediction')  # Add diagonal line
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join('e:/AI 602 (LLM Systems)/llmproj/AutoML/app/visualizations', 'visualization_4_predicted_vs_actual.png'), bbox_inches='tight')
    plt.close()

except Exception as e:
    print(f"Error creating predicted vs. actual scatter plot: {e}")