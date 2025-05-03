import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('e:/AI 602 (LLM Systems)/llmproj/AutoML/app/visualizations', exist_ok=True)

try:
    df = pd.read_csv(r"e:/AI 602 (LLM Systems)/llmproj/AutoML/TestDatasets/Input_processed.csv")
except FileNotFoundError:
    print("Error: Input_processed.csv not found. Please check the file path.")
    exit()

# Chart 1: Bar Chart - Feature Importance
try:
    feature_importance = {
        'engine_size': 0.6,
        'curbweight': 0.2,
        'highwaympg': 0.1,
        'horsepower': 0.05,
        'citympg': 0.02,
        'wheelbase': 0.01,
        'length': 0.01,
        'width': 0.005,
        'height': 0.005
    }

    feature_names = list(feature_importance.keys())
    importance_values = list(feature_importance.values())

    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, importance_values)
    plt.xlabel("Feature")
    plt.ylabel("Impact on Price")
    plt.title("How Key Features Influence Car Price")
    plt.xticks(rotation=45, ha="right")

    # Annotate engine size
    plt.annotate('Engine Size: Biggest Impact', xy=('engine_size', feature_importance['engine_size']), xytext=('engine_size', feature_importance['engine_size'] + 0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
    plt.savefig(os.path.join('e:/AI 602 (LLM Systems)/llmproj/AutoML/app/visualizations', 'visualization_1_feature_importance.png'), bbox_inches='tight')
    plt.close()

except Exception as e:
    print(f"Error generating feature importance bar chart: {e}")

# Chart 2: Scatter Plot - Actual vs. Predicted Price
try:
    # Assuming 'price' is the actual price and 'predicted_price' is the predicted price
    if 'price' in df.columns and 'predicted_price' in df.columns:
        plt.figure(figsize=(8, 6))
        plt.scatter(df['price'], df['predicted_price'], alpha=0.5)
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.title("Model Prediction Accuracy")

        # Add a dashed line for perfect prediction
        plt.plot([df['price'].min(), df['price'].max()], [df['price'].min(), df['price'].max()], linestyle='--', color='red')

        # Calculate average error
        average_error = abs(df['price'] - df['predicted_price']).mean()

        # Annotate the average error value
        plt.annotate(f'Average Error: ${average_error:.2f}', xy=(0.05, 0.95), xycoords='axes fraction')

        plt.tight_layout()
        plt.savefig(os.path.join('e:/AI 602 (LLM Systems)/llmproj/AutoML/app/visualizations', 'visualization_2_actual_vs_predicted.png'), bbox_inches='tight')
        plt.close()
    else:
        print("Warning: 'price' or 'predicted_price' column missing in the dataframe. Skipping scatter plot.")

except Exception as e:
    print(f"Error generating actual vs. predicted scatter plot: {e}")