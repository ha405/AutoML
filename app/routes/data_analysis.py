from google import genai

client = genai.Client(api_key="AIzaSyBoAFOxBSX1nxEF8lNuhJudPiHVTCRNK8Q")

SYSTEM_INSTRUCTION_EDA = """
You are an advanced AI Python data scientist. Your task is to generate a **fully executable** Python script that performs Exploratory Data Analysis (EDA) and **business-problem-driven data preprocessing**. 

**Important: Every preprocessing step must be based on the business problem and the possible ML models that may be used later. Do not perform unnecessary steps.** 

The dataset is always located at: `E:\AutoML\TestDatasets\sales_data_sample.csv`.  
Your script must dynamically **adapt preprocessing** based on the following:

### 1. **Business Problem & ML Task Understanding**
- Analyze the provided business problem.
- Determine if the ML task is **classification, regression, clustering, or anomaly detection**.
- Tailor feature engineering, scaling, and preprocessing **only for relevant ML models**.

### 2. **Smart Data Processing Based on ML Model Needs**
- **Regression:** Handle skewed data, scale numerical features if needed, remove extreme outliers if they impact predictions.
- **Classification:** Handle imbalanced classes, encode categorical variables appropriately.
- **Clustering:** Normalize features, reduce dimensionality if necessary.
- **Anomaly Detection:** Detect unusual patterns and retain extreme outliers for analysis.

### 3. **Dynamic Preprocessing Steps**
- **Missing Data Handling:** Drop or impute based on feature importance and ML model impact.
- **Outliers:** Remove or transform only if they negatively affect model performance.
- **Categorical Encoding:** Choose encoding strategy based on ML model type (tree-based, distance-based, etc.).
- **Date Features:** Extract only relevant features if they contribute to predictions.
- **Feature Scaling:** Apply only when necessary (e.g., for SVM, kNN, or neural networks).
- **Feature Selection:** Remove redundant features if correlation is too high and affects performance.

### 4. **Final Export**
- Save the **cleaned and preprocessed dataset** in the same directory as `processed_sales_data_sample.csv`.
"""

def generate_response(messages):
    """
    Generate a response from Gemini AI to produce a dynamic EDA script.
    """
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=messages
    )
    text = response.text.strip()

    # Remove code block markers if present
    if text.startswith("```python"):
        text = text[len("```python"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    return text

def generate_data_analysis_code(filedetails, business_problem):
    """
    Generate a Python EDA script dynamically based on the business problem and ML task.
    """
    context = f"""
    File Details: {filedetails}

    Business Problem: {business_problem}

    {SYSTEM_INSTRUCTION_EDA}
    """
    eda_code = generate_response(context)
    return eda_code

def save_code_to_file(code, filename="generated_eda.py"):
    """
    Save the dynamically generated EDA script to a specified file.
    """
    import os

    save_path = r"E:\AutoML\app\scripts"
    os.makedirs(save_path, exist_ok=True)
    
    file_path = os.path.join(save_path, filename)
    with open(file_path, "w") as f:
        f.write(code)

    print(f"EDA script saved at: {file_path}")
