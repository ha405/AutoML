import os
from google import genai

client = genai.Client(api_key="AIzaSyBoAFOxBSX1nxEF8lNuhJudPiHVTCRNK8Q")

SYSTEM_INSTRUCTION_ML = """
You are an expert Python machine learning engineer. Generate a complete, fully executable Python script that builds and evaluates a machine learning model based on the following context:
- The final business problem is provided.
- Dataset metadata is provided, including number of rows, columns, column names, data types, and any other relevant details.
- The CSV dataset is always located at: E:\\AutoML\\processed_sales_data_sample.csv.
The script must:
1. Import all necessary libraries (such as pandas, numpy, scikit-learn, matplotlib, and seaborn).
2. Load the CSV file from the specified path.
3. Display basic dataset information (e.g., shape, column names, data types).
5. Select appropriate features based on the dataset metadata.
6. Split the data into training and testing sets.
7. Train an appropriate machine learning model (e.g., regression or classification) as guided by the final business problem.
8. Evaluate the model's performance using appropriate metrics (e.g., accuracy, RMSE) and print the results.
9. Do not include any comments inside code
Output only the final Python code without any markdown formatting (no triple backticks).
"""

def generate_response(messages):
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=messages
    )
    text = response.text.strip()
    # Remove markdown code fences if present.
    if text.startswith("```python"):
        text = text[len("```python"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text

def generate_ml_code(final_business_problem, dataset_info):
    """
    Generates an ML script using the provided final business problem and dataset info.
    
    Parameters:
        final_business_problem (str): The final business problem statement.
        dataset_info (dict): Metadata about the dataset (e.g., rows, columns, types).
    
    Returns:
        str: A complete Python script as a string.
    """
    context = f"""
Final Business Problem: {final_business_problem}

Dataset Information: {dataset_info}

Dataset Path: E:\\AutoML\\processed_sales_data_sample.csv

{SYSTEM_INSTRUCTION_ML}
"""
    ml_code = generate_response(context)
    return ml_code

def save_ml_code_to_file(code, filename="ML.py"):
    save_path = r"E:\AutoML\app\scripts"
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, filename)
    with open(file_path, "w") as f:
        f.write(code)
    print(f"ML script saved at: {file_path}")
