from groq import Groq
import os

# --- Configuration ---
# WARNING: Hardcoding API keys is insecure. Use environment variables or secrets management in production.
API_KEY = "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"
MODEL_NAME = "llama3-70b-8192"

os.environ["GROQ_API_KEY"] = API_KEY

try:
    client = Groq()
    print("Groq client configured for AnalysisPlanner.")
except Exception as e:
    print(f"❌ Error configuring Groq client in AnalysisPlanner: {e}")
    client = None

# --- Prompt Template for Super LLM ---
SYSTEM_INSTRUCTION_ANALYSIS_PLANNER_TEMPLATE = r"""
You are a highly experienced Senior Data Scientist acting as an ML Strategist.
Your task is to analyze the results of an automated Exploratory Data Analysis (EDA) process
and formulate a clear, well refined plan for the subsequent Machine Learning (ML) modeling phase.

**Input Context:**

<business_problem>
{business_problem_str}
</business_problem>

<initial_dataset_details>
{file_details_str}
</initial_dataset_details>

<executed_eda_script>
{eda_code_str}
</executed_eda_script>

<eda_execution_output_logs>
{eda_output_logs_str}
</eda_execution_output_logs>

Summarize EDA Findings: robustly and rigorously summarize the key outcomes reported in the <eda_execution_output_logs>. Focus on:
- Final data shape after preprocessing.
- How missing values were handled (imputation/drops mentioned in logs).
- How categorical features were encoded (based on logs/code).
- Any significant columns dropped (mentioned in logs/code).
- Any feature engineering performed (mentioned in logs/code).
- Identify Target Variable: Based on the <business_problem> and column names in <initial_dataset_details> or <eda_execution_output_logs>, state the most likely target variable for the ML model.
- Determine ML Task Type: Based on the identified target variable (e.g., its data type - check logs/details, number of unique values) and the <business_problem>, determine if the primary task is Classification or Regression. Justify your choice briefly.
- Recommend ML Model Types: Based on the ML Task Type, the final data characteristics (e.g., number of features/rows from EDA logs), and the business problem, recommend 2-3 specific types of ML models suitable for the next step (ML pipeline generation). Examples:
--->If Regression: "Linear Regression (baseline)", "Random Forest Regressor (handles non-linearity)", "Gradient Boosting Regressor (potential high accuracy)".
--->If Classification: "Logistic Regression (baseline)", "Random Forest Classifier (robust)", "LightGBM Classifier (efficient for larger data)".
- Briefly justify why these types are suitable (e.g., "handles mixed data types well", "good baseline", "interpretable").
- Suggest Specific Next Steps (Optional but helpful): Mention any specific considerations for the ML pipeline generation based on the EDA (e.g., "Ensure scaling is applied consistently," "Pay attention to feature 'X' identified as important during EDA").

Output Format:
- Provide your response as a structured plan using Markdown headings. Do NOT add conversational text outside this structure.
- EDA Summary
(Your summary based on logs and code)
- Target Variable
(Identified target variable name)
- ML Task Type
(Classification or Regression, with brief justification)
Recommended Model Types
Model Type 1 (e.g., Linear Regression): Justification...
Model Type 2 (e.g., Random Forest Regressor): Justification...
Model Type 3 (e.g., Gradient Boosting Regressor): Justification...
Next Step Considerations
(Optional: Specific points for the ML pipeline generator)

"""

def generate_response_for_planner(messages_list):
    if not client:
        return "# Error: Groq client not configured."
    try:
        print(f"Sending to {MODEL_NAME}...")
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages_list,
            temperature=0.2,
        )
        text = completion.choices[0].message.content or ""
        return text.strip()
    except Exception as e:
        return f"# Error generating analysis plan: {e}"

def generate_ml_plan(business_problem: str,
                     file_details: dict,
                     eda_code: str,
                     eda_output_logs: str) -> str:
    print("--- Generating ML Plan ---")
    # limit sample data for prompt length
    try:
        limited = file_details.get("sample_data", [])[:3]
        fd = file_details.copy()
        fd["sample_data"] = limited
        file_details_str = "\n".join(f"- {k}: {v}" for k, v in fd.items())
    except Exception as e:
        file_details_str = f"# Error formatting file details: {e}"

    prompt = SYSTEM_INSTRUCTION_ANALYSIS_PLANNER_TEMPLATE.format(
        business_problem_str=business_problem,
        file_details_str=file_details_str,
        eda_code_str=eda_code,
        eda_output_logs_str=eda_output_logs
    )
    messages = [{"role": "user", "content": prompt}]
    plan = generate_response_for_planner(messages)
    print("ML Plan Generation Complete.")
    return plan
