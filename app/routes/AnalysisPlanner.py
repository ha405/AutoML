import os
import sys
import time
import re
import google.generativeai as genai
from constants import AUTOML_ROOT_DIR, SCRIPTS_PATH_REL

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyBF8Ik7v2Uwy_cRVzoDEj30g2oNpXPPlrQ")
MODEL_NAME = "gemini-2.0-flash"

OUTPUT_DIR_BASE = os.path.join(AUTOML_ROOT_DIR, SCRIPTS_PATH_REL)
os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)

EDA_GUIDANCE_TXT_FILE_PATH = os.path.join(OUTPUT_DIR_BASE, "eda_guidance_plan.txt")
ML_PLAN_TXT_FILE_PATH = os.path.join(OUTPUT_DIR_BASE, "ml_plan.txt") 

client = None
client_configured = False

if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GOOGLE_AI_API_KEY":
    print(
        "⚠ Warning: Google API Key not set or is placeholder."
        " Ensure the GOOGLE_API_KEY environment variable is set correctly.",
        file=sys.stderr
    )
else:
    genai.configure(api_key=GEMINI_API_KEY)
    client = genai.GenerativeModel(MODEL_NAME)
    print(f"Google AI client configured for Guided Planner using model {MODEL_NAME}.")
    client_configured = True


SYSTEM_TEMPLATE_GUIDED_PLANNER = r"""
You are an expert Senior Data Scientist creating a **Unified Analysis & ML Plan**.
Your role is to provide strategic guidance for the Exploratory Data Analysis (EDA) and preprocessing phase, followed by an outline for the Machine Learning (ML) plan. Your guidance should focus on *what* needs to be achieved and *how to communicate findings clearly* to non-technical stakeholders, rather than dictating exact code implementations.

*Input Context Provided:*

1.  Business Problem:
    <business_problem>
    {business_problem_str}
    </business_problem>

2.  Initial Dataset Details:
    <initial_dataset_details>
    {file_details_str}
    </initial_dataset_details>

**Your Task: Generate the Unified Plan**

Output **ONLY** the Markdown formatted plan below, starting *exactly* with "### Unified Analysis & ML Plan". Ensure clear separation between "Part 1: EDA & Preprocessing Guidance" and "Part 2: Machine Learning Plan".

### Unified Analysis & ML Plan

**Part 1: EDA & Preprocessing Guidance**

*   **Overall Goal:** Analyze the initial data to understand its characteristics and prepare it for the likely ML task identified below. The aim is to produce a cleaned dataset (`processed_dataset.csv`) suitable for modeling.
*   **1. Initial ML Task Assessment:**
    *   Based on the business problem and initial data, what is the most probable ML task? (e.g., Binary Classification, Regression). Briefly justify.
    *   Identify the most likely target variable column name. What is its apparent data type?
*   **2. Target Variable Investigation:**
    *   **Guidance for EDA:**
        *   Confirm the presence and data type of the potential target variable (`<target_variable_name>`). Are type conversions needed (e.g., object to numeric)?
        *   Analyze its distribution. For classification, check for class imbalance. Quantify if significant imbalance exists, as this will impact modeling choices.
*   **3. Feature Exploration & Refinement:**
    *   **Guidance for EDA:**
        *   Identify columns that appear irrelevant (e.g., unique IDs, high cardinality free text) or redundant. Consider dropping them.
        *   Examine date/time columns. Could components like month, day of week, or time differences be useful features? Consider extracting or converting them.
        *   Investigate categorical features. Pay attention to those with potentially high cardinality. How might these be handled effectively (e.g., grouping, encoding strategies)?
        *   Assess relationships: Explore correlations between numerical features and between features and the potential target.
*   **4. Data Quality & Cleaning:**
    *   **Guidance for EDA:**
        *   Thoroughly identify and profile missing values (NaNs, placeholders). Understand the extent and patterns of missingness.
        *   Based on the findings, consider appropriate imputation strategies for numerical (e.g., median, mean) and categorical (e.g., mode, 'Unknown' category) features. The EDA should implement a reasonable approach.
*   **5. Feature Representation:**
    *   **Guidance for EDA:**
        *   Determine how categorical features should be represented numerically for modeling. Consider techniques like One-Hot Encoding (for lower cardinality) or alternatives for higher cardinality features identified earlier.
        *   Consider if numerical features require scaling (e.g., using StandardScaler or MinMaxScaler), especially if distance-based algorithms or regularized models are likely candidates later. Recommend applying scaling as a standard good practice.
*   **6. Desired State of Output Dataset (`processed_dataset.csv`):**
    *   **Guidance for EDA:** The goal is to save a CSV named `processed_dataset.csv` where:
        *   Columns represent the final selected/engineered features and the target variable.
        *   All data is numerical (including encoded categoricals).
        *   Missing values have been addressed appropriately.
        *   The target variable is clean and in a suitable format for the ML task.

**Part 2: Machine Learning Plan**

*   **1. Anticipated ML Task & Target:**
    *   Reiterate the ML Task identified in Part 1.
    *   Confirm the expected target variable column name in `processed_dataset.csv`.
*   **2. Potential Feature Set:**
    *   Describe the types of features expected in `processed_dataset.csv` (e.g., original numerical scaled, OHE features, engineered date features). Exact names depend on EDA execution, but list key original columns likely to be included in some form.
*   **3. Modeling Strategy:**
    *   **Baseline Model(s):** Suggest simple baselines for comparison (e.g., Logistic Regression/Dummy Classifier for classification, Linear Regression/Dummy Regressor for regression).
    *   **Candidate Model(s):** Recommend 1-2 potentially stronger models based on the task (e.g., RandomForest, GradientBoosting, LightGBM). Briefly mention why they might be suitable.
*   **4. Evaluation Approach:**
    *   **Primary Metric:** Recommend a primary metric aligned with the business goal (e.g., F1-score, Accuracy, RMSE, MAE). Justify the recommendation.
    *   **Secondary Metrics:** Suggest other metrics to provide a more complete picture.
    *   **Validation:** Recommend a robust validation strategy (e.g., k-fold Cross-Validation, StratifiedKFold).
*   **5. Further Modeling Considerations:**
    *   Highlight key activities for the modeling phase itself, such as hyperparameter tuning, feature importance analysis, and iterating on feature engineering based on model insights.

"""

def _call_model(prompt: str) -> str:
    if not client_configured or not client:
        return "# Error: Google AI client not configured."

    print(f"Sending request to {MODEL_NAME} (GuidedPlanner)...")
    response = client.generate_content(contents={'parts': [{'text': prompt}]})
    print("Response received (GuidedPlanner).")

    if response.parts:
         return response.parts[0].text.strip().lstrip('```markdown').rstrip('```').strip()
    else:
         if response.prompt_feedback:
              print(f"⚠ Response may be blocked: {response.prompt_feedback}", file=sys.stderr)
              return f"# Error: Response generation failed or blocked by safety settings. Feedback: {response.prompt_feedback}"
         return "# Error: Received empty response from API."


def _split_and_save_plan(unified_plan: str, eda_guidance_path: str, ml_plan_path: str) -> tuple[str | None, str | None]:
    eda_plan_part = None
    ml_plan_part = None

    part1_heading = "**Part 1: EDA & Preprocessing Guidance**"
    part2_heading = "**Part 2: Machine Learning Plan**"

    part1_match = re.search(re.escape(part1_heading), unified_plan, re.IGNORECASE | re.MULTILINE)
    part2_match = re.search(re.escape(part2_heading), unified_plan, re.IGNORECASE | re.MULTILINE)

    if part1_match and part2_match:
        part1_content_start_index = part1_match.end() 
        part2_content_start_index = part2_match.end() 
        part2_heading_start_index = part2_match.start() 
        eda_plan_part = unified_plan[part1_content_start_index:part2_heading_start_index].strip()
        ml_plan_part = unified_plan[part2_content_start_index:].strip()

    elif part1_match:
         print("⚠ Warning: Found EDA Guidance (Part 1) heading but not ML Plan (Part 2) heading.", file=sys.stderr)
         eda_plan_part = unified_plan[part1_match.end():].strip() 
         ml_plan_part = "# Error: ML Plan (Part 2) marker not found in LLM response."
    else:
        print("❌ Error: Could not find 'Part 1: EDA & Preprocessing Guidance' heading marker in LLM response.", file=sys.stderr)
        return None, None

    if not eda_plan_part:
         print("⚠ Warning: Extracted EDA Guidance content is empty.", file=sys.stderr)
         eda_plan_part = "# Info: EDA Guidance content extracted as empty." # Placeholder if empty after split
    if not ml_plan_part:
         print("⚠ Warning: Extracted ML Plan content is empty.", file=sys.stderr)
         ml_plan_part = "# Info: ML Plan content extracted as empty." # Placeholder if empty after split

    with open(eda_guidance_path, 'w', encoding='utf-8') as f:
        f.write(eda_plan_part)
    print(f"EDA Guidance saved to: {eda_guidance_path}")

    with open(ml_plan_path, 'w', encoding='utf-8') as f:
        f.write(ml_plan_part)
    print(f"ML Plan saved to: {ml_plan_path}")

    return eda_plan_part, ml_plan_part


def generate_ml_plan(
    business_problem: str,
    file_details: dict
) -> str:
    missing = []
    if not business_problem:
        missing.append('business_problem')
    if not isinstance(file_details, dict) or not file_details:
        missing.append('file_details (must be a non-empty dictionary)')
    if missing:
        return f"# Error: Missing or invalid inputs for planning: {', '.join(missing)}"

    details_copy = file_details.copy()
    if 'sample_data' in details_copy and isinstance(details_copy['sample_data'], list):
        details_copy['sample_data'] = details_copy['sample_data'][:3]
    elif 'sample_data' in details_copy:
         del details_copy['sample_data']

    if 'columns' in details_copy and isinstance(details_copy['columns'], list) and len(details_copy['columns']) > 50:
         details_copy['columns'] = details_copy['columns'][:50] + ['... (truncated)']

    file_details_str = "\n".join(f"- {k}: {v}" for k, v in details_copy.items())

    prompt = SYSTEM_TEMPLATE_GUIDED_PLANNER.format(
        business_problem_str=business_problem,
        file_details_str=file_details_str,
    )

    unified_plan = _call_model(prompt)

    if unified_plan.startswith("# Error:"):
        return unified_plan

    eda_guidance_part, ml_plan_part = _split_and_save_plan(
        unified_plan,
        EDA_GUIDANCE_TXT_FILE_PATH, # Save EDA guidance to .txt
        ML_PLAN_TXT_FILE_PATH       # Save ML plan to .txt
    )

    if ml_plan_part is None or ml_plan_part.startswith("# Error"):
        return ml_plan_part if ml_plan_part else "# Error: Failed to split or save the generated plan."
    else:
        return ml_plan_part