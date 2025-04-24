# AnalysisPlanner.py
import os
from groq import Groq
import time # Optional for potential delays

# --- Configuration ---
# WARNING: Hardcoding API keys is insecure. Use environment variables or secrets management in production.
API_KEY = "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" # User requested hardcoding
MODEL_NAME = "llama3-70b-8192" # Using the model known to work with Groq

# Set environment variable for Groq client (it often reads this automatically)
os.environ["GROQ_API_KEY"] = API_KEY

client = None # Initialize client as None
try:
    # Initialize Groq client
    client = Groq(timeout=30.0) # Added a timeout
    print(f"✅ Groq client configured successfully for AnalysisPlanner (Model: {MODEL_NAME}).")
except Exception as e:
    print(f"❌ Error configuring Groq client: {e}")
    # Consider how the application should behave if the client fails to initialize

# --- Refined & Robust Prompt Template ---
# This template provides detailed instructions and emphasizes critical analysis of inputs.
SYSTEM_TEMPLATE_ROBUST_PLANNER = r"""
You are an expert Senior Data Scientist functioning as a strategic Machine Learning Planner.
Your primary objective is to *synthesize* information from the business context, initial data assessment, the EDA code generated, and MOST IMPORTANTLY, the *execution logs* from that EDA code. Based on this synthesis, you will create a *clear, actionable, and data-driven* plan for the subsequent ML modeling phase. Avoid generic advice; tailor recommendations to the specifics observed.

*Input Context Provided:*

1.  *Business Problem:* The high-level goal driving the analysis.
    <business_problem>
    {business_problem_str}
    </business_problem>

2.  *Initial Dataset Details:* Metadata extracted before the EDA script ran.
    <initial_dataset_details>
    {file_details_str}
    </initial_dataset_details>

3.  *Generated EDA Code:* The Python script that was intended to be run for EDA and preprocessing.
    <generated_eda_code>
    python
    {eda_code_str}
    
    </generated_eda_code>

4.  *EDA Execution Logs:* The *actual output (stdout/stderr)* generated when the EDA code was executed. *This is the most critical input for understanding the final state of the data.*
    <eda_execution_logs>
    
    {eda_output_logs_str}
    
    </eda_execution_logs>

*Your Task: Generate the ML Plan*

Analyze all inputs meticulously and produce a plan structured EXACTLY as follows using Markdown:

### ML Plan

*1. EDA Summary & Key Findings (Log-Driven):*
    *   **Focus exclusively on information confirmed by the <eda_execution_logs>.**
    *   State the final dimensions (shape) of the data after EDA, as reported in the logs.
    *   List any columns confirmed dropped or explicitly mentioned as kept/processed in the logs.
    *   Summarize the final missing value status based on the logs (e.g., "Logs confirm no missing values remain.").
    *   Mention any key transformations (e.g., encoding, scaling) indicated as completed in the logs.
    *   Note any errors or warnings reported in the stderr section of the logs.

*2. Target Variable Identification:*
    *   Based on the <business_problem> and confirmed by reviewing the <eda_execution_logs> (e.g., checking final columns), state the *exact column name* identified as the target variable (y).
    *   Specify its data type after EDA preprocessing (as inferred from logs or final df.dtypes if printed).

*3. ML Task Type:*
    *   Based on the target variable's nature (continuous, binary categorical, multi-class categorical) and the <business_problem>, clearly state the ML task: *Regression* or *Classification* (specify Binary or Multi-class if classification).

*4. Feature Set for Modeling:*
    *   List the *exact column names* that constitute the feature set (X) available for modeling after the EDA script's execution, according to the final state shown in the <eda_execution_logs>. Cross-reference with the EDA code's column dropping/creation steps.

*5. Recommended Modeling Approach:*
    *   *Baselines:* Suggest 1-2 simple baseline models appropriate for the task (e.g., DummyRegressor/DummyClassifier, LinearRegression/LogisticRegression).
    *   *Primary Candidates:* Recommend *2-3 more complex models* likely to perform well, justifying briefly based on data characteristics (if inferrable from logs/details) or task type. Examples:
        *   Regression: RandomForestRegressor, GradientBoostingRegressor, SVR, Lasso/Ridge. (Justification: "Tree ensembles for potential non-linearities", "Regularized linear models if high dimensionality").
        *   Classification: RandomForestClassifier, GradientBoostingClassifier, SVC, LogisticRegression (with regularization). (Justification: "Ensembles for robustness", "SVC for complex boundaries").
    *   Note: Explicitly state *NOT* to implement these yet, just recommend them for the next phase.

*6. Evaluation Strategy:*
    *   *Primary Metric:* Recommend the single most important metric to optimize based on the <business_problem> and task type (e.g., "Regression: R-squared (R2) to explain variance", "Classification (imbalanced): F1-score (weighted or macro) or Precision-Recall AUC", "Classification (balanced): Accuracy or F1-score").
    *   *Secondary Metrics:* List other relevant metrics to monitor (e.g., Regression: MAE, RMSE; Classification: Precision, Recall, Accuracy, Confusion Matrix).
    *   *Validation:* Strongly recommend *K-Fold Cross-Validation* (e.g., 5 or 10 folds) on the training set for comparing the recommended models robustly.

*7. Critical Next Step Considerations & Potential Issues (Log-Driven):*
    *   **Analyze the <eda_execution_logs> and <initial_dataset_details> for red flags or important notes for the ML Engineer.** Be specific and actionable. Examples:
        *   "*Data Size:* Logs show final dataset has [N] rows and [M] features. This might be small for complex models; emphasize robust CV."
        *   "*Imbalance:* Target variable analysis (if printed in logs or inferrable from initial details) suggests potential class imbalance for classification tasks. ML phase must use appropriate metrics (F1, PR-AUC) and consider techniques like stratification or resampling (SMOTE)."
        *   "*Feature Scaling:* Logs confirm/suggest numerical features were scaled using StandardScaler. Ensure any new data follows the same scaling."
        *   "*Encoding:* Logs confirm/suggest categorical features were OneHotEncoded. Note the potential increase in dimensionality ([M] final features)."
        *   "*High Cardinality:* Although EDA aimed to drop high-cardinality features, verify from logs if any remain unexpectedly."
        *   "*Errors in EDA:* Logs reported errors during [specific EDA step, e.g., encoding 'column_x']. This column might be missing or problematic for ML."
        *   *Feature Engineering:* "EDA logs indicate no significant feature engineering was performed. If baseline models perform poorly, exploring feature interactions/ratios could be a next step."
        *   *Data Leakage:* "Confirm that all preprocessing steps requiring fitting (imputation, scaling, encoding) were applied after the train-test split, using only training data for fitting (verify based on EDA code structure)."

Ensure the output is only the Markdown plan, starting directly with ### ML Plan.
"""

# --- Helper Function to Call Model ---
def _call_model(messages_list) -> str:
    """Sends messages list to Groq and returns cleaned text response."""
    if not client:
        return "# Error: Groq client not configured."
    try:
        print(f"Sending request to {MODEL_NAME} (AnalysisPlanner)...")
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages_list,
            temperature=0.2, # Keep low temp for structured output
            # max_tokens=2048, # Adjust if plans get truncated
        )
        print("Response received (AnalysisPlanner).")

        if completion.choices and completion.choices[0].message:
            text = completion.choices[0].message.content or "" # Ensure text is not None
        else:
            print("Warning: Groq response (AnalysisPlanner) might be empty or blocked.")
            text = f"# Error: No valid response text received. Finish Reason: {completion.choices[0].finish_reason if completion.choices else 'UNKNOWN'}"

        return text.strip() # Return raw text for the main function to handle

    except Exception as e:
        print(f"❌ An error occurred during Groq API call (AnalysisPlanner): {e}")
        # Consider logging the full traceback here for debugging
        return f"# Error generating response via API: {e}"

# --- Main Planner Function ---
def generate_ml_plan(
    business_problem: str,
    file_details: dict,
    eda_code: str,
    eda_output_logs: str
) -> str:
    """Generates the ML plan by analyzing EDA results using the robust prompt."""

    # Basic validation of inputs
    if not all([business_problem, file_details, isinstance(file_details, dict), eda_code, eda_output_logs]):
         missing = [name for name, val in locals().items() if not val or (name == 'file_details' and not isinstance(val, dict))]
         error_msg = f"# Error: Missing or invalid inputs for ML Plan generation: {', '.join(missing)}"
         print(error_msg)
         return error_msg

    # Prepare file details string, limiting sample data
    details_copy = file_details.copy()
    details_copy["sample_data"] = details_copy.get("sample_data", [])[:3] # Limit sample
    file_details_str = "\n".join(f"- {k}: {v}" for k, v in details_copy.items())

    # Ensure code and logs are treated as strings
    eda_code_str = str(eda_code)
    eda_output_logs_str = str(eda_output_logs)

    print("Formatting robust ML Plan prompt...")
    # Use the robust template
    prompt_content = SYSTEM_TEMPLATE_ROBUST_PLANNER.format(
        business_problem_str=business_problem,
        file_details_str=file_details_str,
        eda_code_str=eda_code_str,
        eda_output_logs_str=eda_output_logs_str
    )

    # Prepare messages list for the API call
    messages_list = [{"role": "user", "content": prompt_content}]

    print("Generating ML Plan using robust prompt...")
    ml_plan = _call_model(messages_list)

    # Basic check if the response indicates an error from the API call itself
    if ml_plan.startswith("# Error"):
        print(f"ML Plan generation failed with API error: {ml_plan}")
    else:
        print("ML Plan generated successfully.")

    return ml_plan