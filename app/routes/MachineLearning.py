# MachineLearning.py
import time
from groq import Groq  # Import Groq
# from utils import filepreprocess # Assuming this is not needed directly here
import os

# --- Configuration ---
# WARNING: Hardcoding API keys is insecure. Use environment variables or secrets management in production.
API_KEY = "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" # User requested hardcoding
MODEL_NAME = "llama3-70b-8192" # Using the model known to work with Groq

# Set environment variable for Groq client (it often reads this automatically)
os.environ["GROQ_API_KEY"] = API_KEY

try:
    # Initialize Groq client
    # It will automatically use the GROQ_API_KEY environment variable if set
    client = Groq()
    print(f"Groq client configured with model: {MODEL_NAME}")
except Exception as e:
    print(f"❌ Error configuring Groq client: {e}")
    client = None # Set client to None if configuration fails

# --- Prompt Templates ---
# (SYSTEM_INSTRUCTION_ML_GENERATOR_TEMPLATE and SYSTEM_INSTRUCTION_ML_REFLECTOR_TEMPLATE remain the same as you provided)
SYSTEM_INSTRUCTION_ML_GENERATOR_TEMPLATE = r"""
You are an expert AI/ML engineer and Python programmer specializing in generating end-to-end machine learning pipelines.
Your goal is to create a complete, clean, robust, and executable Python script based on the provided context.

Note: The code should literally start off with the import statements. Don't include any introduction like "Here is the corrected script". 

**Input Context:**

<file_path>
{file_path_str}
</file_path>

<business_problem>
{business_problem_str}
</business_problem>

**Instructions for Python Script Generation:**

Generate a Python script that performs the following steps IN ORDER:

1.  **Imports:** Import necessary libraries: `pandas`, `numpy`, `sklearn.model_selection` (train_test_split, KFold, cross_val_score), `sklearn.preprocessing` (StandardScaler, LabelEncoder - if needed), relevant `sklearn.linear_model`, `sklearn.tree`, `sklearn.ensemble`, `sklearn.svm`, `sklearn.neural_network` models, and `sklearn.metrics`. Import `shap` if you plan to use it.
2.  **Load Data:**
    *   Load the CSV file from the path specified in `<file_path>`. Use `pd.read_csv(r"{file_path_str}")`.
    *   Handle potential `FileNotFoundError` with a clear error message and exit.
    *   Create a copy: `df = df_original.copy()`.
3.  **Initial Data Preparation (Minimal):**
    *   Identify the likely target variable based on the `<business_problem>` (e.g., 'price', 'churn', 'sales'). Print the identified target column name. If unsure, make a reasonable guess and print it.
    *   Handle obvious non-feature columns (e.g., drop unique ID columns if present and not the target).
    *   Perform basic missing value imputation (e.g., median for numeric, mode for categorical) ONLY IF ABSOLUTELY NECESSARY before determining task type. Prefer handling missing values *after* train/test split if possible, using fit on train data only. *For simplicity here, let's allow basic imputation before split if needed for target identification.*
4.  **Infer Task Type & Prepare Target:**
    *   Examine the identified target column:
        *   If dtype is 'object' or has few unique numeric values (e.g., <= 10), assume **Classification**. If target is object/categorical, use `LabelEncoder` to convert it to numeric BEFORE splitting.
        *   Otherwise, assume **Regression**.
    *   Print the inferred task type ("Classification" or "Regression").
    *   Define features `X` (all columns except target) and target `y`. Ensure `y` is numeric.
5.  **Train-Test Split:**
    *   Split `X` and `y` into training and testing sets (e.g., 80/20 split, `random_state=42`).
6.  **Preprocessing Pipeline (Fit on Train, Transform Both):**
    *   Identify numeric and categorical features in `X_train`.
    *   Create a `ColumnTransformer` pipeline:
        *   For numeric features: Use `StandardScaler()`. Handle missing values *within the pipeline* using `SimpleImputer(strategy='median')` before scaling.
        *   For categorical features: Handle missing values *within the pipeline* using `SimpleImputer(strategy='most_frequent')` then apply `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
    *   Fit the transformer ONLY on `X_train`.
    *   Transform both `X_train` and `X_test`. Store the processed data (it might be numpy arrays). Get feature names after transformation if possible (using `get_feature_names_out`).
7.  **Model Selection:**
    *   Based on the inferred task type, select **at least 3 appropriate models**:
        *   *If Classification:* e.g., `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`.
        *   *If Regression:* e.g., `LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`.
    *   Instantiate the selected models (use default hyperparameters or common settings like `random_state=42`).
8.  **Model Training and Evaluation (using Cross-Validation on Training Data):**
    *   Define `KFold` (e.g., `n_splits=5`, `shuffle=True`, `random_state=42`).
    *   For each selected model:
        *   Perform cross-validation using `cross_val_score` on the *preprocessed training data* (`X_train_processed`, `y_train`).
        *   Use appropriate scoring metric(s) based on task type:
            *   *If Classification:* `'accuracy'`, `'f1_macro'`
            *   *If Regression:* `'neg_mean_squared_error'`, `'r2'`
        *   Calculate and print the mean and standard deviation of the cross-validation scores for each metric.
9.  **Final Model Training and Test Set Evaluation:**
    *   Choose the best model based on cross-validation results (e.g., highest mean F1/Accuracy or R2/lowest MSE). State which model was chosen.
    *   Train the chosen best model on the *entire preprocessed training set* (`X_train_processed`, `y_train`).
    *   Make predictions on the *preprocessed test set* (`X_test_processed`).
    *   Calculate and print the final evaluation metrics on the test set predictions:
        *   *If Classification:* `accuracy_score`, `classification_report` (includes precision, recall, f1-score).
        *   *If Regression:* `mean_absolute_error`, `mean_squared_error`, `r2_score`.
10. **Feature Importance / Interpretation (Optional but Recommended):**
    *   If the best model has `feature_importances_` (like RandomForest), print the top 10 feature importances with their names (use names obtained from the preprocessor).
    *   *Advanced (Optional):* If `shap` is imported and the model is suitable (e.g., tree-based), calculate and print SHAP summary plot information (this might be complex to generate code for reliably, focus on `feature_importances_` first).
11. The code shouldnt contain anything like "Here is corrected code". It should ONLY contain code, no comments, nothing else. just python code
12. It shouldn't contain any introductory or concluding remarks about code either as no text is needed.
13. The code should literally start off with the import statements. Don't include any introduction like "Here is the corrected script". Omit any starting text please.

**Output Format:**
*   Your response MUST contain ONLY the raw Python code for the script.
*   Do NOT include any markdown formatting (like ```python ... ```).
*   Do NOT include any comments in the final code, unless explicitly needed (e.g., target variable guess).
*   The script must be fully executable if the `<file_path>` is valid and the necessary libraries are installed.
*   Include necessary imports at the beginning.

VERY IMPORTANT: The code shouldnt contain anything like "Here is corrected code" in starting. It should ONLY contain code, no comments, nothing else. just python code
AND: It shouldn't contain any introductory or concluding remarks about code either as no text is needed. ONLY RAW PYTHON CODE

The code should literally start off with the import statements. Don't include any introduction like "Here is the corrected script"
"""

SYSTEM_INSTRUCTION_ML_REFLECTOR_TEMPLATE = r"""
You are an expert Python code reviewer and AI/ML Quality Assurance specialist.
Your task is to meticulously review the provided Python script intended for building a machine learning pipeline.

**Context:**
The script was generated based on the following requirements summarized below:
<requirements_summary>
{requirements_summary}
</requirements_summary>

**Provided Script:**
<script>
{generated_code}
</script>

**Review Criteria:**

1.  **Correctness & Logic:**
    *   Does the code run without syntax errors?
    *   Is the file loaded correctly using the specified path (`{file_path_str}`)? Is `FileNotFoundError` handled?
    *   Is the task type (Classification/Regression) inferred correctly based on the likely target variable?
    *   Is the target variable prepared correctly (e.g., LabelEncoded if needed)?
    *   Is the train-test split performed correctly?
    *   Is the preprocessing pipeline (imputation, scaling, encoding) structured correctly using `ColumnTransformer`? Is it fitted ONLY on training data and used to transform both train and test sets?
    *   Are the selected models appropriate for the inferred task type?
    *   Is cross-validation performed correctly on the training set *after* preprocessing? Are appropriate scoring metrics used?
    *   Is the final model trained on the full (preprocessed) training set and evaluated on the (preprocessed) test set? Are appropriate final metrics calculated and printed?
    *   Is feature importance calculated correctly if applicable?
2.  **Completeness:**
    *   Does the script include all necessary imports?
    *   Are all major steps present (Load, Prep, Infer Task, Split, Preprocess, Select Models, CV Eval, Final Eval, Importance)?
    *   Are all specified print statements included (e.g., shape, target, task type, CV scores, final metrics)?
3.  **Adherence:**
    *   Does the script strictly follow the output format (only raw Python code)?
    *   Are there any unnecessary comments or markdown?
    *   Are only the allowed libraries imported?
4.  **Robustness:**
    *   Is the logic sound (e.g., avoiding data leakage between train/test sets during preprocessing)?
5.  **Others:**
    *   The code doesn't need to cater to error handling or data cleaning tasks.
6. The code shouldnt contain anything like "Here is corrected code". It should ONLY contain code, no comments, nothing else. just python code
7. It shouldn't contain any introductory or concluding remarks about code either as no text is needed.
8. Don't be critical for no accurate reason. If it does everything correctly, response ONLY with <OK>, NOTHING ELSE

Note: The code should literally start off with the import statements. Don't include any introduction like "Here is the corrected script". 

**Output:**
*   If the script meets all criteria, appears logically sound, and is likely to run correctly, respond ONLY with: `<OK>`
Note: The code should literally start off with the import statements. Don't include any introduction like "Here is the corrected script". 
*   Otherwise, provide concise, constructive feedback listing the specific issues found and suggest exact corrections needed. Be specific (e.g., "Line 45: Preprocessing pipeline should be fitted only on X_train, not the whole X."). Do NOT provide the fully corrected code, only the feedback/corrections list. Start feedback with "Issues found:".
"""

# --- Core Functions ---

def generate_response(messages_list): # Changed parameter name for clarity
    """Sends a prompt (message list) to the Groq model and returns the cleaned text response."""
    if not client:
        return "# Error: Groq client not configured."
    try:
        print(f"Sending request to {MODEL_NAME}...")

        # Use Groq's chat completions API
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages_list, # Pass the list of messages directly
            temperature=0.1, # Lower temperature for more deterministic code
            # max_tokens=4096, # Optional: set max tokens if needed
            # top_p=1, # Optional
            # stop=None, # Optional
            # stream=False, # Optional
        )
        print("Response received.")

        # Access the response content using Groq/OpenAI structure
        if completion.choices and completion.choices[0].message:
            text = completion.choices[0].message.content
        else:
            # Handle cases where the response might be blocked or empty
            print("Warning: Groq response might be empty or blocked.")
            # You might want to inspect completion.choices[0].finish_reason here
            text = f"# Error: No valid response text received. Finish Reason: {completion.choices[0].finish_reason if completion.choices else 'UNKNOWN'}"

        # Clean markdown (same as before)
        text = text.strip()
        if text.startswith("```python"):
            text = text[len("```python"):].strip()
        if text.startswith("```"):
             text = text[len("```"):].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
        return text

    except Exception as e:
        print(f"❌ An error occurred during Groq API call: {e}")
        return f"# Error generating response: {e}"

def generate_initial_ml_code(business_problem, file_path):
    """Generates the initial Python ML script."""
    print("Preparing prompt for initial ML code generation...")
    # Create the initial message list for the generator
    initial_messages = [
        {
            "role": "system",
            "content": SYSTEM_INSTRUCTION_ML_GENERATOR_TEMPLATE.format(
                business_problem_str=business_problem,
                file_path_str=file_path
            )
            # Note: The template itself acts as the system message here.
            # If you wanted a separate system message AND the template as user message,
            # you'd structure it differently. This approach is simpler.
        }
        # No initial 'user' message needed here as the system prompt contains all instructions
    ]
    ml_code = generate_response(initial_messages)
    print("Initial ML code generated.")
    return ml_code

def generate_and_refine_ml_code(business_problem, file_path, max_refinements=3):
    """Generates and refines the ML code using a self-evaluation loop."""

    # Generate concise requirements summary for the reflector
    requirements_summary = f"""
    - Load CSV from '{file_path}', handle FileNotFoundError.
    - Basic Prep: Identify target (print it), handle obvious non-features, minimal imputation if needed for target ID.
    - Infer Task Type (Classification/Regression) based on target, print type. Prep target (LabelEncode if needed). Define X, y.
    - Split data (80/20, random_state=42).
    - Preprocessing Pipeline (ColumnTransformer): Fit on train ONLY, transform train/test. Numeric: Impute(median)+Scale. Categorical: Impute(mode)+OneHot.
    - Select >= 3 appropriate models based on task type.
    - CV (KFold=5) on preprocessed TRAIN data. Use appropriate scoring (Class: acc, f1; Reg: neg_mse, r2). Print mean/std scores.
    - Train best model (from CV) on full preprocessed TRAIN data. Evaluate on preprocessed TEST data. Print final metrics (Class: acc, classification_report; Reg: mae, mse, r2).
    - Feature Importance: Print top 10 for tree models if applicable.
    - Output ONLY raw Python code (imports: pandas, numpy, sklearn), no comments (unless needed), no viz libs.
    """

    print("--- Generating Initial ML Code (Attempt 1) ---")
    current_code = generate_initial_ml_code(business_problem, file_path)
    # print(f"Initial Code:\n```python\n{current_code}\n```") # Debug

    if current_code.startswith("# Error"):
        print(f"Initial code generation failed: {current_code}")
        return current_code

    # --- Reflection Loop ---
    # History for the *generator* agent
    generation_history = [
        {
            "role": "system", # System prompt for the generator
            "content": SYSTEM_INSTRUCTION_ML_GENERATOR_TEMPLATE.format(
                business_problem_str=business_problem,
                file_path_str=file_path
            )
        },
        # The first 'assistant' response is the initial code
        {"role": "assistant", "content": current_code}
    ]

    for i in range(max_refinements):
        print(f"\n--- Reflection Cycle {i+1}/{max_refinements} ---")

        # Prepare message list for the *reflector* agent
        reflector_messages = [
             {
                 "role": "system",
                 "content": SYSTEM_INSTRUCTION_ML_REFLECTOR_TEMPLATE.format(
                     requirements_summary=requirements_summary,
                     generated_code=current_code,
                     file_path_str=file_path
                 )
             }
             # No user message needed, the system prompt contains the code to review
        ]

        print("Requesting critique...")
        critique = generate_response(reflector_messages) # Reflector uses its own prompt list
        print(f"Critique Received:\n{critique}")

        if critique.strip() == "<OK>":
            print("--- Code passed reflection. Finalizing. ---")
            return current_code
        elif not critique.strip().startswith("Issues found:") and critique.strip() != "<OK>":
             print("Warning: Reflector did not provide specific issues or <OK>. Using current code.")
             return current_code
        elif critique.startswith("# Error"):
             print(f"Error during reflection: {critique}. Returning current code.")
             return current_code
        else:
            print("Code needs refinement. Requesting revision...")
            # Add critique to the *generator's* history as if it's user feedback
            generation_history.append({"role": "user", "content": critique})

            # Request revision using the *generator's* perspective and history
            print("Sending refinement request with history...")
            # Send the updated history to the generator
            current_code = generate_response(generation_history)
            print("Code Revised.")
            # print(f"Revised Code:\n```python\n{current_code}\n```") # Debug

            if current_code.startswith("# Error"):
                print(f"Code refinement failed: {current_code}")
                print("Returning code from *before* this failed refinement attempt.")
                # Find the last valid assistant response before the error
                last_assistant_response = next((msg['content'] for msg in reversed(generation_history) if msg['role'] == 'assistant'), None)
                return last_assistant_response if last_assistant_response else "# Error: Refinement failed and couldn't retrieve previous code."

            # Update history with the latest *assistant* response
            generation_history.append({"role": "assistant", "content": current_code})


    print(f"\n--- Max refinements ({max_refinements}) reached. Returning last generated code. ---")
    return current_code