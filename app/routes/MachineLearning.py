import google.generativeai as genai
import os
import sys
import re # Keep re import
import time # Keep time import

# --- Constants ---
# Assuming PROCESSED_DATASET_PATH is defined elsewhere if needed by the generated code itself
# from constants import PROCESSED_DATASET_PATH

# --- Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyBF8Ik7v2Uwy_cRVzoDEj30g2oNpXPPlrQ")
MODEL_NAME = "gemini-2.0-flash" # Use 1.5 Flash

client_configured = False
model = None
try:
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_AI_API_KEY":
        print("⚠ Warning: Google API Key not set or is placeholder.", file=sys.stderr)
    else:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)
        print(f"Google AI client configured for ML Generator/Reflector using model {MODEL_NAME}.")
        client_configured = True
except Exception as e:
    print(f"❌ Error configuring Google AI client: {e}", file=sys.stderr)


# --- Prompt Templates ---

SYSTEM_INSTRUCTION_ML_GENERATOR_TEMPLATE = r"""
You are an expert AI/ML engineer and Python programmer specializing in generating end-to-end machine learning pipelines.
Your goal is to create a complete, clean, robust, and executable Python script based on the provided context, **paying close attention to the EDA output logs for insights about the final processed data.**

**Input Context:**

<file_path>
{file_path_str}
</file_path>

<business_problem>
{business_problem_str}
</business_problem>

<ml_plan>
{ml_plan_str}
</ml_plan>

<eda_output_logs>
{eda_output_logs_str}
</eda_output_logs>

**Instructions for Python Script Generation:**

Generate a Python script that performs the following steps IN ORDER, **using information from the EDA logs to confirm data state**:

1.  **Imports:** Import necessary libraries: pandas, numpy, sklearn.model_selection, sklearn.metrics, and specific models mentioned in the `<ml_plan>` or suitable standard ones (e.g., LogisticRegression, RandomForestClassifier/Regressor). Import joblib for saving the model.
2.  **Load Data:**
    *   Load the **processed** CSV file from the path specified in `<file_path>`. **Assume this file (`{file_path_str}`) is the output of the EDA script.**
    *   Handle potential `FileNotFoundError`.
    *   Create a copy: `df = df_original.copy()`.
3.  **Identify Target Variable:**
    *   Determine the target variable column name based on the `<ml_plan>` and verify its presence using column names mentioned in the final checks of the `<eda_output_logs>`. Print the identified target column name.
    *   **Crucially, check the "Final Data Types" section in the EDA logs to confirm the target variable's type.** If it's not numeric and the task is classification, apply LabelEncoder (ensure it was likely handled in EDA based on the plan/logs, but include encoder just in case if type is still object).
4.  **Define Features (X) and Target (y):**
    *   Define `y` using the identified target column.
    *   Define `X` as all columns *except* the target column. Verify these feature columns match those expected from the EDA logs ("Final Processed Shape", "Final Data Types").
5.  **Train-Test Split:**
    *   Split `X` and `y` into training and testing sets (e.g., 80/20 split, `random_state=42`). Use `stratify=y` if the EDA logs or ML plan indicated a classification task, especially if imbalance was noted.
6.  **Model Selection & Instantiation:**
    *   Instantiate the models recommended in the `<ml_plan>` (Baseline and Candidate models). Use default hyperparameters or common settings like `random_state=42`.
7.  **Model Training and Evaluation (using Cross-Validation on Training Data):**
    *   Define KFold (e.g., `n_splits=5`, `shuffle=True`, `random_state=42`). Use `StratifiedKFold` if classification.
    *   For each selected model:
        *   Perform cross-validation using `cross_val_score` on the **training data (X_train, y_train)**. **Note:** Preprocessing (scaling) should have been done in the EDA step according to the plan, so it's generally *not* needed again here unless the plan specifically deferred it. Assume data loaded from `<file_path>` is ready.
        *   Use the scoring metric(s) specified in the `<ml_plan>`'s Evaluation Approach.
        *   Calculate and print the mean and standard deviation of the cross-validation scores.
8.  **Final Model Training and Test Set Evaluation:**
    *   Choose the best model based on the primary cross-validation metric mentioned in the `<ml_plan>`. State which model was chosen via print.
    *   Train the chosen best model on the **entire training set (X_train, y_train)**.
    *   Make predictions on the **test set (X_test)**.
    *   Calculate and print the final evaluation metrics on the test set predictions, using the primary and secondary metrics specified in the `<ml_plan>`.
9.  **Feature Importance / Interpretation (Optional but Recommended):**
    *   If the best model allows (e.g., RandomForest, LogisticRegression with coef_), print the top N feature importances or coefficients. Use the column names from `X.columns`.
10. **Save the Trained Model:**
    *   Save the **trained best model** using `joblib.dump()`. Choose a reasonable filename (e.g., `trained_model.joblib`). Print a confirmation message.

**Output Format & Constraints:**
*   Output ONLY raw Python code.
*   Start directly with imports. No introductory/concluding text, no markdown fences (```python).
*   No comments unless explaining a necessary assumption not covered by the plan/logs.
*   Ensure the script uses the file path `{file_path_str}` for loading the **processed** data.

"""

SYSTEM_INSTRUCTION_ML_REFLECTOR_TEMPLATE = r"""
You are an expert Python code reviewer and AI/ML Quality Assurance specialist.
Your task is to meticulously review the provided Python script intended for training and evaluating machine learning models, ensuring it aligns with the ML Plan and considers the EDA outputs.

**Input Context:**

<business_problem>
{business_problem_str}
</business_problem>

<ml_plan>
{ml_plan_str}
</ml_plan>

<eda_output_logs>
{eda_output_logs_str}
</eda_output_logs>

<processed_data_path>
{file_path_str}
</processed_data_path>

**Provided Script:**
<script>
{generated_code}
</script>

**Review Criteria:**

1.  **Plan/Log Adherence:**
    *   Does the script load the processed data from the correct `<processed_data_path>`?
    *   Does it correctly identify the target variable based on the `<ml_plan>` and confirmed in `<eda_output_logs>`?
    *   Are the features (X) defined correctly based on the expected columns from EDA logs?
    *   Is stratification used in train/test split if indicated by the task/plan?
    *   Does the script instantiate the models mentioned in the `<ml_plan>`?
    *   Is cross-validation performed using the metrics and strategy outlined in the `<ml_plan>`? **Crucially, does it correctly assume data is already preprocessed based on EDA?** (i.e., no redundant scaling/encoding unless explicitly planned).
    *   Is the final evaluation performed using the metrics specified in the `<ml_plan>`?
    *   Is the trained model saved?
2.  **Correctness & Logic:**
    *   Is the code syntactically correct?
    *   Is `FileNotFoundError` handled during loading?
    *   Is the train/test split performed correctly?
    *   Are models trained on `X_train` and evaluated on `X_test`?
    *   Are metrics calculated correctly?
3.  **Completeness:**
    *   Are necessary libraries imported (pandas, numpy, sklearn, joblib)?
    *   Are all major steps from the plan present (Load, Define X/y, Split, CV, Final Train/Eval, Save Model)?
    *   Are key results printed (target, chosen model, CV scores, final metrics)?
4.  **Format Adherence:**
    *   Is the output *only* raw Python code? No markdown, no extra text, no unnecessary comments?

**Output:**
*   If the script correctly implements the ML plan based on the context and meets all other criteria, respond ONLY with: `<OK>`. NOTHING ELSE.
*   Otherwise, provide concise, constructive feedback listing the specific issues (e.g., "Line 55: Script includes StandardScaler, but EDA plan indicates scaling was already done. Remove scaling step."). Start feedback with "Issues found:". Do NOT provide the fully corrected code.
"""


def generate_response(messages_list):
    """Sends a prompt (message list) to the Gemini model and returns the cleaned text response."""
    if not client_configured or model is None:
        error_msg = "# Error: Google AI client not configured properly for ML generation."
        if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_AI_API_KEY":
            error_msg = "# Error: Google AI API Key not configured."
        print(f"ERROR ({__file__}): {error_msg}", file=sys.stderr)
        return error_msg

    try:
        print(f"Sending request to {MODEL_NAME} (ML Generator/Reflector)...")
        generation_config = genai.types.GenerationConfig(temperature=0.1)
        response = model.generate_content(contents=messages_list, generation_config=generation_config)
        print("Response received.")

        if response.prompt_feedback and response.prompt_feedback.block_reason:
            error_msg = f"# Error: Prompt blocked by Google AI due to {response.prompt_feedback.block_reason}"
            print(f"❌ {error_msg}", file=sys.stderr)
            return error_msg

        try:
            text = response.text
        except (ValueError, AttributeError) as e:
            error_msg = f"# Error extracting text from response ({type(e).__name__}). Blocked or unexpected format?"
            print(f"❌ {error_msg} (Candidates: {response.candidates})", file=sys.stderr)
            if response.candidates and response.candidates[0].finish_reason != 'STOP':
                error_msg += f" Finish Reason: {response.candidates[0].finish_reason}"
            return error_msg
        except Exception as e: # Catch other potential issues during text extraction
            error_msg = f"# Error processing response content: {e}"
            print(f"❌ {error_msg} (Response object: {response})", file=sys.stderr)
            return error_msg


        cleaned = text.strip()
        if cleaned.startswith("```python"):
            cleaned = cleaned[len("```python"):].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[len("```"):].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        # Added check: If after cleaning, it's empty, treat as error
        if not cleaned:
             print(f"WARNING ({__file__}): Cleaned response text is empty. Original text was likely minimal or formatting.", file=sys.stderr)
             return "# Error: Received empty or non-code response after cleaning."
        return cleaned

    except Exception as e:
        print(f"❌ An error occurred during Google AI API call: {e}", file=sys.stderr)
        return f"# Error generating response via Google AI: {e}"


# --- Updated Generator Function ---
def generate_initial_ml_code(business_problem, file_path, ml_plan, eda_output_logs): # Added eda_output_logs
    """Generates the initial Python ML script considering EDA logs."""
    print("Preparing prompt for initial ML code generation with EDA context...")

    if not eda_output_logs:
        print("Warning: EDA output logs are empty. ML code generation context will be limited.", file=sys.stderr)
        eda_output_logs = "# No EDA logs available."

    # Limit log length
    eda_logs_snippet = eda_output_logs[:4000] # Allow more context from logs

    try:
        prompt_text = SYSTEM_INSTRUCTION_ML_GENERATOR_TEMPLATE.format(
            business_problem_str=business_problem,
            file_path_str=file_path, # This should be the PROCESSED dataset path
            ml_plan_str=ml_plan,
            eda_output_logs_str=eda_logs_snippet + ('...' if len(eda_output_logs) > 4000 else '')
        )
    except KeyError as e:
         print(f"Error formatting ML generator prompt: Missing key {e}", file=sys.stderr)
         return f"# Error: ML generator prompt formatting failed, missing key {e}"
    except Exception as e:
         print(f"Error formatting ML generator prompt: {e}", file=sys.stderr)
         return f"# Error: ML generator prompt formatting failed: {e}"

    # Message structure for Gemini API
    initial_messages = [{"role": "user", "parts": [{"text": prompt_text}]}]
    ml_code = generate_response(initial_messages)
    print("Initial ML code generated.")
    return ml_code


# --- Updated Refinement Loop Function ---
def generate_and_refine_ml_code(business_problem, file_path, ml_plan, eda_output_logs, max_refinements=3): # Added eda_output_logs
    """Generates and refines the ML code using self-evaluation, considering EDA logs."""

    if not client_configured:
        return "# Error: Cannot generate ML code, Google AI client not configured."

    if not eda_output_logs:
        print("Warning: EDA output logs empty for refinement loop. Reflector context limited.", file=sys.stderr)
        eda_output_logs = "# No EDA logs available."
    eda_logs_snippet = eda_output_logs[:4000] # Limit log length for reflector too

    requirements_summary = f"""
    - **Goal:** Implement ML pipeline based on ML Plan, using processed data from '{file_path}' and insights from EDA Logs.
    - Load processed data from '{file_path}'. Handle FileNotFoundError.
    - Identify target based on Plan/Logs. Define X, y.
    - Split data (stratify if classification).
    - Instantiate models from Plan.
    - Perform CV on TRAIN data using metrics/strategy from Plan. **Assume data is preprocessed (no redundant scaling/encoding)**. Print CV results.
    - Train best model (from CV) on full TRAIN data. Evaluate on TEST data using Plan metrics. Print results.
    - Print feature importance if applicable.
    - Save trained model using joblib.
    - Output ONLY raw Python code (imports: pandas, numpy, sklearn, joblib), no comments, no viz libs, no markdown.
    """

    print("--- Generating Initial ML Code (Attempt 1) ---")
    current_code = generate_initial_ml_code(business_problem, file_path, ml_plan, eda_output_logs) # Pass logs here

    if current_code.startswith("# Error"):
        print(f"Initial code generation failed: {current_code}", file=sys.stderr)
        return current_code

    previous_code = current_code # Store initial code

    for i in range(max_refinements):
        print(f"\n--- ML Reflection Cycle {i+1}/{max_refinements} ---")

        try:
            reflector_prompt_content = SYSTEM_INSTRUCTION_ML_REFLECTOR_TEMPLATE.format(
                requirements_summary=requirements_summary,
                business_problem_str=business_problem, # Add business problem for context
                ml_plan_str=ml_plan,
                eda_output_logs_str=eda_logs_snippet + ('...' if len(eda_output_logs) > 4000 else ''), # Pass logs snippet
                file_path_str=file_path, # Pass processed data path
                generated_code=current_code
            )
        except KeyError as e:
             print(f"Error formatting ML reflector prompt: Missing key {e}. Using previous code.", file=sys.stderr)
             return previous_code
        except Exception as e:
            print(f"Error formatting ML reflector prompt: {e}. Using previous code.", file=sys.stderr)
            return previous_code

        print("Requesting critique...")
        # Reflector expects a string prompt, not a message list
        critique = generate_response(reflector_prompt_content)
        print(f"Critique Received:\n{critique[:500]}...")

        cleaned_critique = critique.strip()

        if cleaned_critique == "<OK>":
            print("--- Code passed reflection. Finalizing. ---")
            return current_code
        elif cleaned_critique.startswith("# Error"):
            print(f"Error during reflection phase: {cleaned_critique}. Returning previous code.", file=sys.stderr)
            return previous_code
        elif not cleaned_critique.startswith("Issues found:") and cleaned_critique != "<OK>":
            print("Warning: Reflector provided non-standard feedback. Using current code.", file=sys.stderr)
            return current_code
        else:
            print("Code needs refinement. Requesting revision...")
            previous_code = current_code # Store code before attempting revision

            try:
                # Refinement prompt also needs context, including EDA logs
                refinement_prompt_content = f"""
You are an expert AI/ML engineer and Python programmer revising code based on feedback.

**Original Goal:** Generate a Python script to train/evaluate ML models based on a plan, using processed data from '{file_path}' and considering EDA log insights '{eda_logs_snippet[:200]}...'.

**Previous Script Attempt:**
<previous_code>
{current_code}
</previous_code>

**Critique Received:**
<critique>
{critique}
</critique>

**ML Plan (for context):**
<ml_plan>
{ml_plan[:1000]}...
</ml_plan>

**Task:** Revise the *entire* Python script based *only* on the critique. Ensure the revised script still loads from '{file_path}', follows the ML plan, considers EDA insights, saves the model, and meets all formatting requirements (raw Python, specific imports, no comments, no markdown).

Output ONLY the fully revised, raw Python code.
"""
                # Refinement also takes a string prompt
                revised_code = generate_response(refinement_prompt_content)
                print("Code Revised.")

                if revised_code.startswith("# Error"):
                    print(f"Code refinement failed: {revised_code}. Returning code from *before* this failed refinement.", file=sys.stderr)
                    return previous_code
                current_code = revised_code
            except Exception as e:
                 print(f"Error during refinement call: {e}. Returning code from *before* this failed refinement.", file=sys.stderr)
                 return previous_code


    print(f"\n--- Max refinements ({max_refinements}) reached. Returning last generated code. ---")
    return current_code


def save_ml_code_to_file(code: str, file_path: str) -> bool:
    """Saves the ML script code to the specified file path."""
    if code is None or code.startswith("# Error"):
        print(f"Not saving ML code due to generation error or None value.", file=sys.stderr)
        return False

    cleaned_code = code.strip()
    if cleaned_code.startswith("```python"):
        cleaned_code = cleaned_code[len("```python"):].strip()
    if cleaned_code.endswith("```"):
        cleaned_code = cleaned_code[:-3].strip()

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(cleaned_code)
        print(f"ML script saved successfully at: {file_path}")
        return True
    except Exception as e:
        print(f"Error saving ML code to {file_path}: {e}", file=sys.stderr)
        return False