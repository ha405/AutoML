import google.generativeai as genai
import os
import sys
import re # Keep re import
import time # Keep time import

# --- Constants ---
# Assuming PROCESSED_DATASET_PATH is defined elsewhere if needed by the generated code itself
# from constants import PROCESSED_DATASET_PATH

# --- Configuration ---
# GOOGLE_API_KEY should be set as an environment variable for security
# Example placeholder provided for structure, DO NOT HARDCODE REAL KEYS
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
MODEL_NAME = "gemini-2.0-flash"

# Initialize client with proper error handling
model = None
try:
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    print(f"Error configuring Gemini API client: {e}", file=sys.stderr)


# --- Prompt Templates ---

SYSTEM_INSTRUCTION_ML_GENERATOR_TEMPLATE = r"""
You are an expert AI/ML engineer and Python programmer specializing in generating end-to-end machine learning pipelines with visualizations.
Your goal is to create a complete, clean, robust, and executable Python script based on the provided context, **paying close attention to the EDA output logs for insights about the final processed data.**

**Input Context:**

<file_path>
{file_path_str}
</file_path>

<viz_directory>
{viz_directory_str}
</viz_directory>

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

1.  **Imports:** Import necessary libraries: `os`, `pandas`, `numpy`, `sklearn.model_selection`, `sklearn.metrics`, specific models from `<ml_plan>` (or standard ones like `LogisticRegression`, `RandomForestClassifier/Regressor`), `joblib`, `matplotlib.pyplot`, `seaborn`.
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
    *   Make predictions on the **test set (X_test)** using `predict()` (or `predict_proba()` if needed later, but stick to `predict()` for metrics unless plan specifies otherwise). Store predictions in `y_pred`.
    *   Calculate and print the final evaluation metrics on the test set predictions (`y_test`, `y_pred`), using the primary and secondary metrics specified in the `<ml_plan>`.
9.  **Feature Importance / Interpretation (Optional but Recommended):**
    *   If the best model allows (e.g., RandomForest, LogisticRegression with `coef_`), calculate feature importances or coefficients. Use the column names from `X.columns`. Store them, perhaps in a Pandas Series or DataFrame, sorted descending.
10. **Save the Trained Model:**
    *   Save the **trained best model** using `joblib.dump()`. Choose a reasonable filename (e.g., `trained_model.joblib`). Print a confirmation message.
11. **Generate Explanatory Visualizations:**
    *   Create the visualization directory: `os.makedirs({viz_directory_str}, exist_ok=True)`.
    *   Generate 1-2 **simple, clean visualizations** helpful for non-technical users and save them to the `{viz_directory_str}`.
    *   **Required Visualization: Feature Importance Bar Chart:**
        *   If feature importances/coefficients were calculated in Step 9:
        *   Select the top 10-15 features.
        *   Create a horizontal bar chart (`seaborn.barplot` or `matplotlib.pyplot.barh`).
        *   Ensure **clear title** ("Top Feature Importances"), **labeled axes** (Feature Name, Importance Score), and **no overlapping/rotated labels**. Horizontal format helps with long names.
        *   Use `plt.tight_layout()` before saving.
        *   Save the plot to `os.path.join({viz_directory_str}, 'feature_importance.png')`.
        *   Close the plot using `plt.close()`.
    *   **Optional Visualization (choose one based on task type):**
        *   *If Classification:* A count plot showing the distribution of predicted classes on the test set (`seaborn.countplot(x=y_pred)`). Add title, labels, use `plt.tight_layout()`, save to `os.path.join({viz_directory_str}, 'prediction_distribution.png')`, and `plt.close()`.
        *   *If Regression:* A scatter plot of Actual vs. Predicted values on the test set (`seaborn.scatterplot(x=y_test, y=y_pred)`). Add a diagonal reference line (y=x). Add title ("Actual vs. Predicted Values"), labels ("Actual", "Predicted"), use `plt.tight_layout()`, save to `os.path.join({viz_directory_str}, 'actual_vs_predicted.png')`, and `plt.close()`.
    *   Print confirmation messages after saving plots.

**Output Format & Constraints:**
*   Output ONLY raw Python code.
*   Start directly with imports. No introductory/concluding text, no markdown fences (```python).
*   **NO COMMENTS** in the Python code, except maybe a brief one if a crucial assumption isn't covered by the plan/logs.
*   Ensure the script uses the file path `{file_path_str}` for loading the **processed** data and `{viz_directory_str}` for saving visualizations.
"""

SYSTEM_INSTRUCTION_ML_REFLECTOR_TEMPLATE = r"""
You are an expert Python code reviewer and AI/ML Quality Assurance specialist.
Your task is to meticulously review the provided Python script intended for training, evaluating machine learning models, and generating basic visualizations, ensuring it aligns with the ML Plan and considers the EDA outputs.

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

<viz_directory>
{viz_directory_str}
</viz_directory>

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
    *   Are models trained on `X_train` and evaluated on `X_test` (predictions on `X_test` vs `y_test`)?
    *   Are metrics calculated correctly?
    *   Is the visualization directory created (`os.makedirs`)?
    *   Does the feature importance logic (if applicable) correctly extract and prepare data for plotting?
3.  **Completeness:**
    *   Are necessary libraries imported (`os`, `pandas`, `numpy`, `sklearn`, `joblib`, `matplotlib.pyplot`, `seaborn`)?
    *   Are all major steps from the plan present (Load, Define X/y, Split, CV, Final Train/Eval, Save Model, Visualizations)?
    *   Are key results printed (target, chosen model, CV scores, final metrics, viz save confirmations)?
4.  **Visualization Quality & Adherence:**
    *   Is a feature importance horizontal bar chart generated (if model permits)?
    *   Is an appropriate second visualization generated (countplot for classification / scatter for regression)?
    *   Are plots saved to the correct `<viz_directory>`?
    *   Does the code include elements for **clean visualization** (clear titles, axis labels, `plt.tight_layout()`, `plt.close()`)?
    *   Are the visualizations generally suitable for a non-technical audience (simple, direct, avoiding complex plots like ROC/Confusion Matrix)?
5.  **Format Adherence:**
    *   Is the output *only* raw Python code? No markdown, no extra text?
    *   Are there **NO COMMENTS** in the generated code?

**Output:**
*   If the script correctly implements the ML plan and visualization requirements based on the context and meets all other criteria, respond ONLY with: `<OK>`. NOTHING ELSE.
*   Otherwise, provide concise, constructive feedback listing the specific issues (e.g., "Line 95: Missing creation of viz_directory.", "Line 110: Feature importance plot is missing labels.", "Line 60: Redundant StandardScaler found."). Start feedback with "Issues found:". Do NOT provide the fully corrected code.
"""


def generate_response(messages_list):
    """Sends a prompt (message list or string) to the Gemini model and returns the cleaned text response."""
    if not model:
        error_msg = "# Error: Gemini API client not properly configured. Please set GOOGLE_API_KEY environment variable."
        print(f"ERROR ({__file__}): {error_msg}", file=sys.stderr)
        return error_msg

    try:
        print(f"Sending request to {MODEL_NAME} (ML Generator/Reflector)...")
        generation_config = genai.types.GenerationConfig(
            temperature=0.1,
            max_output_tokens=8192,  # Increase max tokens
            top_p=0.8,
            top_k=40
        )

        # Handle both list of messages and single string prompt
        if isinstance(messages_list, list):
            contents = messages_list
        elif isinstance(messages_list, str):
            contents = [{"role": "user", "parts": [{"text": messages_list}]}]
        else:
            raise TypeError("generate_response expects a list of messages or a single string prompt.")

        response = model.generate_content(contents=contents, generation_config=generation_config)
        print("Response received.")

        # Check for blocked prompt or safety issues
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            error_msg = f"# Error: Prompt blocked by Google AI due to {response.prompt_feedback.block_reason}"
            print(f"❌ {error_msg}", file=sys.stderr)
            return error_msg

        # Extract text from response, handling both candidate and direct text access
        try:
            if hasattr(response, 'text'):
                text = response.text
            elif response.candidates and len(response.candidates) > 0:
                text = response.candidates[0].content.parts[0].text
            else:
                raise ValueError("No text content found in response")
                
            # Clean the response
            cleaned = text.strip()
            cleaned = re.sub(r'^```[a-zA-Z]*\n', '', cleaned)
            cleaned = re.sub(r'\n```$', '', cleaned)
            cleaned = cleaned.strip()
            
            if not cleaned:
                raise ValueError("Cleaned response text is empty")
                
            return cleaned
            
        except Exception as e:
            error_msg = f"# Error extracting/cleaning response text: {str(e)}"
            if response.candidates:
                error_msg += f"\nFinish reason: {response.candidates[0].finish_reason}"
            print(f"❌ {error_msg}", file=sys.stderr)
            return error_msg

    except Exception as e:
        error_msg = f"# Error during Google AI API call ({type(e).__name__}): {str(e)}"
        print(f"❌ {error_msg}", file=sys.stderr)
        return error_msg


# --- Updated Generator Function ---
def generate_initial_ml_code(business_problem, file_path, ml_plan, eda_output_logs, viz_directory): # Added viz_directory
    """Generates the initial Python ML script considering EDA logs and viz path."""
    print("Preparing prompt for initial ML code generation with EDA context and Viz path...")

    if not model:
         return "# Error: Gemini API client not properly configured. Please set GOOGLE_API_KEY environment variable."

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
            eda_output_logs_str=eda_logs_snippet + ('...' if len(eda_output_logs) > 4000 else ''),
            viz_directory_str=viz_directory # Add viz directory
        )
    except KeyError as e:
         print(f"Error formatting ML generator prompt: Missing key {e}", file=sys.stderr)
         return f"# Error: ML generator prompt formatting failed, missing key {e}"
    except Exception as e:
         print(f"Error formatting ML generator prompt: {e}", file=sys.stderr)
         return f"# Error: ML generator prompt formatting failed: {e}"

    # Message structure for Gemini API (list of messages)
    initial_messages = [{"role": "user", "parts": [{"text": prompt_text}]}]
    ml_code = generate_response(initial_messages)
    print("Initial ML code generated.")
    return ml_code


# --- Updated Refinement Loop Function ---
def generate_and_refine_ml_code(business_problem, file_path, ml_plan, eda_output_logs, viz_directory, max_refinements=3): # Added viz_directory
    """Generates and refines the ML code with visualizations using self-evaluation."""

    if not model:
        return "# Error: Cannot generate ML code, Google AI client not configured."

    if not eda_output_logs:
        print("Warning: EDA output logs empty for refinement loop. Reflector context limited.", file=sys.stderr)
        eda_output_logs = "# No EDA logs available."
    eda_logs_snippet = eda_output_logs[:4000] # Limit log length for reflector too

    # Removed viz_output from requirements summary as it's not used directly
    requirements_summary = f"""
    - **Goal:** Implement ML pipeline based on ML Plan, using processed data from '{file_path}', insights from EDA Logs, and save visualizations to '{viz_directory}'.
    - Load processed data from '{file_path}'. Handle FileNotFoundError.
    - Identify target based on Plan/Logs. Define X, y.
    - Split data (stratify if classification).
    - Instantiate models from Plan.
    - Perform CV on TRAIN data using metrics/strategy from Plan. **Assume data is preprocessed (no redundant scaling/encoding)**. Print CV results.
    - Train best model (from CV) on full TRAIN data. Evaluate on TEST data using Plan metrics. Print results.
    - Print feature importance if applicable.
    - Save trained model using joblib.
    - Generate 1-2 simple, non-technical visualizations (feature importance mandatory if applicable, plus one task-specific) and save them to '{viz_directory}'. Ensure plots are clean (labels, titles, no overlap, tight_layout, close).
    - Output ONLY raw Python code (imports: os, pandas, numpy, sklearn, joblib, matplotlib, seaborn), NO comments, no markdown.
    """

    print("--- Generating Initial ML Code (Attempt 1) ---")
    current_code = generate_initial_ml_code(business_problem, file_path, ml_plan, eda_output_logs, viz_directory) # Pass viz_directory

    if current_code.startswith("# Error"):
        print(f"Initial code generation failed: {current_code}", file=sys.stderr)
        return current_code

    # Keep track of the previous *valid* code attempt in case refinement fails
    last_valid_code = current_code

    for i in range(max_refinements):
        print(f"\n--- ML Reflection Cycle {i+1}/{max_refinements} ---")

        try:
            reflector_prompt_content = SYSTEM_INSTRUCTION_ML_REFLECTOR_TEMPLATE.format(
                requirements_summary=requirements_summary, # Reflector doesn't need full req summary
                business_problem_str=business_problem,
                ml_plan_str=ml_plan,
                eda_output_logs_str=eda_logs_snippet + ('...' if len(eda_output_logs) > 4000 else ''),
                file_path_str=file_path,
                viz_directory_str=viz_directory, # Pass viz dir to reflector prompt
                generated_code=current_code
            )
        except KeyError as e:
             print(f"Error formatting ML reflector prompt: Missing key {e}. Using previous valid code.", file=sys.stderr)
             return last_valid_code # Return the last known good code
        except Exception as e:
            print(f"Error formatting ML reflector prompt: {e}. Using previous valid code.", file=sys.stderr)
            return last_valid_code # Return the last known good code

        print("Requesting critique...")
        # Reflector expects a string prompt, which generate_response handles
        critique = generate_response(reflector_prompt_content)
        print(f"Critique Received:\n{critique[:500]}...") # Print start of critique

        cleaned_critique = critique.strip()

        if cleaned_critique == "<OK>":
            print("--- Code passed reflection. Finalizing. ---")
            return current_code
        elif cleaned_critique.startswith("# Error"):
            print(f"Error during reflection phase: {cleaned_critique}. Returning previous valid code.", file=sys.stderr)
            return last_valid_code # Return the last known good code
        # Check if critique looks valid before proceeding
        elif not cleaned_critique.startswith("Issues found:") and cleaned_critique != "<OK>":
            print(f"Warning: Reflector provided non-standard feedback ('{cleaned_critique[:100]}...'). Assuming code is acceptable and returning current version.", file=sys.stderr)
            return current_code # Trust the current code if feedback is weird
        else:
            print("Code needs refinement based on critique. Requesting revision...")
            # Update last valid code *before* attempting revision
            last_valid_code = current_code

            try:
                # Refinement prompt also needs context, including viz directory
                refinement_prompt_content = f"""
You are an expert AI/ML engineer and Python programmer revising code based on feedback.

**Original Goal:** Generate a Python script to train/evaluate ML models based on a plan, using processed data from '{file_path}', considering EDA log insights '{eda_logs_snippet[:200]}...', and saving visualizations to '{viz_directory}'.

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

**Task:** Revise the *entire* Python script based *only* on the critique. Ensure the revised script still loads from '{file_path}', follows the ML plan, considers EDA insights, saves the model, generates clean visualizations to '{viz_directory}', and meets all formatting requirements (raw Python, specific imports, NO COMMENTS, no markdown).

Output ONLY the fully revised, raw Python code. NO extra text or explanations.
"""
                # Refinement also takes a string prompt
                revised_code = generate_response(refinement_prompt_content)
                print("Code Revision Attempted.")

                if revised_code.startswith("# Error"):
                    print(f"Code refinement failed: {revised_code}. Returning code from *before* this failed refinement attempt.", file=sys.stderr)
                    # Return the last valid code, not the current code which failed generation
                    return last_valid_code
                # Basic check: Is the revised code substantially different? Or empty?
                if not revised_code or len(revised_code) < 50:
                     print(f"Warning: Code refinement resulted in very short or empty code. Returning code from *before* this failed refinement attempt.", file=sys.stderr)
                     return last_valid_code
                # If revision seems successful, update current_code
                current_code = revised_code
                # The loop will continue, and last_valid_code will be updated *if* this iteration passes reflection next time

            except Exception as e:
                 # Catch errors during the API call itself
                 print(f"Error during refinement API call: {e}. Returning code from *before* this failed refinement attempt.", file=sys.stderr)
                 return last_valid_code


    print(f"\n--- Max refinements ({max_refinements}) reached. Returning last generated code (may still contain issues noted in final critique). ---")
    # Return the latest code, even if the last critique wasn't <OK>
    return current_code


def save_ml_code_to_file(code: str, file_path: str) -> bool:
    """Saves the ML script code to the specified file path after cleaning."""
    if code is None or code.startswith("# Error") or not code.strip():
        print(f"Not saving ML code to {file_path} due to generation error, None value, or empty code.", file=sys.stderr)
        return False

    # Use the same cleaning logic as generate_response
    cleaned_code = code.strip()
    cleaned_code = re.sub(r'^```[a-zA-Z]*\n', '', cleaned_code) # Remove starting ``` optional_language newline
    cleaned_code = re.sub(r'\n```$', '', cleaned_code) # Remove ending newline ```
    cleaned_code = cleaned_code.strip() # Final strip

    if not cleaned_code:
        print(f"Not saving ML code to {file_path} - code became empty after cleaning.", file=sys.stderr)
        return False

    try:
        # Ensure the target directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(cleaned_code)
        print(f"ML script saved successfully at: {file_path}")
        return True
    except Exception as e:
        print(f"Error saving ML code to {file_path}: {e}", file=sys.stderr)
        return False