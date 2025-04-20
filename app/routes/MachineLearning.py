import time
import google.generativeai as genai
import os
import sys

GOOGLE_API_KEY = "AIzaSyBoAFOxBSX1nxEF8lNuhJudPiHVTCRNK8Q"  
MODEL_NAME = "gemini-2.0-flash"

client_configured = False
model = None
try:
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_AI_API_KEY":
        print("⚠ Warning: Google API Key not set or is placeholder. Please replace 'YOUR_GOOGLE_AI_API_KEY'.", file=sys.stderr)
    else:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)
        print(f"Google AI client configured for ML Generator/Reflector using model {MODEL_NAME}.")
        client_configured = True
except Exception as e:
    print(f"❌ Error configuring Google AI client: {e}", file=sys.stderr)


SYSTEM_INSTRUCTION_ML_GENERATOR_TEMPLATE = r"""
You are an expert AI/ML engineer and Python programmer specializing in generating end-to-end machine learning pipelines.
Your goal is to create a complete, clean, robust, and executable Python script based on the provided context.

Note: The code should literally start off with the import statements. Don't include any introduction like "Here is the corrected script".

Input Context:

<file_path>
{file_path_str}
</file_path>

<business_problem>
{business_problem_str}
</business_problem>

<ml_plan>
{ml_guide}
</ml_plan>

Important: The code shouldnt contain anything like "Here is corrected code". It should ONLY contain code, no comments, nothing else. just python code

Instructions for Python Script Generation:

Generate a Python script that performs the following steps IN ORDER:

1.  Imports: Import necessary libraries: pandas, numpy, sklearn.model_selection (train_test_split, KFold, cross_val_score), sklearn.preprocessing (StandardScaler, LabelEncoder - if needed), relevant sklearn.linear_model, sklearn.tree, sklearn.ensemble, sklearn.svm, sklearn.neural_network models, and sklearn.metrics. Import shap if you plan to use it.
2.  Load Data:
    *   Load the CSV file from the path specified in <file_path>. Use pd.read_csv(r"{file_path_str}").
    *   Handle potential FileNotFoundError with a clear error message and exit.
    *   Create a copy: df = df_original.copy().
3.  Initial Data Preparation (Minimal):
    *   Identify the likely target variable based on the <business_problem> (e.g., 'price', 'churn', 'sales'). Print the identified target column name. If unsure, make a reasonable guess and print it.
    *   Handle obvious non-feature columns (e.g., drop unique ID columns if present and not the target).
    *   Perform basic missing value imputation (e.g., median for numeric, mode for categorical) ONLY IF ABSOLUTELY NECESSARY before determining task type. Prefer handling missing values after train/test split if possible, using fit on train data only. For simplicity here, let's allow basic imputation before split if needed for target identification.
4.  Infer Task Type & Prepare Target:
    *   Examine the identified target column:
        *   If dtype is 'object' or has few unique numeric values (e.g., <= 10), assume Classification. If target is object/categorical, use LabelEncoder to convert it to numeric BEFORE splitting.
        *   Otherwise, assume Regression.
    *   Print the inferred task type ("Classification" or "Regression").
    *   Define features X (all columns except target) and target y. Ensure y is numeric.
5.  Train-Test Split:
    *   Split X and y into training and testing sets (e.g., 80/20 split, random_state=42).
6.  Preprocessing Pipeline (Fit on Train, Transform Both):
    *   Identify numeric and categorical features in X_train.
    *   Create a ColumnTransformer pipeline:
        *   For numeric features: Use StandardScaler(). Handle missing values within the pipeline using SimpleImputer(strategy='median') before scaling.
        *   For categorical features: Handle missing values within the pipeline using SimpleImputer(strategy='most_frequent') then apply OneHotEncoder(handle_unknown='ignore', sparse_output=False).
    *   Fit the transformer ONLY on X_train.
    *   Transform both X_train and X_test. Store the processed data (it might be numpy arrays). Get feature names after transformation if possible (using get_feature_names_out).
7.  Model Selection:
    *   Based on the inferred task type, select at least 3 appropriate models:
        *   If Classification: e.g., LogisticRegression, DecisionTreeClassifier, RandomForestClassifier.
        *   If Regression: e.g., LinearRegression, DecisionTreeRegressor, RandomForestRegressor.
    *   Instantiate the selected models (use default hyperparameters or common settings like random_state=42).
8.  Model Training and Evaluation (using Cross-Validation on Training Data):
    *   Define KFold (e.g., n_splits=5, shuffle=True, random_state=42).
    *   For each selected model:
        *   Perform cross-validation using cross_val_score on the preprocessed training data (X_train_processed, y_train).
        *   Use appropriate scoring metric(s) based on task type:
            *   If Classification: 'accuracy', 'f1_macro'
            *   If Regression: 'neg_mean_squared_error', 'r2'
        *   Calculate and print the mean and standard deviation of the cross-validation scores for each metric.
9.  Final Model Training and Test Set Evaluation:
    *   Choose the best model based on cross-validation results (e.g., highest mean F1/Accuracy or R2/lowest MSE). State which model was chosen.
    *   Train the chosen best model on the entire preprocessed training set (X_train_processed, y_train).
    *   Make predictions on the preprocessed test set (X_test_processed).
    *   Calculate and print the final evaluation metrics on the test set predictions:
        *   If Classification: accuracy_score, classification_report (includes precision, recall, f1-score).
        *   If Regression: mean_absolute_error, mean_squared_error, r2_score.
10. Feature Importance / Interpretation (Optional but Recommended):
    *   If the best model has feature_importances_ (like RandomForest), print the top 10 feature importances with their names (use names obtained from the preprocessor).
    *   Advanced (Optional): If shap is imported and the model is suitable (e.g., tree-based), calculate and print SHAP summary plot information (this might be complex to generate code for reliably, focus on feature_importances_ first).
11. The code shouldnt contain anything like "Here is corrected code". It should ONLY contain code, no comments, nothing else. just python code
12. It shouldn't contain any introductory or concluding remarks about code either as no text is needed.
13. The code should literally start off with the import statements. Don't include any introduction like "Here is the corrected script". omit any starting text please.

Output Format:
*   Your response MUST contain ONLY the raw Python code for the script.
*   Do NOT include any markdown formatting (like python ... ).
*   Do NOT include any comments in the final code, unless explicitly needed (e.g., target variable guess).
*   The script must be fully executable if the <file_path> is valid and the necessary libraries are installed.
*   Include necessary imports at the beginning.

VERY IMPORTANT: The code shouldnt contain anything like "Here is corrected code" in starting. It should ONLY contain code, no comments, nothing else. just python code
AND: It shouldn't contain any introductory or concluding remarks about code either as no text is needed. ONLY RAW PYTHON CODE

The code should literally start off with the import statements. Don't include any introduction like "Here is the corrected script"

Important: The code shouldnt contain anything like "Here is corrected code". It should ONLY contain code, no comments, nothing else. just python code
"""

SYSTEM_INSTRUCTION_ML_REFLECTOR_TEMPLATE = r"""
You are an expert Python code reviewer and AI/ML Quality Assurance specialist.
Your task is to meticulously review the provided Python script intended for building a machine learning pipeline.

Important: The code shouldnt contain anything like "Here is corrected code". It should ONLY contain code, no comments, nothing else. just python code

Context:
The script was generated based on the following requirements summarized below:
<requirements_summary>
{requirements_summary}
</requirements_summary>

Provided Script:
<script>
{generated_code}
</script>

<ml_plan>
{ml_guide}
</ml_plan>

Important: The code shouldnt contain anything like "Here is corrected code". It should ONLY contain code, no comments, nothing else. just python code

Review Criteria:

0. ONLY PYTHON CODE, NO TEXT
    * The code shouldnt contain anything like "Here is corrected code".
    * It should ONLY contain code, no comments, nothing else. just python code
1.  Correctness & Logic:
    *   Does the code run without syntax errors?
    *   Is the file loaded correctly using the specified path ({file_path_str})? Is FileNotFoundError handled?
    *   Is the task type (Classification/Regression) inferred correctly based on the likely target variable?
    *   Is the target variable prepared correctly (e.g., LabelEncoded if needed)?
    *   Is the train-test split performed correctly?
    *   Is the preprocessing pipeline (imputation, scaling, encoding) structured correctly using ColumnTransformer? Is it fitted ONLY on training data and used to transform both train and test sets?
    *   Are the selected models appropriate for the inferred task type?
    *   Is cross-validation performed correctly on the training set after preprocessing? Are appropriate scoring metrics used?
    *   Is the final model trained on the full (preprocessed) training set and evaluated on the (preprocessed) test set? Are appropriate final metrics calculated and printed?
    *   Is feature importance calculated correctly if applicable?
2.  Completeness:
    *   Does the script include all necessary imports?
    *   Are all major steps present (Load, Prep, Infer Task, Split, Preprocess, Select Models, CV Eval, Final Eval, Importance)?
    *   Are all specified print statements included (e.g., shape, target, task type, CV scores, final metrics)?
3.  Adherence:
    *   Does the script strictly follow the output format (only raw Python code)?
    *   Are there any unnecessary comments or markdown?
    *   Are only the allowed libraries imported?
4.  Robustness:
    *   Is the logic sound (e.g., avoiding data leakage between train/test sets during preprocessing)?
5.  Others:
    *   The code doesn't need to cater to error handling or data cleaning tasks.
6. The code shouldnt contain anything like "Here is corrected code". It should ONLY contain code, no comments, nothing else. just python code
7. It shouldn't contain any introductory or concluding remarks about code either as no text is needed.
8. Don't be critical for no accurate reason. If it does everything correctly, response ONLY with <OK>, NOTHING ELSE

Note: The code should literally start off with the import statements. Don't include any introduction like "Here is the corrected script".

Important: The code shouldnt contain anything like "Here is corrected code". It should ONLY contain code, no comments, nothing else. just python code
Important: Do NOT include any markdown formatting (like ```python ... ```).

Output:
*   If the script meets all criteria, appears logically sound, and is likely to run correctly, respond ONLY with: <OK>. NOTHING ELSE
Note: The code should literally start off with the import statements. Don't include any introduction like "Here is the corrected script".
*   Otherwise, provide concise, constructive feedback listing the specific issues found and suggest exact corrections needed. Be specific (e.g., "Line 45: Preprocessing pipeline should be fitted only on X_train, not the whole X."). Do NOT provide the fully corrected code, only the feedback/corrections list. Start feedback with "Issues found:".
"""

# --- Modified generate_response Function ---
def generate_response(messages_list):
    """Sends a prompt (message list) to the Gemini model and returns the cleaned text response."""
    if not client_configured or model is None:
        if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_AI_API_KEY":
            return "# Error: Google AI API Key not configured. Please set the GOOGLE_API_KEY variable."
        return "# Error: Google AI client not configured properly."

    try:
        print(f"Sending request to {MODEL_NAME}...")

        # Configure generation parameters
        generation_config = genai.types.GenerationConfig(
            temperature=0.1  # Lower temperature for more deterministic code
        )

        # Use the model to generate content based on the message history
        response = model.generate_content(
            contents=messages_list,
            generation_config=generation_config
        )

        print("Response received.")

        # Simple error check based on prompt feedback
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            error_msg = f"# Error: Prompt blocked by Google AI due to {response.prompt_feedback.block_reason}"
            print(f"❌ {error_msg}", file=sys.stderr)
            return error_msg

        # Extract text, handling potential errors or empty responses
        try:
            text = response.text
        except ValueError:
            error_msg = "# Error: No content generated by the model or response was blocked."
            print(f"❌ {error_msg} (Candidates: {response.candidates})", file=sys.stderr)
            if response.candidates and response.candidates[0].finish_reason != 'STOP':
                error_msg += f" Finish Reason: {response.candidates[0].finish_reason}"
            return error_msg
        except AttributeError:
            error_msg = "# Error: Unexpected response format from Google AI."
            print(f"❌ {error_msg} (Response object: {response})", file=sys.stderr)
            return error_msg
        except Exception as e:
            error_msg = f"# Error extracting text from response: {e}"
            print(f"❌ {error_msg} (Response object: {response})", file=sys.stderr)
            return error_msg

        # Clean markdown
        text = text.strip()
        if text.startswith("python"):
            text = text[len("python"):].strip()
        if text.startswith(""):
            text = text[len(""):].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
        return text

    except Exception as e:
        print(f"❌ An error occurred during Google AI API call: {e}", file=sys.stderr)
        return f"# Error generating response via Google AI: {e}"

# --- Updated generate_initial_ml_code ---
def generate_initial_ml_code(business_problem, file_path, ML_PLAN):
    """Generates the initial Python ML script."""
    print("Preparing prompt for initial ML code generation...")
    # Adjusted message structure: Combine instructions into a single "user" message
    initial_messages = [
        {
            "role": "user",
            "parts": [{
                "text": SYSTEM_INSTRUCTION_ML_GENERATOR_TEMPLATE.format(
                    business_problem_str=business_problem,
                    file_path_str=file_path,
                    ml_guide=ML_PLAN
                ) + "\n\nPlease generate the Python script now."
            }]
        }
    ]
    ml_code = generate_response(initial_messages)
    print("Initial ML code generated.")
    return ml_code


def generate_and_refine_ml_code(business_problem, file_path, ML_PLAN, max_refinements=3):

    if not client_configured:
        return "# Error: Cannot generate ML code, Google AI client not configured."

    # Generate concise requirements summary for the reflector
    requirements_summary = f"""
    - Load CSV from '{file_path}', handle FileNotFoundError.
    - Do NOT include any markdown formatting (like ```python ... ```).
    - Basic Prep: Identify target (print it), handle obvious non-features, minimal imputation if needed for target ID.
    - Infer Task Type (Classification/Regression) based on target, print type. Prep target (LabelEncode if needed). Define X, y.
    - Split data (80/20, random_state=42).
    - Preprocessing Pipeline (ColumnTransformer): Fit on train ONLY, transform train/test. Numeric: Impute(median)+Scale. Categorical: Impute(mode)+OneHot.
    - Select >= 3 appropriate models based on task type.
    - CV (KFold=5) on preprocessed TRAIN data. Use appropriate scoring (Class: acc, f1; Reg: neg_mse, r2). Print mean/std scores.
    - Train best model (from CV) on full preprocessed TRAIN data. Evaluate on preprocessed TEST data. Print final metrics (Class: acc, classification_report; Reg: mae, mse, r2).
    - Feature Importance: Print top 10 for tree models if applicable.
    - Output ONLY raw Python code (imports: pandas, numpy, sklearn), no comments (unless needed), no viz libs.
    - Do NOT include any markdown formatting (like ```python ... ```).
    """

    print("--- Generating Initial ML Code (Attempt 1) ---")
    # Use the existing function to get initial code
    current_code = generate_initial_ml_code(business_problem, file_path, ML_PLAN)

    if current_code.startswith("# Error"):
        print(f"Initial code generation failed: {current_code}")
        return current_code # Return the error message

    for i in range(max_refinements):
        print(f"\n--- Reflection Cycle {i+1}/{max_refinements} ---")

        reflector_prompt_content = SYSTEM_INSTRUCTION_ML_REFLECTOR_TEMPLATE.format(
            requirements_summary=requirements_summary,
            generated_code=current_code,
            ml_guide=ML_PLAN,
            file_path_str=file_path 
        )

        print("Requesting critique...")
        critique = generate_response(reflector_prompt_content)
        print(f"Critique Received:\n{critique[:500]}...") # Print start of critique

        # 2. Analyze Critique
        cleaned_critique = critique.strip()

        if cleaned_critique == "<OK>":
            print("--- Code passed reflection. Finalizing. ---")
            # Return the code that was deemed OK
            return current_code
        elif cleaned_critique.startswith("# Error"):
            print(f"Error during reflection phase: {cleaned_critique}. Returning code from previous step.")
            # Return the code before this failed reflection attempt
            return current_code
        elif not cleaned_critique.startswith("Issues found:") and cleaned_critique != "<OK>":
            print("Warning: Reflector did not provide standard feedback ('<OK>' or 'Issues found:'). Using current code.")
            return current_code # Return the current code as a fallback
        else:
            # 3. Refine Code if Issues Found
            print("Code needs refinement. Requesting revision...")

            # Prepare the refinement prompt, giving the previous code and the critique
            refinement_prompt_content = f"""
You are an expert AI/ML engineer and Python programmer.
Your goal is to revise Python scripts based on reviewer feedback.

You previously generated the following script:
<previous_code>
{current_code}
</previous_code>

A code reviewer provided the following critique:
<critique>
{critique}
</critique>

Please revise the entire Python script based only on the critique provided above.
Ensure the revised script still adheres to all original requirements, including:
- Loading data from '{file_path}'.
- Addressing the business problem: '{business_problem}'.
- Following the ML plan details.
- Outputting ONLY the raw Python code (no markdown, no explanations, no comments unless necessary).
- Starting directly with imports.

Output ONLY the fully revised, raw Python code.
"""
            # Call generate_response with the refinement prompt string
            revised_code = generate_response(refinement_prompt_content)
            print("Code Revised.")
            # print(f"Revised Code Snippet:\n{revised_code[:500]}...") # Optional: Print snippet

            if revised_code.startswith("# Error"):
                print(f"Code refinement failed: {revised_code}")
                print("Returning code from before this failed refinement attempt.")
                # Return the code that existed before this failed refinement
                return current_code

            # Update current_code for the next potential cycle or final return
            current_code = revised_code

    print(f"\n--- Max refinements ({max_refinements}) reached. Returning last generated code. ---")
    return current_code

import re
def save_ml_code_to_file(code: str, file_path: str, filename: str = "ML.py") -> bool:
    """
    Saves the provided ML script code to a Python file after stripping any markdown code fences.

    Args:
        code (str): The ML script code, possibly containing markdown-style fences.
        file_path (str): The directory where the file should be saved.
        filename (str): The filename to use for saving (default is 'ML.py').

    Returns:
        bool: True if saved successfully.
    """
    # Remove markdown-style fences like ``` or ```python
    cleaned_code = re.sub(r"^```[\w+\-]*\s*", "", code.strip(), flags=re.IGNORECASE)
    cleaned_code = re.sub(r"\n?```$", "", cleaned_code.strip())

    os.makedirs(file_path, exist_ok=True)
    full_path = os.path.join(file_path, filename)

    with open(full_path, "w", encoding="utf-8") as f:
        f.write(cleaned_code)

    print(f"ML script saved successfully at: {full_path}")
    return True
