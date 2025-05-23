import google.generativeai as genai
import os
from constants import PROCESSED_DATASET_PATH


genai.configure(api_key="AIzaSyBoAFOxBSX1nxEF8lNuhJudPiHVTCRNK8Q")
client = genai.GenerativeModel("gemini-2.0-flash") # Renamed 'model' to 'client'

# Prompt Template for EDA Code Generation
# Prompt Template for EDA Code Generation
SYSTEM_INSTRUCTION_EDA_TEMPLATE = r"""
You are an expert Python data scientist specializing in automated Exploratory Data Analysis (EDA) and preprocessing for machine learning tasks.
Your goal is to generate a complete, clean, and fully executable Python script based on the provided file details and business problem.

**Input Context:**

<file_details>
{file_details_str}
</file_details>

<business_problem>
{business_problem_str}
</business_problem>

<target_file_path>
{file_path_str}
</target_file_path>

<processed_file_path>
{processed_file_path_str}  
</processed_file_path>

**Instructions for Python Script Generation:**

Generate a Python script that performs the following steps in order:

1.  **Imports:** Import necessary libraries: `pandas`, `numpy`, `os`. Also import needed classes from `sklearn.impute` and `sklearn.preprocessing`. Do NOT import visualization libraries like matplotlib or seaborn. # Added 'os' import requirement
2.  **Load Data:**
    *   Load the CSV file from the path specified in `<target_file_path>`.
3.  **Initial Inspection (Print Outputs):**
    *   Print the DataFrame shape: `print(f"DataFrame Shape: {{df.shape}}")`
    *   Print the column names: `print(f"Columns: {{df.columns.tolist()}}")`
    *   Print data types: `print("Data Types:\n", df.dtypes)`
    *   Print missing value counts per column: `print("Missing Values:\n", df.isnull().sum())`
    *   Print the number of duplicate rows: `print(f"Duplicate Rows: {{df.duplicated().sum()}}")`
4.  **Preprocessing - Missing Values:**
    *   Identify numeric columns (select_dtypes include 'number').
    *   For each numeric column: if missing values count is > 0 and < 30% of total rows, impute with the median. If >= 30%, drop the column and print a message.
    *   Identify categorical columns (select_dtypes include 'object', 'category'). Impute missing values using the mode (use SimpleImputer(strategy='most_frequent')).
    *   Print missing value counts again after handling: `print("Missing Values After Handling:\n", df.isnull().sum())`
5.  **Preprocessing - Duplicates:**
    *   Drop duplicate rows *in place* (`df.drop_duplicates(inplace=True)`). Print the shape after dropping duplicates: `print(f"Shape after dropping duplicates: {{df.shape}}")`
6.  **Preprocessing - Column Handling (Based on File Details & Common Practices):**
    *   Create a list of columns to drop.
    *   Add columns likely to be identifiers (e.g., containing 'id' or 'num' in the name, case-insensitive) to the drop list, UNLESS the column name suggests it might be a meaningful feature (e.g., 'customer_id'). Use judgment based on column names in `<file_details>`. Check if column exists before adding.
    *   Add columns with more than 90% unique values relative to the number of rows to the drop list, UNLESS it's explicitly relevant to the `<business_problem>` or a known high-cardinality feature that should be kept. Check if column exists.
    *   Drop the collected columns from the DataFrame (`df.drop(columns=columns_to_drop, inplace=True, errors='ignore')`). Print dropped columns.
7.  **Preprocessing - Encoding Categorical Features:**
    *   Identify remaining categorical columns.
    *   For each categorical column:
    *   If it has exactly 2 unique values (excluding NaN if handled), use `LabelEncoder`.
    *   If it seems ordinal (like 'low', 'medium', 'high'), attempt `OrdinalEncoder`. *State any assumptions about order in a comment*. If order is unknown, treat as nominal.
    *   If nominal and has <= 10 unique categories, use `OneHotEncoder` (set `sparse_output=False`, `handle_unknown='ignore'`, `drop='first'`). Create new columns and drop the original.
    *   If nominal and has > 10 unique categories, add the column to a list of columns to drop later.
    *   Drop any high-cardinality nominal columns identified in the previous step.
8.  **Preprocessing - Feature Engineering (Business Problem Driven):**
    *   Review the `<business_problem>`.
    *   If the problem *clearly* suggests a potentially useful interaction or ratio between *existing numerical columns*, create 1 (max 2) new features.
    *   Example: If predicting churn and columns 'TotalCharges' and 'MonthlyCharges' exist, you *might* create 'ChargeRatio'.
    *   Implement calculations safely (e.g., handle potential division by zero using `np.where(df['denominator_col'] != 0, df['numerator_col'] / df['denominator_col'], 0)` or adding a small epsilon). Check columns exist first.
    *   If no obvious, directly relevant feature engineering is suggested, skip this step.
9.  **Final Checks (Print Outputs):**
    *   Print the first 5 rows: `print("Processed DataFrame Head:\n", df.head())`
    *   Print the final shape: `print(f"Final Processed Shape: {{df.shape}}")`
    *   Print final data types: `print("Final Data Types:\n", df.dtypes)` Check for remaining 'object' types.
    *   Print summary statistics: `print("Summary Statistics (Processed Data):\n", df.describe(include='all'))`

10. **Save Processed Data:** # <-- MODIFIED INSTRUCTION
    *   Define the processed file path variable within the script: `processed_file_path = r'{processed_file_path_str}'`
    *   Ensure the directory for the processed file exists: `os.makedirs(os.path.dirname(processed_file_path), exist_ok=True)`.
    *   Save the final processed DataFrame `df` to a new CSV file at the path `processed_file_path`.
    *   Use `df.to_csv(processed_file_path, index=False)`.
    *   Print a confirmation message: `print(f"Processed data saved to: {{processed_file_path}}")`

**Output Format:**
*   Your response MUST contain ONLY the raw Python code for the script.
*   Do NOT include any import statements for visualization libraries (matplotlib, seaborn).
*   Do NOT include any markdown formatting (like ```python ... ```).
*   Do NOT include any comments in the final code, unless explicitly asked for (like for OrdinalEncoder assumptions).
*   The script must be fully executable if the paths are valid and libraries are installed.
"""

    # ---------------------------------------------------------- #

def generate_response(prompt_text):
    """
    Sends a prompt to the Gemini model and returns the cleaned text response.
    """
    try:
        print("Sending request to Gemini...")
        response = client.generate_content(contents=prompt_text) # Pass the text directly
        # Accessing the text part safely
        if response.parts:
             text = response.parts[0].text
        else:
            # Handle cases where the response might be blocked or empty
            print("Warning: Gemini response might be empty or blocked.")
            # Check candidate feedback if available
            if response.candidates and response.candidates[0].finish_reason != 'STOP':
                 print(f"Finish Reason: {response.candidates[0].finish_reason}")
                 if response.candidates[0].safety_ratings:
                     print(f"Safety Ratings: {response.candidates[0].safety_ratings}")
            text = "" # Return empty string or handle as needed

        print("Response received from Gemini.")
        # Remove any markdown formatting (e.g., ```python and ``` markers)
        text = text.strip()
        if text.startswith("```python"):
            text = text[len("```python"):].strip()
        if text.startswith("```"): # Handle cases with just ```
             text = text[len("```"):].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
        return text
    except Exception as e:
        print(f"An error occurred during Gemini API call: {e}")
        # Consider how to handle API errors, maybe return None or raise exception
        return f"# Error generating response: {e}"
    

    # ---------------------------------------------------------- #

# def generate_data_analysis_code(filedetails, business_problem, file_path):
#     """
#     Generates the initial Python EDA script using file details, business problem, and file path.
#     """
#     if 'error' in filedetails:
#         print(f"Cannot generate code due to file preprocessing error: {filedetails['error']}")
#         return "# Error in file preprocessing, cannot generate code."

#     print("Preparing prompt for initial code generation...")
#     # Convert filedetails dict to a string representation for the prompt
#     # Limit sample data length to avoid overly long prompts
#     limited_sample_data = filedetails.get("sample_data", [])[:3] # Show only 3 sample rows
#     details_copy = filedetails.copy()
#     details_copy["sample_data"] = limited_sample_data
#     file_details_str = "\n".join([f"- {key}: {value}" for key, value in details_copy.items()])

#     # Format the template with the dynamic context
#     prompt = SYSTEM_INSTRUCTION_EDA_TEMPLATE.format(
#         file_details_str=file_details_str,
#         business_problem_str=business_problem,
#         file_path_str=file_path # Pass the actual file path here
#     )

#     # print(f"Prompt Sent:\n{prompt[:500]}...") # Debug: Print start of prompt

#     eda_code = generate_response(prompt)
#     print("Initial EDA code generated.")
#     return eda_code

#     # ---------------------------------------------------------- #

# Reflector Prompt Template
SYSTEM_INSTRUCTION_REFLECTOR = r"""
You are an expert Python code reviewer and Quality Assurance specialist.
Your task is to review the provided Python script intended for EDA and preprocessing.

**Context:**
The script was generated based on the following requirements:
<requirements>
{requirements_summary}
</requirements>
(*Note: requirements_summary should be a concise version of the main points from SYSTEM_INSTRUCTION_EDA_TEMPLATE*)

**Provided Script:**
<script>
{generated_code}
</script>

**Review Criteria:**
1.  **Correctness:** Does the code run without syntax errors? Is the logic correct according to the requirements (e.g., correct imputation strategy, correct encoding, safe feature engineering)? Does it handle potential errors like FileNotFoundError during loading?
2.  **Completeness:** Does the script perform all the requested steps (loading, inspection, handling missing values, duplicates, column drops, encoding, feature eng [if applicable], final checks)? Are all specified print statements included?
3.  **Adherence:** Does the script strictly follow the output format (only Python code, no forbidden libraries, no comments unless specified)? Does it use the correct file path provided in requirements?
4.  **Robustness:** Does it handle potential edge cases mentioned (e.g., division by zero)? Does it correctly identify numeric/categorical columns?

**Output:**
*   If the script meets all criteria and will run correctly producing accurate code, respond ONLY with: `<OK>`. NOTHING ELSE.
*   Otherwise, provide concise, constructive feedback listing the specific issues found and suggest exact corrections needed. Be specific (e.g., "Line 25: Imputation logic for numeric column 'X' is incorrect, should be median not mean"). Do NOT provide the fully corrected code, only the feedback/corrections list. Start feedback with "Issues found:".
"""

    # ---------------------------------------------------------- #

# # --- Function to run the refinement loop ---
# def generate_and_refine_eda_code(filedetails, business_problem, file_path, max_refinements=10):
#     """
#     Generates and refines the EDA code using a self-evaluation loop.
#     """
#     if 'error' in filedetails:
#        return f"# Cannot generate code due to file preprocessing error: {filedetails['error']}"

#     # Generate concise requirements summary for the reflector
#     requirements_summary = """
#     - Load CSV from path '{file_path}'. Handle FileNotFoundError.
#     - Print shape, columns, dtypes, missing counts, duplicates.
#     - Handle missing: numeric(<30%): median, else drop; categorical: mode. Print counts after.
#     - Drop duplicates inplace. Print shape after.
#     - Drop ID-like/high-cardinality (>90% unique) columns (use judgment, check existence). Print dropped columns.
#     - Encode categoricals: Binary(2 unique): LabelEncode; Ordinal: OrdinalEncode (comment assumptions); Nominal(<=10 unique): OneHot; Nominal(>10 unique): Drop. Drop original cols after encoding.
#     - Optional: 1-2 relevant feature engineering steps if clearly suggested by business problem (handle division by zero).
#     - Print final checks: head(5), shape, dtypes, describe(include='all'). Check for remaining 'object' types.
#     - Output ONLY raw Python code (imports: pandas, numpy, sklearn), no comments (unless OrdinalEncoder assumption), no viz libs.
#     """.format(file_path=file_path) # Include the dynamic path here too!

#     print("--- Generating Initial Code (Attempt 1) ---")
#     current_code = generate_data_analysis_code(filedetails, business_problem, file_path)
#     # print(f"Initial Code:\n```python\n{current_code}\n```") # Optional debug

#     if current_code.startswith("# Error"): # Handle generation error
#         print(f"Initial code generation failed: {current_code}")
#         return current_code

#     for i in range(max_refinements):
#         print(f"\n--- Reflection Cycle {i+1}/{max_refinements} ---")

#         # Prepare reflector prompt
#         reflector_prompt = SYSTEM_INSTRUCTION_REFLECTOR.format(
#             requirements_summary=requirements_summary,
#             generated_code=current_code
#         )

#         print("Requesting critique...")
#         # print(f"Reflector Prompt Sent:\n{reflector_prompt[:500]}...") # Debug
#         critique = generate_response(reflector_prompt)
#         print(f"Critique Received:\n{critique}") # Print full critique

#         if critique.strip() == "<OK>":
#             print("--- Code passed reflection. Finalizing. ---")
#             return current_code
#         elif not critique.strip().startswith("Issues found:"):
#              print("Warning: Reflector did not provide specific issues or <OK>. Using current code.")
#              # Decide how to handle non-compliant feedback (e.g., return current code, try again, raise error)
#              return current_code # For now, just return the current code
#         else:
#             print("Code needs refinement. Requesting revision...")
#             # Prepare refinement prompt (using the original data scientist persona)
#             refinement_prompt = f"""
# You are an expert Python data scientist.
# Your goal is to generate correct EDA and preprocessing scripts.

# You previously generated the following script based on initial requirements:
# <previous_code>
# {current_code}
# </previous_code>

# A code reviewer provided the following critique:
# <critique>
# {critique}
# </critique>

# Please revise the *entire* Python script based *only* on the critique provided above. Ensure the revised script still adheres to *all* original requirements, including output format (raw Python code only, specific imports, no comments unless specified, no viz libs).
# Output ONLY the fully revised, raw Python code. Do not include explanations or apologies.
# """
#             # print(f"Refinement Prompt Sent:\n{refinement_prompt[:500]}...") # Debug
#             current_code = generate_response(refinement_prompt)
#             print("Code Revised.")
#             # print(f"Revised Code:\n```python\n{current_code}\n```") # Optional debug
#             if current_code.startswith("# Error"): # Handle refinement error
#                 print(f"Code refinement failed: {current_code}")
#                 print("Returning code from before refinement attempt.")
#                 # Need to decide how to handle this - return previous version?
#                 # For safety, maybe return the code *before* this failed refinement attempt
#                 # This requires storing the previous version before calling generate_response for refinement
#                 # Let's keep it simple for now and return the error message/string
#                 return current_code


#     print(f"\n--- Max refinements ({max_refinements}) reached. Returning last generated code. ---")
#     return current_code

#    # ------------------------------------#

# def save_code_to_file(code, filename="Eda_code.py"):
    # current_script_path = os.path.abspath(__file__)
    # current_dir = os.path.dirname(current_script_path)
    # app_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    # save_dir = os.path.join(app_dir, 'scripts')
    # file_path = os.path.join(save_dir, filename)
    # if code is None:
    #     return False
    # os.makedirs(save_dir, exist_ok=True)
    # with open(file_path, "w", encoding='utf-8') as f:
    #     f.write(code)
    # return True




def generate_data_analysis_code(filedetails, business_problem, file_path, processed_file_path=PROCESSED_DATASET_PATH): # <-- Added processed_file_path arg
    """
    Generates the initial Python EDA script using file details, business problem,
    input file path, and specifying the output processed file path.
    """
    if 'error' in filedetails:
        print(f"Cannot generate code due to file preprocessing error: {filedetails['error']}")
        return "# Error in file preprocessing, cannot generate code."

    print("Preparing prompt for initial EDA code generation...")
    # Convert filedetails dict to a string representation for the prompt
    limited_sample_data = filedetails.get("sample_data", [])[:3]
    details_copy = filedetails.copy()
    details_copy["sample_data"] = limited_sample_data
    # Ensure filedetails are reasonably formatted for the prompt
    file_details_str = "\n".join([f"- {key}: {str(value)[:200]}" for key, value in details_copy.items()]) # Limit value length

    # Format the template with ALL dynamic context variables
    try:
        prompt = SYSTEM_INSTRUCTION_EDA_TEMPLATE.format(
            file_details_str=file_details_str,
            business_problem_str=str(business_problem), # Ensure strings
            file_path_str=str(file_path),
            processed_file_path_str=str(processed_file_path) # <-- ADDED to format call
        )
    except KeyError as e:
         print(f"Error formatting prompt template: Missing key {e}")
         return f"# Error: Prompt template formatting failed, missing key {e}"
    except Exception as e:
         print(f"Error formatting prompt template: {e}")
         return f"# Error: Prompt template formatting failed: {e}"


    # print(f"Prompt Sent:\n{prompt[:500]}...") # Debug: Print start of prompt

    eda_code = generate_response(prompt)
    print("Initial EDA code generated.")
    return eda_code

# ... (rest of the file including reflector template and refinement loop) ...

# --- Modify the refinement loop to handle the processed_file_path ---

def generate_and_refine_eda_code(filedetails, business_problem, file_path, processed_file_path=PROCESSED_DATASET_PATH, max_refinements=3): # <-- Added processed_file_path
    """
    Generates and refines the EDA code using a self-evaluation loop.
    """
    if 'error' in filedetails:
       return f"# Cannot generate code due to file preprocessing error: {filedetails['error']}"

    # Generate concise requirements summary for the reflector, including the processed path
    requirements_summary = f"""
    - Load CSV from path '{file_path}'. Handle FileNotFoundError.
    - Print shape, columns, dtypes, missing counts, duplicates.
    - Handle missing: numeric(<30%): median, else drop; categorical: mode. Print counts after.
    - Drop duplicates inplace. Print shape after.
    - Drop ID-like/high-cardinality (>90% unique) columns (use judgment, check existence). Print dropped columns.
    - Encode categoricals: Binary(2 unique): LabelEncode; Ordinal: OrdinalEncode (comment assumptions); Nominal(<=10 unique): OneHot; Nominal(>10 unique): Drop. Drop original cols after encoding.
    - Optional: 1-2 relevant feature engineering steps if clearly suggested by business problem (handle division by zero).
    - Print final checks: head(5), shape, dtypes, describe(include='all'). Check for remaining 'object' types.
    - Save processed data to '{processed_file_path}', ensure directory exists. Print confirmation. # <-- Added requirement
    - Output ONLY raw Python code (imports: pandas, numpy, os, sklearn), no comments (unless OrdinalEncoder assumption), no viz libs.
    """
    # Note: .format() is not needed here as paths are directly in the f-string

    print("--- Generating Initial Code (Attempt 1) ---")
    # Pass processed_file_path to the generator function
    current_code = generate_data_analysis_code(filedetails, business_problem, file_path, processed_file_path)

    if current_code.startswith("# Error"):
        print(f"Initial code generation failed: {current_code}")
        return current_code

    for i in range(max_refinements):
        print(f"\n--- Reflection Cycle {i+1}/{max_refinements} ---")

        # Prepare reflector prompt, passing the processed path for context if needed (though summary includes it)
        reflector_prompt = SYSTEM_INSTRUCTION_REFLECTOR.format(
            requirements_summary=requirements_summary,
            generated_code=current_code
            # Add processed_file_path_str=processed_file_path if the reflector template needs it explicitly
        )

        print("Requesting critique...")
        critique = generate_response(reflector_prompt)
        print(f"Critique Received:\n{critique[:500]}...") # Print start of critique

        cleaned_critique = critique.strip()

        if cleaned_critique == "<OK>":
            print("--- Code passed reflection. Finalizing. ---")
            return current_code
        elif cleaned_critique.startswith("# Error"):
             print(f"Error during reflection phase: {cleaned_critique}. Returning current code.")
             return current_code # Return code before failed reflection
        elif not cleaned_critique.startswith("Issues found:") and cleaned_critique != "<OK>":
             print("Warning: Reflector did not provide standard feedback ('<OK>' or 'Issues found:'). Using current code.")
             return current_code # Fallback
        else:
            print("Code needs refinement. Requesting revision...")
            # Prepare refinement prompt including the processed path for context
            refinement_prompt = f"""
You are an expert Python data scientist.
Your goal is to generate correct EDA and preprocessing scripts.

You previously generated the following script based on initial requirements (loading from '{file_path}', saving processed data to '{processed_file_path}'):
<previous_code>
{current_code}
</previous_code>

A code reviewer provided the following critique:
<critique>
{critique}
</critique>

Please revise the *entire* Python script based *only* on the critique provided above.
Ensure the revised script still adheres to *all* original requirements, including:
- Loading data from '{file_path}'.
- Saving processed data to '{processed_file_path}'.
- Addressing the business problem: '{business_problem}'.
- Outputting ONLY the raw Python code (no markdown, no explanations, no comments unless specified).
- Starting directly with imports (including os).

Output ONLY the fully revised, raw Python code.
"""
            current_code = generate_response(refinement_prompt)
            print("Code Revised.")
            if current_code.startswith("# Error"):
                print(f"Code refinement failed: {current_code}")
                print("Returning code from before this failed refinement attempt.")
                # Need to retrieve the code *before* this failed attempt.
                # For simplicity now, we might just return the error string.
                # To return previous code, you'd need to store it before calling generate_response.
                return current_code # Return the error string for now


    print(f"\n--- Max refinements ({max_refinements}) reached. Returning last generated code. ---")
    return current_code


# --- Modify save_code_to_file ---
# It seems you want to save to a specific path passed from main.py (EDA_CODE_FILE_PATH)
# So the filename argument might be confusing. Let's make it accept the full path directly.

def save_code_to_file(code, file_path): # Changed 'filename' to 'file_path'
    """Saves the provided code string to the specified absolute file path."""
    if code is None or code.startswith("# Error"):
        print(f"Not saving code due to generation error or None value.", file=sys.stderr)
        return False
    try:
        # Directory creation should happen in main.py before calling this,
        # or the generated script should handle it before saving data.
        # For saving the script itself, main.py handles the directory.
        print(f"Attempting to save code to: {file_path}")
        with open(file_path, "w", encoding='utf-8') as f:
            f.write(code)
        print(f"Code saved successfully to: {file_path}")
        return True
    except Exception as e:
        print(f"Error saving code to {file_path}: {e}", file=sys.stderr)
        return False