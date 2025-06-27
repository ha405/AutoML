import os
import sys
import google.generativeai as genai
from constants import PROCESSED_DATASET_PATH

# --- Configuration ---
try:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    genai.configure(api_key="AIzaSyBF8Ik7v2Uwy_cRVzoDEj30g2oNpXPPlrQ")
    client = genai.GenerativeModel("gemini-2.0-flash")
except Exception as e:
    print(f"[ERROR] Failed to configure Gemini API: {str(e)}")
    sys.exit(1)

# --- Prompt Templates ---
SYSTEM_INSTRUCTION_EDA = r"""
You are an expert Python data scientist focused on automated, standards‑driven Exploratory Data Analysis (EDA),
preprocessing for machine learning pipelines, and non‑technical, user‑friendly visualizations.

Your objective is to generate a single, self‑contained Python script that executes the following steps *strictly as outlined* by the provided EDA guidance plan without deviation:

1. **Imports**:
   - Only import `pandas`, `numpy`, `os`, and classes from `sklearn.impute`, `sklearn.preprocessing`.
   - For visualizations, import `matplotlib.pyplot as plt`.

2. **Data Loading**:
   - Read CSV from `<target_file_path>`. Handle `FileNotFoundError` by printing an error message and exiting.

3. **Initial Diagnostics**:
   - Print DataFrame shape, column list with dtypes.
   - Print count of missing values per column and number of duplicate rows.

4. **Guided Preprocessing** (follow `<eda_guidance_plan>` exactly):
   - **Target Variable**: Convert or balance as directed.
   - **Feature Selection/Dropping**: Drop or retain columns per plan.
   - **Missing Value Strategy**: Impute using methods (mean/median/mode/constant) specified.
   - **Duplicate Removal**: Drop duplicates.
   - **Encoding**: Apply LabelEncoder, OneHotEncoder, or OrdinalEncoder as recommended, respecting cardinality thresholds.
   - **Feature Engineering**: Implement only the transformations or new features explicitly listed.
   - **Scaling**: Scale numerical features only if the plan prescribes, using StandardScaler or MinMaxScaler accordingly.

5. **Visualizations**:
   - Create **3–5** non‑technical, easy‑to‑understand charts using matplotlib.
   - If plotting a large number of features, automatically select the top K (e.g. top 10) by importance or variance.
   - Ensure **all** x‑axis labels remain horizontal (no rotation) and fully visible.
   - Use generous margins, clear titles, and legible fonts for a clean look.
   - Save each figure as a PNG into the provided `<viz_directory>`, using descriptive names (e.g. `missing_values.png`, `top_features_distribution.png`).

6. **Final Diagnostics**:
   - Print transformed DataFrame head, new shape, dtypes, and summary statistics.
   - Verify no unexpected object‑type columns remain.

7. **Save Output**:
   - Ensure directory for `<processed_file_path>` exists.
   - Save processed DataFrame to `<processed_file_path>` without an index.
   - Print success message with the file path.

**Format Requirements**:
- Output only the raw Python code; no markdown or comments except minimal inline notes justifying assumptions dictated by the guidance.
- Do not import or use any other visualization libraries.
- All plots must adhere to the neat, non‑technical style guidelines above.

<file_details>
{file_details_str}
</file_details>
<business_problem>
{business_problem_str}
</business_problem>
<eda_guidance_plan>
{eda_guidance_str}
</eda_guidance_plan>
<target_file_path>
{file_path_str}
</target_file_path>
<processed_file_path>
{processed_file_path_str}
</processed_file_path>
<viz_directory>
{viz_directory_str}
</viz_directory>
"""

SYSTEM_INSTRUCTION_REFLECTOR = r"""
You are a Python QA specialist verifying that a generated EDA script exactly implements the instructions in the provided EDA guidance plan.

Check the script for:
- Strict adherence to every preprocessing step in `<eda_guidance_plan>`.
- Absence of unauthorized imports (no visualization libs beyond matplotlib).
- Correct error handling for file operations.
- Correct use of imputation, encoding, scaling as specified.
- Proper diagnostics and save logic.

Respond with `<OK>` if flawless. Otherwise, begin with `Issues found:` and list precise, actionable corrections.

<requirements_summary>
{requirements_summary}
</requirements_summary>
<eda_guidance_plan>
{eda_guidance_str}
</eda_guidance_plan>
<script>
{generated_code}
</script>
"""

def generate_response(prompt_text):
    """
    Sends a prompt to the Gemini model and returns the cleaned text response.
    """
    try:
        print("Sending request to Gemini...")
        response = client.generate_content(contents=prompt_text) 
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

def generate_data_analysis_code(
    filedetails: dict,
    business_problem: str,
    file_path: str,
    eda_guidance: str,
    processed_file_path: str = PROCESSED_DATASET_PATH,
    viz_directory: str | None = None
) -> str:
    if 'error' in filedetails:
        return f"# Error: {filedetails['error']}"
    details = filedetails.copy()
    details['sample_data'] = details.get('sample_data', [])[:3]
    file_details_str = "\n".join(f"- {k}: {str(v)[:200]}" for k, v in details.items())
    prompt = SYSTEM_INSTRUCTION_EDA.format(
        file_details_str=file_details_str,
        business_problem_str=str(business_problem),
        eda_guidance_str=str(eda_guidance),
        file_path_str=str(file_path),
        processed_file_path_str=str(processed_file_path),
        viz_directory_str=str(viz_directory)
    )
    return generate_response(prompt)

def generate_and_refine_eda_code(
    filedetails: dict,
    business_problem: str,
    file_path: str,
    eda_guidance: str,
    processed_file_path: str = PROCESSED_DATASET_PATH,
    max_refinements: int = 3,
    viz_directory: str | None = None
) -> str:
    requirements_summary = (
        "- Load CSV from '{file_path}'. Handle FileNotFoundError.\n"
        "- Initial diagnostics: shape, cols, dtypes, missing, duplicates.\n"
        "- Guided preprocessing per EDA plan.\n"
        "- Visualizations: non-technical, neat, 3-5 charts.\n"
        "- Final diagnostics: head, shape, dtypes, stats, no unexpected object types.\n"
        "- Save to '{processed_file_path}', ensure directory, print confirmation."
    )
    code = generate_data_analysis_code(
        filedetails, business_problem, file_path,
        eda_guidance, processed_file_path,
        viz_directory
    )
    for _ in range(max_refinements):
        reflector_prompt = SYSTEM_INSTRUCTION_REFLECTOR.format(
            requirements_summary=requirements_summary,
            eda_guidance_str=str(eda_guidance),
            generated_code=code
        )
        critique = generate_response(reflector_prompt)
        if critique.strip() == "<OK>" or not critique.startswith("Issues found:"):
            return code
        revision_prompt = (
            f"Revise the script to address the following issues:\n{critique}\n"
            "Ensure adherence to EDA plan and formatting. Output only raw Python code."
        )
        code = generate_response(revision_prompt)
        if code.startswith("# Error"):
            break
    return code

def save_code_to_file(code: str, file_path: str) -> bool:
    if not code or code.startswith("# Error"):
        return False
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(code)
    return True
