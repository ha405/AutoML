# routes/visualization_planner.py

import google.generativeai as genai
import os
import sys

# --- Configuration for Google AI Studio ---
# WARNING: Hardcoding API keys is generally insecure. Consider environment variables.
GOOGLE_API_KEY = "AIzaSyBF8Ik7v2Uwy_cRVzoDEj30g2oNpXPPlrQ"
MODEL_NAME = "gemini-2.0-flash" # Using the specified Gemini model

# Configure the Gemini client globally
client_configured = False
model = None
try:
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_AI_API_KEY": # Basic check
        print("⚠️ Warning [VizPlanner]: Google API Key not set or is placeholder.", file=sys.stderr)
    else:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)
        print(f"✅ Google AI client configured for Visualization Planner using model {MODEL_NAME}.")
        client_configured = True
except Exception as e:
    print(f"❌ Error configuring Google AI client [VizPlanner]: {e}", file=sys.stderr)

# --- REVISED PROMPT TEMPLATE V5 (Removed Conflict, Reinforced Mandate) ---
SYSTEM_INSTRUCTION_VISUALIZATION_PLANNER_TEMPLATE = r"""
You are a data visualization consultant creating a concise, actionable visualization plan for a **Small/Medium Enterprise (SME)** manager (non-technical audience).
Your goal is to plan **exactly 5 essential, easy-to-understand charts** that tell a coherent story addressing the business problem, using insights derived ONLY from the provided EDA context, ML context, and processed data details.

**CRITICAL:** Base your plan ONLY on the processed data, EDA results, and ML results provided below. Adhere strictly to the 5 mandated chart types.

*Input Context:*

<business_problem>
{business_problem_str}
</business_problem>

<processed_data_details>
Details about the processed data file used for ML (column names, types, basic stats). Use this to confirm column names for plotting.
{file_details_processed_str}
</processed_data_details>

<processed_data_path>
Path to the processed data CSV file: {processed_dataset_path_str}
</processed_data_path>

<eda_context>
Code and logs from the Exploratory Data Analysis phase performed on the processed data. Look here for insights about distributions, relationships, or initial feature relevance.
Code:
{eda_code_str}

Logs:
{eda_output_logs_str}
</eda_context>

<ml_context>
Code and logs from the Machine Learning phase performed on the processed data. Look here for feature importances, model performance metrics (R-squared, MAE, etc.), predictions, and residuals.
Code:
{ml_code_str}

Output Logs:
{ml_output_str}
</ml_context>

**Instructions for SME Visualization Plan (Mandated Plots):**

1.  **Overall Goal (Business Focus):**
    *   State the primary business objective the 5 visualizations collectively achieve (e.g., "To illustrate typical car prices, show price relationships with key features, identify main price drivers, and assess the prediction model's reliability for strategic decisions.").
2.  **Mandatory Visualizations (You MUST plan these specific 5 charts):**

    *   **Chart 1: Target Variable Distribution (from EDA Context):**
        *   **Type:** Histogram
        *   **Business Insight:** Understanding the typical range and frequency of the target variable (e.g., 'price').
        *   **Business Value:** Sets baseline understanding of what's being predicted.
        *   **Data Source:** Target variable column identified in `<processed_data_details>`, insights from `<eda_context>`.
        *   **Clarity Tip:** Title: "Distribution of [Target Variable Name, e.g., Car Prices]"; Annotate average/median.

    *   **Chart 2: Target vs. Top Continuous Feature (from EDA Context):**
        *   **Type:** Scatter Plot
        *   **Business Insight:** Visualizing the relationship between the target variable and the most influential continuous feature (identified from `<eda_context>` or `<ml_context>` importances).
        *   **Business Value:** Shows how the primary driver directly impacts the target, informing strategy.
        *   **Data Source:** Target variable and the top continuous predictor column from `<processed_data_details>`, based on `<eda_context>`/`<ml_context>`.
        *   **Clarity Tip:** Title: "[Target Var] Relationship with [Top Feature]"; Clear axis labels.

    *   **Chart 3: Key Drivers / Feature Importance (from ML Context):**
        *   **Type:** Bar Chart (Horizontal Recommended)
        *   **Business Insight:** Ranking the factors identified by the ML model as most impactful on the target variable.
        *   **Business Value:** Pinpoints features management can potentially manipulate or focus on.
        *   **Data Source:** Feature importances extracted *primarily* from `<ml_output_logs>`. Use column names from `<processed_data_details>`.
        *   **Clarity Tip:** Title: "Key Factors Influencing [Target Var]"; Clear labels; highlight top 2-3.

    *   **Chart 4: Prediction Accuracy - Actual vs. Predicted (from ML Context):**
        *   **Type:** Scatter Plot
        *   **Business Insight:** Assessing the overall accuracy and reliability of the ML model's predictions.
        *   **Business Value:** Builds confidence (or flags issues) in using the model for estimations.
        *   **Data Source:** Actual target variable values vs. Predicted values (extract from `<ml_output_logs>` if possible, e.g., from test set results).
        *   **Clarity Tip:** Title: "Model Prediction Accuracy: Actual vs. Predicted [Target Var]"; Annotate with R-squared (extracted from logs). Add a y=x reference line.

    *   **Chart 5: Prediction Error Analysis (from ML Context):**
        *   **Type:** Histogram
        *   **Business Insight:** Understanding the typical size and distribution of prediction errors (residuals).
        *   **Business Value:** Quantifies the model's uncertainty, helping managers understand the likely +/- range for predictions.
        *   **Data Source:** Residuals (Actual - Predicted) calculated or extracted from `<ml_output_logs>`.
        *   **Clarity Tip:** Title: "Typical Range of Prediction Error"; Annotate with Mean Absolute Error (MAE) or Median Absolute Error (extracted from logs).

3.  **Data Extraction & Handling:**
    *   Crucially, values for charts 3, 4, 5 (importances, predictions, residuals, metrics like R2/MAE) **MUST be extracted from the `<ml_output_logs>`**. Scan the logs carefully for these values. Do NOT make up numbers.
    *   If essential values *cannot* be reliably extracted from the logs, state this limitation clearly in the plan for that chart's "Data Source" section (e.g., "Data Source: Residuals (Actual - Predicted) - NOTE: Requires extraction from ML logs, may not be available.").
4.  **LLM Code Guidance (Clarity Focus):**
    *   Recommend libraries (matplotlib/seaborn).
    *   Emphasize clear, business-friendly titles/labels and annotations as per the Clarity Tips.

**Final Check:** Ensure your output plan includes sections for *all 5* specified chart types in the requested format. Do not omit any.

**Output Format:**
Use Markdown headings for each of the 5 charts, following the structure: `### Chart [N]: [Type] - [Purpose]`. Include the specified sections (Business Insight, Business Value, Data Source, Clarity Tip).
"""
# --- END OF REVISED PROMPT V5 ---


# --- Function to Generate Visualization Plan using Google AI ---
def generate_visualization_plan(
    business_problem_str: str,
    file_details_processed_str: str, # Details of the processed data
    processed_dataset_path_str: str, # Path to the processed data
    eda_code_str: str,               # EDA code used
    eda_output_logs_str: str,        # Logs from EDA execution
    ml_code_str: str,                # ML code used
    ml_output_str: str               # Logs from ML execution (contains results)
) -> str:
    """
    Generates a visualization plan for SMEs using Google AI (Gemini),
    based on processed data, EDA, and ML context.
    """
    if not client_configured or model is None:
        return "# Error: Google AI client not configured for Visualization Planner."

    processed_dataset_prompt_path = processed_dataset_path_str.replace("\\", "/")

    try:
        prompt = SYSTEM_INSTRUCTION_VISUALIZATION_PLANNER_TEMPLATE.format(
            business_problem_str=business_problem_str,
            file_details_processed_str=file_details_processed_str,
            processed_dataset_path_str=processed_dataset_prompt_path,
            eda_code_str=eda_code_str,
            eda_output_logs_str=eda_output_logs_str,
            ml_code_str=ml_code_str,
            ml_output_str=ml_output_str
        )
    except KeyError as e:
        print(f"❌ Error formatting prompt [VizPlanner]: Missing key {e}. Check template and function arguments.")
        return f"# Error: Internal prompt formatting error - missing key {e}"
    except Exception as e:
        print(f"❌ Error preparing prompt [VizPlanner]: {e}")
        return f"# Error: Internal error preparing prompt - {e}"

    generation_config = genai.types.GenerationConfig(temperature=0.2) # Keep temperature low for planning

    print(f"Sending request to {MODEL_NAME} [VizPlanner]...")
    try:
        response = model.generate_content(
            contents=prompt,
            generation_config=generation_config
            # safety_settings={'HARASSMENT':'block_none'} # Optional: relax safety slightly if blocking valid plans
        )
        print("Response received [VizPlanner].")

        if response.prompt_feedback and response.prompt_feedback.block_reason:
            error_msg = f"# Error [VizPlanner]: Prompt blocked by Google AI due to {response.prompt_feedback.block_reason}"
            print(f"❌ {error_msg}", file=sys.stderr)
            # Consider logging the prompt here if blocks are frequent
            return error_msg
        try:
            plan_content = response.text.strip()
            if not plan_content:
                 print("❌ Warning [VizPlanner]: Visualization plan generated by LLM was empty.")
                 return "# Error: Failed to generate plan content (LLM returned empty)."
        except (ValueError, AttributeError) as e:
             error_msg = f"# Error [VizPlanner]: No content generated or unexpected format ({e})."
             if response.candidates and response.candidates[0].finish_reason != 'STOP':
                 error_msg += f" Finish Reason: {response.candidates[0].finish_reason}"
             print(f"❌ {error_msg} (Candidates: {response.candidates})", file=sys.stderr)
             return error_msg

        print("✅ Visualization plan generated successfully.")
        return plan_content

    except Exception as e:
        print(f"❌ An error occurred during Google AI API call [VizPlanner]: {e}", file=sys.stderr)
        return f"# Error generating visualization plan via Google AI: {e}"