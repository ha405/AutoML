import google.generativeai as genai
import os
import sys

# --- Configuration for Google AI Studio ---
# WARNING: Hardcoding API keys is generally insecure. Consider environment variables.
GOOGLE_API_KEY = "AIzaSyBF8Ik7v2Uwy_cRVzoDEj30g2oNpXPPlrQ"
MODEL_NAME = "gemini-2.0-flash"  # Using the specified Gemini model

# Configure the Gemini client globally
client_configured = False
model = None
try:
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_AI_API_KEY":
        print("⚠️ Warning [VizPlanner]: Google API Key not set or is placeholder.", file=sys.stderr)
    else:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)
        print(f"✅ Google AI client configured for Visualization Planner using model {MODEL_NAME}.")
        client_configured = True
except Exception as e:
    print(f"❌ Error configuring Google AI client [VizPlanner]: {e}", file=sys.stderr)

SYSTEM_INSTRUCTION_VISUALIZATION_PLANNER_TEMPLATE = r"""
You are a data visualization consultant creating a friendly, easy-to-understand plan for a small business manager with no technical background.

**Make It Look Good & Simple:**
- Choose simple chart types (bar, line, scatter) that are visually appealing and immediately clear.
- Use a consistent, muted color palette and light grid lines for readability.
- No distribution or error-range jargon—if you show differences, call it "gap" or "difference" (e.g., "Gap between predicted and actual sales").

**Limit Category Counts:**
- If an axis has many categories, pick the top 5 or 7 by value. Do not list more.
- Make sure x-axis labels are straight and not reclined; never keep long labels horizontal maybe use shorthand notation for them.

**Everyday Language Only:**
- Titles and labels must use plain English (e.g., "Sales by Region", not "Histogram of Sales Distribution").
- Avoid technical terms like "median", "variance", "residual". Use "middle value", "spread", "difference".

**Context & Output:**
<business_problem>
{business_problem_str}
</business_problem>

<processed_data_details>
{file_details_processed_str}
</processed_data_details>

<processed_data_path>
{processed_dataset_path_str}
</processed_data_path>

<eda_context>
{eda_code_str}
--- Logs ---
{eda_output_logs_str}
</eda_context>

<ml_context>
{ml_code_str}
--- Logs ---
{ml_output_str}
</ml_context>

**Plan Instructions:**
- Suggest 4-6 charts that directly answer the business question you may use dataset or ML model to make relevant graphs.
- For each chart, specify:
  - **Chart Name:** Plain short title.
  - **Type:** Bar, Line, or Scatter.
  - **Insight:** One-sentence why it matters.
  - **Data Fields:** Which columns or values to plot.
  - **Design Tips:** e.g., "Use a light grid", "Rotate labels 45°", "Highlight top bar in a contrasting color".

**Final Format:**
Use Markdown headings: `### Chart 1: Sales by Region` and list the sections above. 
"""

# --- Function to Generate Visualization Plan using Google AI ---
def generate_visualization_plan(
    business_problem_str: str,
    file_details_processed_str: str,
    processed_dataset_path_str: str,
    eda_code_str: str,
    eda_output_logs_str: str,
    ml_code_str: str,
    ml_output_str: str
) -> str:
    """
    Generates a jargon-free, SME-friendly visualization plan using Google AI (Gemini).
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
        print(f"❌ Error formatting prompt [VizPlanner]: Missing key {e}.", file=sys.stderr)
        return f"# Error: Internal prompt formatting error - missing key {e}"
    except Exception as e:
        print(f"❌ Error preparing prompt [VizPlanner]: {e}", file=sys.stderr)
        return f"# Error: Internal error preparing prompt - {e}"

    generation_config = genai.types.GenerationConfig(temperature=0.2)
    print(f"Sending request to {MODEL_NAME} [VizPlanner]...")

    try:
        response = model.generate_content(
            contents=prompt,
            generation_config=generation_config
        )
        print("Response received [VizPlanner].")

        if response.prompt_feedback and response.prompt_feedback.block_reason:
            error_msg = f"# Error [VizPlanner]: Prompt blocked due to {response.prompt_feedback.block_reason}"
            print(f"❌ {error_msg}", file=sys.stderr)
            return error_msg

        plan_content = response.text.strip() if hasattr(response, 'text') else ''
        if not plan_content:
            print("❌ Warning [VizPlanner]: Empty plan content.", file=sys.stderr)
            return "# Error: Failed to generate plan content."

        print("✅ Visualization plan generated successfully.")
        return plan_content

    except Exception as e:
        print(f"❌ Error during Google AI API call [VizPlanner]: {e}", file=sys.stderr)
        return f"# Error generating visualization plan via Google AI: {e}"
