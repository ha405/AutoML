# routes/visualization_generator.py

import google.generativeai as genai
import os
import sys

# ... (Keep API Key, Model Name, Client Configuration as before) ...
GOOGLE_API_KEY = "AIzaSyBF8Ik7v2Uwy_cRVzoDEj30g2oNpXPPlrQ"
MODEL_NAME = "gemini-2.0-flash" # Or "gemini-pro". Flash is faster, Pro might be slightly better at complex code.

# Configure the Gemini client globally
client_configured = False
model = None
try:
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_AI_API_KEY":
        print("⚠ Warning [VizGen]: Google API Key not set or is placeholder.", file=sys.stderr)
    else:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)
        print(f"Google AI client configured for Visualization Generator using model {MODEL_NAME}.")
        client_configured = True
except Exception as e:
    print(f"❌ Error configuring Google AI client [VizGen]: {e}", file=sys.stderr)


# --- REVISED Prompt Template with Added Context ---
SYSTEM_INSTRUCTION_VISUALIZATION_GENERATOR_TEMPLATE = r"""
You are an expert Python data visualization programmer generating code for a small-scale business audience with no data science background.

**Make it Simple:** Focus on straightforward, relevant charts—avoid distribution plots or detailed error ranges. If you include an error-related chart, present it as plain differences (e.g., "How far predictions were off from actual sales").

**Limit Categories:** When there are many categories on an axis, pick only the top 5 or 7 by count or importance. Ensure x-axis labels are angled or vertical (not horizontal) for easy reading.

**No Jargon:** Do not use terms like "median", "distribution", "residual", or "heteroscedasticity". Use everyday language (e.g., "middle value" or "difference").

Create a variety of clear, relevant charts based on the business problem and dataset, using plain-English titles and labels.

*Input Context:*

<visualization_plan>
{visualization_plan_str}
</visualization_plan>

<processed_data_path>
{processed_data_path_str}
</processed_data_path>

<business_problem>
{business_problem_str}
</business_problem>

<visualization_output_directory>
{visualization_output_dir_str}
</visualization_output_directory>

<processed_file_details>
This section contains details about the processed data file (column names and basic stats). Use this to confirm columns before plotting.
{processed_file_details_str}
</processed_file_details>

<ml_output_logs>
This section contains logs from the ML script (feature importances, R², MAE, sample predictions).
{ml_output_logs_str}
</ml_output_logs>

*Instructions for Python Script Generation:*

1. **Imports:** pandas, matplotlib.pyplot as plt, seaborn as sns, os. Only include numpy or sklearn if plan explicitly requires.
2. **Prepare Output Folder:** `os.makedirs('{visualization_output_dir_str}', exist_ok=True)`.
3. **Load Data:**
   ```python
   try:
       df = pd.read_csv(r"{processed_data_path_str}")
   except FileNotFoundError:
       print("❌ Data file not found; check the path.")
       sys.exit(1)
   ```
   Check required columns from `<processed_file_details>`; if missing, print a warning and skip that chart.
4. **Parse ML Outputs:**
   - Extract values (importances, predictions, differences) from `<ml_output_logs>` by looking for clear markers.
   - If a value isn't in the logs, print `"# Data not available in logs; skipping this chart."` and move on.
5. **Generate Each Chart (with try/except):**
   - Use simple chart types (bar, scatter, line). No complex diagrams.
   - Use plain-English titles (e.g., "Sales Over Time", "Top 5 Product Counts").
   - Label axes clearly (e.g., "Number of Sales", "Revenue in Dollars").
   - Add basic annotations (e.g., `plt.axhline(avg, linestyle='--', label='Average') # shows typical value`).
   - Save each chart:
     ```python
     fig_path = os.path.join('{visualization_output_dir_str}', 'chart_name.png')
     plt.savefig(fig_path, bbox_inches='tight')
     plt.close()
     ```
6. **Error Handling:**
   Wrap each chart block in `try/except Exception as e:` and print `f"❌ Chart [N] failed: {{e}}"` to continue.
7. **Final Output:**
   Only raw Python code should be returned—no extra commentary.
"""

# --- Generic Gemini Response Function (Keep as before) ---
def generate_response(prompt_content: str):
    # ... (Keep the existing implementation) ...
    if not client_configured or model is None:
        return "# Error: Google AI client not configured for Visualization Generator."

    try:
        print(f"Sending request to {MODEL_NAME} [VizGen]...")
        generation_config = genai.types.GenerationConfig(temperature=0.15)

        response = model.generate_content(
            contents=prompt_content,
            generation_config=generation_config
        )
        print("Response received [VizGen].")

        if response.prompt_feedback and response.prompt_feedback.block_reason:
            error_msg = f"# Error [VizGen]: Prompt blocked by Google AI due to {response.prompt_feedback.block_reason}"
            print(f"❌ {error_msg}", file=sys.stderr)
            return error_msg
        try:
            text = response.text
        except (ValueError, AttributeError) as e:
             error_msg = f"# Error [VizGen]: No content generated or unexpected format ({e})."
             if response.candidates and response.candidates[0].finish_reason != 'STOP':
                 error_msg += f" Finish Reason: {response.candidates[0].finish_reason}"
             print(f"❌ {error_msg} (Candidates: {response.candidates})", file=sys.stderr)
             return error_msg

        text = text.strip()
        # Stronger cleaning for markdown code blocks
        if text.startswith("```python"):
            text = text[len("```python"):].strip()
        elif text.startswith("```"):
             text = text[len("```"):].strip() # Handle case without 'python'
        if text.endswith("```"):
            text = text[:-3].strip()
        return text

    except Exception as e:
        print(f"❌ An error occurred during Google AI API call [VizGen]: {e}", file=sys.stderr)
        return f"# Error generating visualization code via Google AI: {e}"


# --- UPDATED Function to Generate Visualization Code ---
def generate_visualization_code(visualization_plan_str: str,
                                processed_data_path_str: str,
                                business_problem_str: str,
                                visualization_output_dir_str: str,
                                # --- NEW PARAMETERS ---
                                processed_file_details_str: str,
                                ml_output_logs_str: str
                                # --- END NEW PARAMETERS ---
                                ) -> str:
    """
    Generates Python code for visualizations based on the plan and context.
    """
    if not client_configured:
        return "# Error: Cannot generate visualization code, Google AI client not configured."

    print("--- Generating Visualization Code with Context ---")

    # Format the prompt with all context
    prompt_content = SYSTEM_INSTRUCTION_VISUALIZATION_GENERATOR_TEMPLATE.format(
        visualization_plan_str=visualization_plan_str,
        processed_data_path_str=processed_data_path_str,
        business_problem_str=business_problem_str,
        visualization_output_dir_str=visualization_output_dir_str,
        # --- NEW CONTEXT ---
        processed_file_details_str=processed_file_details_str,
        ml_output_logs_str=ml_output_logs_str
        # --- END NEW CONTEXT ---
    )

    # Generate the code
    viz_code = generate_response(prompt_content)

    # Apply cleaning just in case generate_response didn't catch everything
    cleaned_code = viz_code.strip()
    if cleaned_code.startswith("```python"):
        cleaned_code = cleaned_code[len("```python"):].strip()
    elif cleaned_code.startswith("```"):
         cleaned_code = cleaned_code[len("```"):].strip()
    if cleaned_code.endswith("```"):
        cleaned_code = cleaned_code[:-3].strip()


    print("Visualization Code Generation Complete.")
    return cleaned_code # Return potentially cleaned code

# --- Function to Save Visualization Code (Keep as before) ---
def save_visualization_code(code: str, filename: str):
    # ... (Keep the existing implementation) ...
    save_dir = os.path.dirname(filename)
    base_filename = os.path.basename(filename)

    try:
        if not save_dir:
             save_dir = "."
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, base_filename)

        if code is None or code.startswith("# Error"):
            print(f"❌ Error saving file: Code content is invalid or None.")
            if code:
                with open(file_path, "w", encoding='utf-8') as f:
                    f.write(f"# Generation Failed:\n{code}")
                print(f"⚠ Error message saved to: {file_path}")
            return False

        with open(file_path, "w", encoding='utf-8') as f:
            f.write(code)
        print(f"✅ Visualization script saved successfully at: {file_path}")
        return True

    except OSError as e:
        print(f"❌ Error creating directory or saving file {filename}: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred during viz code saving: {e}")
    return False