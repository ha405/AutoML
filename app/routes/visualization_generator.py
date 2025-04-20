# routes/visualization_generator.py

import google.generativeai as genai
import os
import sys

GOOGLE_API_KEY = "AIzaSyBoAFOxBSX1nxEF8lNuhJudPiHVTCRNK8Q" 
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

# --- Prompt Template for Visualization Code Generation ---
SYSTEM_INSTRUCTION_VISUALIZATION_GENERATOR_TEMPLATE = r"""
You are an expert Python data visualization programmer. Your task is to generate a complete and executable Python script to create visualizations based strictly on the provided plan.

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

*Instructions for Python Script Generation:*

1.  *Strictly Follow the Plan:* Carefully read the <visualization_plan>. Your primary goal is to implement the visualizations exactly as described in the "Recommended Visualizations" section of the plan.
2.  *Imports:* Import necessary libraries: pandas, matplotlib.pyplot as plt, seaborn as sns. Also import os. If the plan specifically requests plotly, import plotly.express as px.
3.  *Create Output Directory:* Add code using os.makedirs('{visualization_output_dir_str}', exist_ok=True) at the beginning of the script to ensure the directory for saving plots exists.
4.  *Load Data:* Load the processed dataset from the path specified in <processed_data_path>. Use pd.read_csv(r"{processed_data_path_str}"). Include a try-except FileNotFoundError block around the loading code and print an informative error message if the file is not found.
5.  *Generate Code per Visualization:* For each visualization detailed in the plan:
    *   Use the recommended visualization type (e.g., sns.histplot, plt.scatter, sns.heatmap, px.bar).
    *   Use the exact data columns specified in the "Data to Use" section of the plan for that visualization. Include error handling (e.g., a try-except KeyError block or checking if column in df.columns:) before attempting to plot with specific columns. Print a warning if a required column is missing.
    *   Set clear and relevant plot titles, x-axis labels, and y-axis labels, drawing context from the "Insight to Visualize" section of the plan and the overall <business_problem>.
    *   Apply any specific styling or library preferences mentioned in the "Specific Instructions for Visualization Generation LLM" section of the plan. Default to Seaborn/Matplotlib if not specified.
    *   *Save Each Plot:* Save each generated plot to a separate file inside the <visualization_output_directory>. Use descriptive filenames, like visualization_1_histogram.png, visualization_2_scatterplot.png, etc. Use plt.savefig(os.path.join('{visualization_output_dir_str}', 'filename.png'), bbox_inches='tight') for matplotlib/seaborn plots. For plotly, use fig.write_image(os.path.join('{visualization_output_dir_str}', 'filename.png')) (this requires kaleido to be installed: pip install kaleido). If using plotly, add an import check for kaleido. Add plt.close() after saving matplotlib/seaborn plots to prevent them from displaying in non-interactive environments and overlapping.
6.  *Error Handling:* Include basic error handling (e.g., try-except blocks) around individual plotting sections to prevent the entire script from crashing if one plot fails. Print informative error messages if a plot cannot be generated.
7.  *Executable Script:* The final output must be only a complete, raw Python script, ready to be executed.
8.  *No Extra Text:* Do NOT include any markdown formatting (like python ... ), explanations, comments (unless essential for complex logic), or introductory/concluding remarks. The script should start directly with the import statements.

*Output Format:*
*   Your response MUST contain ONLY the raw Python code for the script.
"""

# --- Generic Gemini Response Function (adapted from your other modules) ---
def generate_response(prompt_content: str):
    """Sends a prompt string to the configured Gemini model and returns the text response."""
    if not client_configured or model is None:
        return "# Error: Google AI client not configured for Visualization Generator."

    try:
        print(f"Sending request to {MODEL_NAME} [VizGen]...")
        # Configure generation parameters if needed (e.g., temperature)
        generation_config = genai.types.GenerationConfig(temperature=0.15) # Slightly lower for code

        response = model.generate_content(
            contents=prompt_content, # Pass the formatted prompt string directly
            generation_config=generation_config
        )
        print("Response received [VizGen].")

        # Error checking (prompt feedback, content existence)
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            error_msg = f"# Error [VizGen]: Prompt blocked by Google AI due to {response.prompt_feedback.block_reason}"
            print(f"❌ {error_msg}", file=sys.stderr)
            return error_msg
        try:
            text = response.text
        except (ValueError, AttributeError) as e:
             error_msg = f"# Error [VizGen]: No content generated or unexpected format ({e})."
             # Add more details if available
             if response.candidates and response.candidates[0].finish_reason != 'STOP':
                 error_msg += f" Finish Reason: {response.candidates[0].finish_reason}"
             print(f"❌ {error_msg} (Candidates: {response.candidates})", file=sys.stderr)
             return error_msg

        # Clean markdown (ensure consistency)
        text = text.strip()
        if text.startswith("python"):
            text = text[len("python"):].strip()
        if text.startswith(""):
            text = text[len(""):].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
        return text

    except Exception as e:
        print(f"❌ An error occurred during Google AI API call [VizGen]: {e}", file=sys.stderr)
        return f"# Error generating visualization code via Google AI: {e}"

# --- Function to Generate Visualization Code ---
def generate_visualization_code(visualization_plan_str: str,
                                processed_data_path_str: str,
                                business_problem_str: str,
                                visualization_output_dir_str: str) -> str:
    """
    Generates Python code for visualizations based on the plan.
    """
    if not client_configured:
        return "# Error: Cannot generate visualization code, Google AI client not configured."

    print("--- Generating Visualization Code ---")

    # Format the prompt
    prompt_content = SYSTEM_INSTRUCTION_VISUALIZATION_GENERATOR_TEMPLATE.format(
        visualization_plan_str=visualization_plan_str,
        processed_data_path_str=processed_data_path_str,
        business_problem_str=business_problem_str,
        visualization_output_dir_str=visualization_output_dir_str
    )

    # Generate the code
    viz_code = generate_response(prompt_content)

    print("Visualization Code Generation Complete.")
    return viz_code

# --- Function to Save Visualization Code ---
def save_visualization_code(code: str, filename: str):
    """Saves the generated visualization code to a file."""
    save_dir = os.path.dirname(filename) 
    base_filename = os.path.basename(filename) 

    try:
        if not save_dir: # Handle case where filename might not have a directory part
             save_dir = "." # Save in current directory
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, base_filename)

        if code is None or code.startswith("# Error"):
            print(f"❌ Error saving file: Code content is invalid or None.")
            if code:
                with open(file_path, "w", encoding='utf-8') as f:
                    f.write(f"# Generation Failed:\n{code}")
                print(f"⚠ Error message saved to: {file_path}")
            return False # Indicate failure

        with open(file_path, "w", encoding='utf-8') as f:
            f.write(code)
        print(f"✅ Visualization script saved successfully at: {file_path}")
        return True # Indicate success

    except OSError as e:
        print(f"❌ Error creating directory or saving file {filename}: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred during viz code saving: {e}")
    return False # Indicate failure