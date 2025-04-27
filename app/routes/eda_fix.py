import os
import sys
import re
import time
import traceback

try:
    import google.generativeai as genai
    print(f"INFO ({__file__}): Successfully imported google.generativeai.", file=sys.stderr)
except ImportError:
    print(f"WARNING ({__file__}): google.generativeai not installed. Code fixing disabled.", file=sys.stderr)
    genai = None

genai_client = None
if genai:
    try:
        GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
        if not GEMINI_API_KEY:
            GEMINI_API_KEY = "AIzaSyBF8Ik7v2Uwy_cRVzoDEj30g2oNpXPPlrQ"
            print(f"WARNING ({__file__}): GOOGLE_API_KEY not set in environment variables. Using placeholder key.", file=sys.stderr)
        
        genai.configure(api_key=GEMINI_API_KEY)
        MODEL_NAME = "gemini-2.0-flash"
        genai_client = genai.GenerativeModel(MODEL_NAME)
        print(f"INFO ({__file__}): Gemini client configured for code fixing using model {MODEL_NAME}.", file=sys.stderr)
    except Exception as e:
        print(f"ERROR ({__file__}): Error setting up Gemini client for code fixing: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        genai_client = None
else:
    print(f"INFO ({__file__}): google.generativeai library not available. Client not configured.", file=sys.stderr)
    genai_client = None

def clean_llm_code_output(raw_code: str) -> str:
    if not raw_code:
        return ""
    cleaned = raw_code.strip()
    if cleaned.startswith("```python"):
        cleaned = cleaned[len("```python"):].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    cleaned = re.sub(
        r"^(here's|here is)\s+(the\s+)?(corrected|fixed|python\s+)?code:?\s*\n",
        "",
        cleaned,
        flags=re.IGNORECASE
    ).strip()
    print("EDA_FIX_RETURN")
    return cleaned

def attempt_code_fix(
    failing_code_str: str,
    error_output_str: str,
    dataset_path_str: str,
    business_problem_str: str,
    file_details_str: str,
    eda_guidance: str
) -> str | None:
    global genai_client
    if not genai_client:
        print(f"ERROR ({__file__}): Global Gemini client not available or not configured. Skipping code fix attempt.", file=sys.stderr)
        return None
    if not failing_code_str:
        print(f"WARNING ({__file__}): No failing code provided. Skipping code fix attempt.", file=sys.stderr)
        return None
    if not error_output_str:
        print(f"WARNING ({__file__}): No error output provided. Skipping code fix attempt.", file=sys.stderr)
        return None
    if not eda_guidance:
        print(f"WARNING ({__file__}): No EDA guidance provided to code fixer. Context will be limited.", file=sys.stderr)
        eda_guidance = "# No EDA guidance available for context."

    error_snippet = error_output_str[:2000]
    file_details_snippet = file_details_str[:1000]
    eda_guidance_snippet = eda_guidance[:2000]

    prompt = f"""
You are an expert Python code reviewer and fixer specializing in EDA and data preprocessing scripts.
The following Python script encountered an error during execution. Your task is to:
1. Analyze the 'Failing Python Code'.
2. Analyze the 'Error Output'.
3. Consider the overall 'Context' (Dataset Path, Business Problem, File Details, **and the original EDA Guidance Plan**).
4. Identify the root cause of the error.
5. Provide a corrected, complete version of the Python script that resolves the error AND **still adheres to the original EDA Guidance Plan as much as possible** while fixing the error.

**Requirements for the Corrected Script:**
- It must be runnable Python code.
- It must address the specific error found in the 'Error Output'.
- It must align with the goals and steps outlined in the 'EDA Guidance Plan'.
- It must correctly load the dataset from the specified 'Dataset Path'.
- It should align with the 'Business Problem' description.
- Use standard libraries (pandas, numpy, os, sklearn).
- **Output ONLY the raw, corrected Python code. Do not include explanations, apologies, or markdown formatting like ```python ... ```.**

**Context:**
- Dataset Path: '{dataset_path_str}'
- Business Problem: '{business_problem_str}'
- File Details Snippet: {file_details_snippet}{'...' if len(file_details_str) > 1000 else ''}
- EDA Guidance Plan Snippet:
<eda_guidance_plan>
{eda_guidance_snippet}{'...' if len(eda_guidance) > 2000 else ''}
</eda_guidance_plan>

**Failing Python Code:**
{failing_code_str}
"""
    try:
        response = genai_client.generate_content(contents={'parts': [{'text': prompt}]})
        if response and getattr(response, 'parts', None):
            raw_code = response.parts[0].text or ''
            return clean_llm_code_output(raw_code)
        else:
            print(f"ERROR ({__file__}): No code returned from Gemini.", file=sys.stderr)
            return None
    except Exception as e:
        print(f"ERROR ({__file__}): Exception during code fix: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return None
