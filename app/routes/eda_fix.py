import os
import sys
import re
import time

# --- Attempt to import google.generativeai ---
try:
    import google.generativeai as genai
    print(f"INFO ({__file__}): Successfully imported google.generativeai.", file=sys.stderr)
except ImportError:
    print(f"WARNING ({__file__}): google.generativeai not installed. Code fixing disabled.", file=sys.stderr)
    genai = None

# --- Global Client Initialization Block ---
genai_client = None  # Initialize to None FIRST
if genai:
    try:
        GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBoAFOxBSX1nxEF8lNuhJudPiHVTCRNK8Q")
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set in environment variables")
        
        # Configure the client
        genai.configure(api_key=GEMINI_API_KEY)
        # Use a valid model name (update based on actual available models)
        MODEL_NAME = "gemini-1.5-flash"  # Example; check documentation for correct model
        genai_client = genai.GenerativeModel(MODEL_NAME)
        print(f"INFO ({__file__}): Gemini client configured using model {MODEL_NAME}.", file=sys.stderr)
    except Exception as e:
        print(f"ERROR ({__file__}): Error setting up Gemini client: {e}", file=sys.stderr)
        genai_client = None  # Ensure it's None on error
else:
    print(f"INFO ({__file__}): google.generativeai library not available. Client not configured.", file=sys.stderr)

# Final check after initialization attempt
print(f"INFO ({__file__}): Global genai_client state after setup: {type(genai_client)}", file=sys.stderr)

# --- Helper Function ---
def clean_llm_code_output(raw_code: str) -> str:
    """
    Strips Markdown code fences from LLM output and returns clean Python code.
    """
    if not raw_code:
        return ""
    # Handle multiple markdown styles
    patterns = [
        r"```(?:[\w+\-]*)\s*\n(.*?)\n```",  # Standard fences
        r"```\s*(.*?)\s*```",               # Simple fences
        r"(^[^\n]*\n)?(.*)",                # Fallback: take all after first line
    ]
    for pattern in patterns:
        match = re.search(pattern, raw_code, re.DOTALL)
        if match:
            return match.group(1).strip() if match.lastindex == 1 else match.group(2).strip()
    return raw_code.strip()  # Last resort

# --- Core Function (Uses Global Client) ---
def attempt_code_fix(
    failing_code_str: str,
    error_output_str: str,
    dataset_path_str: str,
    business_problem_str: str,
    file_details_str: str
) -> str | None:
    """
    Uses the globally configured Gemini LLM client to propose a code fix.
    """
    global genai_client
    if not genai_client:
        print(f"ERROR ({__file__}): Global Gemini client not available or not configured. Skipping code fix attempt.", file=sys.stderr)
        return None

    # Validate input strings
    if not failing_code_str:
        print(f"WARNING ({__file__}): No failing code provided. Skipping code fix attempt.", file=sys.stderr)
        return None
    if not error_output_str:
        print(f"WARNING ({__file__}): No error output provided. Skipping code fix attempt.", file=sys.stderr)
        return None

    # Limit context length to avoid exceeding token limits
    error_snippet = error_output_str[:2000]
    file_details_snippet = file_details_str[:1000]

    # Refined Prompt with example
    prompt = f"""
You are an expert Python code reviewer and fixer.
The following Python script encountered an error during execution. Your task is to:
1. Analyze the provided 'Failing Python Code'.
2. Analyze the 'Error Output' it generated.
3. Consider the 'Context' (Dataset Path, Business Problem, File Details).
4. Identify the root cause of the error.
5. Provide a corrected, complete version of the Python script that resolves the error and fulfills the original script's likely intent based on the context.

**Requirements for the Corrected Script:**
- It must be runnable Python code.
- It should address the specific error found in the 'Error Output'.
- Before applying imputation, check that the list of columns is not empty and that those columns exist in the dataframe.
- It should correctly load the dataset from the specified 'Dataset Path'.
- It should align with the 'Business Problem' description.
- It should handle potential issues gracefully (e.g., file not found, unexpected data).
- Use standard libraries like pandas, numpy, and relevant sklearn modules if appropriate for the task implied by the business problem.
- Follow common Python best practices for clarity and readability.
- **Output ONLY the raw, corrected Python code. Do not include explanations, apologies, or markdown formatting like ```python ... ```.**

**Example Expected Output:**
If the error is about missing columns, ensure the code checks for column existence before processing.
Example:
```python
import pandas as pd
df = pd.read_csv('{dataset_path_str}')
columns = [col for col in ['col1', 'col2'] if col in df.columns]
if columns:
    df[columns] = df[columns].fillna(df[columns].mean())
```

**Context:**
- Dataset Path: '{dataset_path_str}'
- Business Problem: '{business_problem_str}'
- File Details Snippet: {file_details_snippet}{'...' if len(file_details_str) > 1000 else ''}

**Failing Python Code:**
```python
{failing_code_str}
"""
    
    # Retry logic for obtaining a fix
    for attempt in range(1, 6):  # Try up to 5 times
        try:
            response = genai_client.generate_content(prompt)
            cleaned_code = clean_llm_code_output(response.text)
            if cleaned_code:
                print(f"INFO ({__file__}): Fix obtained on attempt {attempt}.", file=sys.stderr)
                return cleaned_code
            print(f"WARNING ({__file__}): Attempt {attempt} returned empty code.", file=sys.stderr)
        except Exception as e:
            print(f"ERROR ({__file__}): Attempt {attempt} failed: {e}", file=sys.stderr)
        time.sleep(5)  # Wait 5 seconds before retrying
    print(f"ERROR ({__file__}): Failed to obtain fix after 5 attempts.", file=sys.stderr)
    return None