import os
import sys
import re
import time
import traceback

try:
    import google.generativeai as genai
    print(f"INFO ({__file__}): Successfully imported google.generativeai.", file=sys.stderr)
except ImportError:
    print(f"WARNING ({__file__}): google.generativeai not installed. ML code fixing disabled.", file=sys.stderr)
    genai = None

genai_client = None
if genai:
    try:
        GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
        if not GEMINI_API_KEY:
            GEMINI_API_KEY = "YOUR_API_KEY_HERE"
            print(f"WARNING ({__file__}): GOOGLE_API_KEY not set in environment variables. Using placeholder key.", file=sys.stderr)
            if GEMINI_API_KEY == "YOUR_API_KEY_HERE":
                 print(f"ERROR ({__file__}): Placeholder API Key detected. Gemini client will likely fail. Please set GOOGLE_API_KEY environment variable.", file=sys.stderr)

        genai.configure(api_key=GEMINI_API_KEY)
        MODEL_NAME = "gemini-1.5-flash-latest"
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=8192
        )
        genai_client = genai.GenerativeModel(
            MODEL_NAME,
            safety_settings=safety_settings,
            generation_config=generation_config
            )
        print(f"INFO ({__file__}): Gemini client configured for ML fix using model {MODEL_NAME}.", file=sys.stderr)
    except Exception as e:
        print(f"ERROR ({__file__}): Error setting up Gemini client for ML fix: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        genai_client = None
else:
    print(f"INFO ({__file__}): google.generativeai library not available. ML Fix Client not configured.", file=sys.stderr)
    genai_client = None

def clean_llm_code_output(raw_code: str) -> str:
    if not raw_code:
        return ""
    cleaned = raw_code.strip()
    cleaned = re.sub(r"^```python\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(
        r"^(here(?:'s| is)\s+(?:the\s+)?(?:corrected|fixed|updated|python\s+)?code:?)\s*",
        "",
        cleaned,
        flags=re.IGNORECASE | re.MULTILINE
    ).strip()
    return cleaned

def attempt_ml_code_fix(
    broken_ml_code: str,
    error_message: str,
    dataset_path: str,
    business_goal: str,
    ml_plan: str,
    eda_output_logs: str
) -> str | None:
    global genai_client
    if not genai_client:
        print(f"ERROR ({__file__}): Gemini client not available. Cannot fix ML code.", file=sys.stderr)
        return None

    if not broken_ml_code or not broken_ml_code.strip():
        print(f"WARNING ({__file__}): Missing or empty failing ML code string. Aborting fix.", file=sys.stderr)
        return None
    if not error_message or not error_message.strip():
        print(f"WARNING ({__file__}): Missing or empty error message string. Aborting fix.", file=sys.stderr)
        return None
    if not dataset_path:
        print(f"WARNING ({__file__}): Missing dataset path. Aborting fix.", file=sys.stderr)
        return None

    code_snippet = broken_ml_code[-8000:]
    error_snippet = error_message[-4000:]
    plan_snippet = ml_plan[:4000] if ml_plan else "[No ML Plan Provided]"
    eda_logs_snippet = eda_output_logs[-4000:] if eda_output_logs else "[No EDA Logs Provided]"
    goal_snippet = business_goal[:1000] if business_goal else "[No Business Goal Provided]"

    prompt = f"""
You are an expert Python machine learning engineer specializing in debugging and correcting code based on context.
Your task is to analyze the provided failing ML script, the exact error message, and the relevant context (dataset path, business goal, ML plan, EDA logs). You must provide a fully corrected version of the *entire original script*.

**Analysis Context:**

1.  **Business Goal:**
    {goal_snippet}

2.  **ML Plan:**
    ```text
    {plan_snippet}
    ```

3.  **Processed Dataset Path:**
    `{dataset_path}`

4.  **Relevant EDA Output Logs (showing final data state loaded from the dataset path):**
    ```text
    {eda_logs_snippet}
    ```

5.  **Failing ML Code:**
    ```python
    {code_snippet}
    ```

6.  **Exact Error Output / Traceback:**
    ```text
    {error_snippet}
    ```

**Correction Requirements:**

*   **Primary Goal:** Fix the specific error(s) indicated in the "Error Output" section, ensuring the code aligns with the "ML Plan" and uses the data structure implied by the "EDA Output Logs" loaded from the "Processed Dataset Path".
*   **Return Full Code:** Output the *complete*, corrected Python script. Do not output only fragments.
*   **Minimal Changes:** Correct only what is necessary to fix the error and align with the plan/data. Avoid unnecessary refactoring, style changes, or adding extensive comments/prints.
*   **Preserve Structure:** Maintain the original script's overall structure, variable names, function names, and existing comments unless modification is essential for the fix.
*   **Output Format:** Respond ONLY with the corrected Python code, enclosed in triple backticks (```python ... ```). Do not include any explanations, apologies, or introductory phrases before or after the code block.

**Corrected Python Code:**
"""

    retry_attempts = 2
    current_attempt = 0
    backoff_time = 2

    while current_attempt < retry_attempts:
        current_attempt += 1
        try:
            print(f"INFO ({__file__}): Sending request to Gemini for ML code fix (Attempt {current_attempt}/{retry_attempts})...", file=sys.stderr)
            response = genai_client.generate_content(contents=prompt)

            if response and getattr(response, 'text', None):
                raw_code = response.text
                cleaned_code = clean_llm_code_output(raw_code)
                if cleaned_code:
                    print(f"INFO ({__file__}): Received and cleaned code fix from Gemini.", file=sys.stderr)
                    return cleaned_code
                else:
                    print(f"ERROR ({__file__}): Gemini response was empty or cleaned to empty.", file=sys.stderr)

            elif response and getattr(response, 'prompt_feedback', None) and getattr(response.prompt_feedback, 'block_reason', None):
                 block_reason = response.prompt_feedback.block_reason
                 block_message = getattr(response.prompt_feedback, 'block_message', 'No specific message.')
                 print(f"ERROR ({__file__}): Gemini request blocked. Reason: {block_reason}. Message: {block_message}. Prompt likely violated safety settings.", file=sys.stderr)
                 return None
            else:
                response_details = str(response)[:500]
                print(f"ERROR ({__file__}): Unexpected or empty response structure from Gemini: {response_details}", file=sys.stderr)

        except Exception as e:
            print(f"ERROR ({__file__}): Exception during Gemini API call (Attempt {current_attempt}/{retry_attempts}): {e}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            if "rate limit" in str(e).lower() and current_attempt < retry_attempts:
                 print(f"INFO ({__file__}): Rate limit likely hit. Waiting {backoff_time} seconds before retry...", file=sys.stderr)
                 time.sleep(backoff_time)
                 backoff_time *= 2
                 continue
            else:
                break

    print(f"ERROR ({__file__}): Failed to get valid code fix from Gemini after {retry_attempts} attempts.", file=sys.stderr)
    return None