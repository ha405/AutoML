import os
import google.generativeai as genai

GOOGLE_API_KEY = "AIzaSyBoAFOxBSX1nxEF8lNuhJudPiHVTCRNK8Q"

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

SYSTEM_TEMPLATE = r"""
You are a Senior Data Scientist acting as an ML Strategist.
Analyze the results of an Exploratory Data Analysis (EDA) and produce a structured plan for the ML modeling phase.

<business_problem>
{business_problem_str}
</business_problem>

<dataset_details>
{file_details_str}
</dataset_details>

<eda_code>
{eda_code_str}
</eda_code>

<eda_logs>
{eda_output_logs_str}
</eda_logs>

Output as Markdown with these sections:
- EDA Summary
- Target Variable
- ML Task Type
- Recommended Model Types
- Next Step Considerations
"""

def _call_model(prompt: str) -> str:
    try:
        config = genai.types.GenerationConfig(temperature=0.2)
        resp = model.generate_content(contents=prompt, generation_config=config)
        if resp.prompt_feedback and resp.prompt_feedback.block_reason:
            return f"# Error: Blocked ({resp.prompt_feedback.block_reason})"
        return getattr(resp, "text", "# Error: No content generated").strip()
    except Exception as e:
        return f"# Error: {e}"


def generate_ml_plan(
    business_problem: str,
    file_details: dict,
    eda_code: str,
    eda_output_logs: str
) -> str:
    # Limit sample data to reduce prompt length
    details_copy = file_details.copy()
    if "sample_data" in details_copy:
        details_copy["sample_data"] = details_copy["sample_data"][:3]
    file_details_str = "\n".join(f"- {k}: {v}" for k, v in details_copy.items())

    prompt = SYSTEM_TEMPLATE.format(
        business_problem_str=business_problem,
        file_details_str=file_details_str,
        eda_code_str=eda_code,
        eda_output_logs_str=eda_output_logs
    )
    return _call_model(prompt)
