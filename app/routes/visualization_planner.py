from groq import Groq
import os

# --- Configuration ---
API_KEY = "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"
MODEL_NAME = "llama3-70b-8192"

try:
    client = Groq(api_key=API_KEY)
    print("Groq client configured for Visualization Planner.")
except Exception as e:
    print(f"❌ Error configuring Groq client in Visualization Planner: {e}")
    client = None

SYSTEM_INSTRUCTION_VISUALIZATION_PLANNER_TEMPLATE = r"""
You are a senior data visualization strategist. Your goal is to create a concise plan of **3–5 high-impact charts** that non-technical stakeholders can easily understand.
Combine the detailed guidance below with the key dense instructions provided.

<business_problem>
{business_problem_str}
</business_problem>

<file_details_original>
{file_details_original_str}
</file_details_original>

<file_details_processed>
{file_details_processed_str}
</file_details_processed>

<datasets>
- Original dataset: {original_dataset_path_str}
- Processed dataset: {processed_dataset_path_str}
</datasets>

<eda_context>
Code:
{eda_code_str}

Logs:
{eda_output_logs_str}
</eda_context>

<ml_context>
Code:
{ml_code_str}

Output:
{ml_output_str}
</ml_context>

**Detailed Instructions (retain all):**
1. **Overall Goal of Visualizations:**
   - State the primary objective and what business questions they answer.
2. **Key Insights from ML & EDA:**
   - Identify the most important findings needing visual support.
3. **Recommended Visualizations:**
   - For each insight, choose suitable chart types (bar, line, pie, histogram, box plot, scatter).
   - Provide a brief justification for each.
4. **Data to Use:**
   - Specify exact columns, predictions, residuals, or feature importances.
5. **LLM Code Guidance:**
   - Mention preferred Python libraries (matplotlib/seaborn), styling tips, titles, labels, annotations.

**Output Format:**
Use Markdown headings:

### Chart [#]: [Chart Type]
**Insight:** ...
**Justification:** ...
**Data:** ...
**Tip:** ...
"""

def generate_visualization_plan(
    business_problem_str: str,
    file_details_original_str: str,
    file_details_processed_str: str,
    original_dataset_path_str: str,
    processed_dataset_path_str: str,
    eda_code_str: str,
    eda_output_logs_str: str,
    ml_code_str: str,
    ml_output_str: str
) -> str:
    if not client:
        return "# Error: Groq client not configured."

    prompt = SYSTEM_INSTRUCTION_VISUALIZATION_PLANNER_TEMPLATE.format(
        business_problem_str=business_problem_str,
        file_details_original_str=file_details_original_str,
        file_details_processed_str=file_details_processed_str,
        original_dataset_path_str=original_dataset_path_str,
        processed_dataset_path_str=processed_dataset_path_str,
        eda_code_str=eda_code_str,
        eda_output_logs_str=eda_output_logs_str,
        ml_code_str=ml_code_str,
        ml_output_str=ml_output_str
    )
    messages = [{"role": "user", "content": prompt}]

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.3
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"# Error generating visualization plan: {e}"
