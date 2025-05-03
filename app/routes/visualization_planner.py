from groq import Groq
import os

# --- Configuration ---
# WARNING: Hardcoding API keys is generally insecure. Consider environment variables.
API_KEY = "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"
MODEL_NAME = "llama3-70b-8192"

try:
    client = Groq(api_key=API_KEY)
    print("Groq client configured for Visualization Planner.")
except Exception as e:
    print(f"❌ Error configuring Groq client in Visualization Planner: {e}")
    client = None

# --- REVISED PROMPT TEMPLATE FOR SMEs (More Comprehensive but Simple) ---
SYSTEM_INSTRUCTION_VISUALIZATION_PLANNER_TEMPLATE = r"""
You are a data visualization consultant advising a **Small/Medium Enterprise (SME)**. Your target audience is **non-technical business managers**.
Your goal is to create a plan for **4-5 essential, easy-to-understand charts** that provide **clear, actionable insights** directly addressing their business problem. Strive for a balance between providing sufficient context and maintaining simplicity. Avoid overly complex or purely academic visualizations.

*Input Context:*

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

**Revised Instructions for SME Visualization Plan (Balanced Approach):**

1.  **Overall Goal (Business Focus):**
    *   State the primary business objective the visualizations will help achieve (e.g., "Understand key price drivers, model accuracy, and data patterns relevant to pricing strategy").
2.  **Key Actionable Insights & Supporting Visualizations:**
    *   Based on the combined EDA and ML context, identify **3-4 critical findings** that directly impact the business problem.
    *   For each finding, propose **one simple, high-impact chart**. Consider these categories:
        *   **A) Data Exploration Insight (from EDA, relevant to Business Problem):** Visualize a key pattern or relationship discovered during EDA that informs the business problem. (e.g., How does the target variable 'price' distribute? How does 'price' relate to one *key* categorical feature identified in EDA?). *Use: Histogram, Box Plot, Bar Chart.*
        *   **B) Key Drivers (from ML):** Visualize the most important factors influencing the outcome. *Use: Bar Chart (Horizontal preferred for readability).*
        *   **C) Model Performance - Overall Accuracy (from ML):** Visualize how well the model's predictions match the actual values. *Use: Scatter Plot (Actual vs. Predicted).*
        *   **D) Model Performance - Error Analysis (from ML):** Visualize the distribution or pattern of prediction errors (residuals). *Use: Histogram (of residuals).*
        *   **E) Direct Business Question:** If the context allows, visualize something directly answering a part of the business question not covered above (e.g., Average price by 'fuel-type' if that's relevant). *Use: Bar Chart, Box Plot.*
    *   **Prioritize Simplicity:** Strongly favor Bar charts, Line charts (if applicable), Scatter plots, Histograms, and Box plots. Avoid complex plots unless absolutely necessary and simple to interpret.
    *   **Justification (Business Value):** For *each* chart, briefly explain *why* this specific visualization helps the business manager understand that aspect of the problem or model.
3.  **Data to Use:**
    *   Specify the *exact* data columns, model outputs (predictions, feature importances, residuals) needed for each chart.
4.  **LLM Code Guidance (Clarity Focus):**
    *   Recommend Python libraries (matplotlib/seaborn).
    *   **Emphasize clear, business-oriented titles and axis labels.** (e.g., Title: "How Key Features Influence Car Price", X-axis: "Feature", Y-axis: "Impact on Price"). Use layman's terms where possible.
    *   Suggest adding **simple annotations** to highlight the main takeaway message on the chart itself (e.g., an arrow pointing to the most important bar, the R-squared value on the scatter plot, the average error range on the residual histogram).

**Output Format:**
Structure the plan clearly, using Markdown headings for each proposed chart:

### Chart 1: [Simple Chart Type, e.g., Histogram] - Data Insight
**Business Insight:** [e.g., Understanding typical car price range]
**Business Value:** [e.g., Helps set realistic price expectations]
**Data:** [e.g., 'price' column from processed data]
**Clarity Tip:** [e.g., Title: "Distribution of Car Prices"; Add annotation for average price]

### Chart 2: [Simple Chart Type, e.g., Bar Chart] - Key Drivers
**Business Insight:** [e.g., Identifying top factors affecting price]
**Business Value:** [e.g., Shows where to focus design/cost efforts]
**Data:** [e.g., Feature importances from ML model]
**Clarity Tip:** [e.g., Title: "Top Factors Driving Car Price"; Horizontal bars]

### Chart 3: [Simple Chart Type, e.g., Scatter Plot] - Prediction Accuracy
**Business Insight:** [e.g., How well the model predicts prices overall]
**Business Value:** [e.g., Builds confidence in using model predictions]
**Data:** [e.g., Actual 'price' vs. Predicted 'price' from ML test set]
**Clarity Tip:** [e.g., Title: "Model Price Predictions vs. Actual Prices"; Add R-squared value]

### Chart 4: [Simple Chart Type, e.g., Histogram] - Prediction Error Range
**Business Insight:** [e.g., Understanding typical prediction error size]
**Business Value:** [e.g., Shows the typical +/- range for predicted prices]
**Data:** [e.g., Residuals (Actual - Predicted price) from ML test set]
**Clarity Tip:** [e.g., Title: "Typical Range of Price Prediction Errors"; Add annotation for mean/median error]

(Include a 5th chart only if a distinct, crucial insight from category E emerges strongly from the context)

Ensure the final plan focuses on visualizations that tell a coherent story addressing the SME's business problem.
"""
# --- END OF REVISED PROMPT ---

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

    # Prepare paths for the prompt (convert backslashes)
    original_dataset_prompt_path = original_dataset_path_str.replace("\\", "/")
    processed_dataset_prompt_path = processed_dataset_path_str.replace("\\", "/")

    prompt = SYSTEM_INSTRUCTION_VISUALIZATION_PLANNER_TEMPLATE.format(
        business_problem_str=business_problem_str,
        file_details_original_str=file_details_original_str,
        file_details_processed_str=file_details_processed_str,
        original_dataset_path_str=original_dataset_prompt_path,
        processed_dataset_path_str=processed_dataset_prompt_path,
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
        print(f"❌ Error during Groq API call in Visualization Planner: {e}")
        return f"# Error generating visualization plan: {e}"