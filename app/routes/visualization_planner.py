from groq import Groq
import os

# --- Configuration ---
API_KEY = "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" 
MODEL_NAME = "llama3-70b-8192"

os.environ["GROQ_API_KEY"] = API_KEY

try:
    client = Groq()
    print("Groq client configured for Visualization Planner.")
except Exception as e:
    print(f"❌ Error configuring Groq client in Visualization Planner: {e}")
    client = None

SYSTEM_INSTRUCTION_VISUALIZATION_PLANNER_TEMPLATE = r"""
You are an expert data visualization strategist. Your task is to create a detailed plan for generating effective visualizations based on a machine learning process. This plan will guide a subsequent AI in creating the actual visualization code.

*Input Context:*

<business_problem>
{business_problem_str}
</business_problem>

<file_details>
{file_details_str}
</file_details>

<ml_code>
{ml_code_str}
</ml_code>

<ml_output>
{ml_output_str}
</ml_output>

*Instructions for Visualization Plan Generation:*

Based on the provided context, create a comprehensive plan for generating visualizations. The plan should include the following sections, clearly structured with Markdown headings:

1.  *Overall Goal of Visualizations:*
    *   State the primary objective of the visualizations. What business questions should they answer? How should they help understand the ML results in the context of the <business_problem>?

2.  *Key Insights from ML Output to Visualize:*
    *   Analyze the <ml_output> and identify the most important findings and metrics that need to be visualized.  For example, if it's classification, mention accuracy, F1-score, confusion matrix results, feature importances. If regression, mention R-squared, MSE, feature importances, etc.
    *   List specific insights that visualizations should highlight (e.g., "Visualize the distribution of predicted probabilities for each class", "Show feature importances to understand which factors are most influential").

3.  *Recommended Visualizations (with Justification):*
    *   For each key insight identified in section 2, recommend one or more specific visualization types that would be effective. For each recommendation, provide a brief justification explaining why this visualization type is suitable for the insight.
    *   Consider a variety of visualization types relevant to the ML task (classification or regression) and data characteristics. Examples:
        *   *For understanding data distributions:* Histograms, Box Plots, Density Plots.
        *   *For relationships between variables:* Scatter Plots, Pair Plots, Heatmaps (correlation).
        *   *For categorical data:* Bar Charts, Pie Charts.
        *   *For ML Model Performance (Classification):* Confusion Matrix Heatmap, ROC Curve, Precision-Recall Curve, Bar chart of evaluation metrics.
        *   *For ML Model Performance (Regression):* Scatter plot of predicted vs. actual values, Residual plots, Line plot of evaluation metrics.
        *   *For Feature Importance:* Bar chart of feature importances (if available from ML model).

4.  *Data to be Used for Each Visualization:*
    *   For each recommended visualization, specify exactly which data should be used to create it. Be precise about the columns, variables, or dataframes needed.
    *   Consider: Original dataset, processed dataset (if applicable), model predictions, model residuals, feature importance scores, etc.

5.  *Specific Instructions for Visualization Generation LLM:*
    *   Provide any specific instructions or preferences for the LLM that will generate the visualization code. This might include:
        *   Preferred Python libraries for visualization (e.g., 'Use matplotlib.pyplot and seaborn for static plots', 'Consider using plotly for interactive plots if appropriate').
        *   Desired color palettes or styling themes.
        *   Specific aspects to emphasize or highlight in the visualizations.
        *   Guidance on plot titles, labels, and annotations to ensure clarity and business relevance.
        *   Mention if interactive elements are desired (e.g., tooltips in plotly).

*Output Format:*

Structure your plan clearly using Markdown headings for each section as described above.  For each recommended visualization in section 3, use a consistent format like:

### Visualization [Number]: [Visualization Type]
*Insight to Visualize:* [State the insight this visualization will reveal]
*Justification:* [Explain why this visualization type is suitable]
*Data to Use:* [Specify the data and columns needed]

Ensure the plan is detailed, actionable, and provides sufficient guidance for an AI to generate relevant and insightful visualizations.
"""

def generate_visualization_plan(business_problem_str: str,
                                 file_details_str: str,
                                 ml_code_str: str,
                                 ml_output_str: str) -> str:
    """
    Generates a visualization plan based on the business problem,
    file details, ML code, and ML output using the Groq model.
    """
    if not client:
        return "# Error: Groq client not configured."

    prompt = SYSTEM_INSTRUCTION_VISUALIZATION_PLANNER_TEMPLATE.format(
        business_problem_str=business_problem_str,
        file_details_str=file_details_str,
        ml_code_str=ml_code_str,
        ml_output_str=ml_output_str
    )
    messages = [{"role": "user", "content": prompt}]

    try:
        print("Sending request to Groq for visualization plan generation...")
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.4, # Slightly higher temperature for more creative planning
        )
        visualization_plan_text = completion.choices[0].message.content or ""
        print("Visualization plan generated.")
        return visualization_plan_text.strip()
    except Exception as e:
        error_message = f"# Error generating visualization plan: {e}"
        print(error_message)
        return error_message