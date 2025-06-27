import google.generativeai as genai
import os
import sys
import re        # Keep re import
import time      # Keep time import
import base64
import json
from PIL import Image

# --- Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyBF8Ik7v2Uwy_cRVzoDEj30g2oNpXPPlrQ")
MODEL_NAME     = "gemini-2.0-flash"

# Configure the GenAI client
try:
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_AI_API_KEY":
        print("⚠ Warning: Google API Key not set or is placeholder.", file=sys.stderr)
    else:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)
        print(f"Google AI client configured for VLM using model {MODEL_NAME}.")
except Exception as e:
    print(f"❌ Error configuring Google AI client: {e}", file=sys.stderr)
    sys.exit(1)

def _make_inline_part(path: str, mime: str = "image/png") -> dict:
    """
    Read the file at `path`, Base64-encode it, and wrap in an inline_data blob
    that matches the Gemini Part shape.
    """
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return {
        "inline_data": {
            "mime_type": mime,
            "data": b64
        }
    }

def generate_plot_explanations(business_problem: str, viz_directory: str, max_images: int = None) -> bool:
    """
    Uses Gemini Vision capabilities to generate 2–3 sentence, non-technical explanations
    for each PNG plot in `viz_directory` that is directly relevant to `business_problem`.
    Stores explanations in a JSON file mapping image filenames to their explanations.
    """
    # 1. Gather PNG images
    image_files = [fn for fn in os.listdir(viz_directory) if fn.lower().endswith(".png")]
    if max_images:
        image_files = image_files[:max_images]
    if not image_files:
        print(f"No PNG images found in {viz_directory}", file=sys.stderr)
        return False

    # 2. Build the shared prompt
    prompt_lines = [
        "You are a business analyst with expertise in translating charts into clear insights.",
        f"Business objective: '{business_problem}'.",
        "Below is a chart produced during data analysis.",
        "Provide a 2–3 sentence, non-technical explanation that includes:",
        "  1. What the chart shows.",
        "  2. Why it matters for the objective.",
        "  3. The key takeaway a non-technical stakeholder should know.",
        "If the chart is not relevant, respond with an empty JSON `{}`.",
        "Respond with a single JSON object mapping the image filename to its explanation."
    ]
    prompt = "\n".join(prompt_lines)

    explanations = {}

    # 3. For each image: wrap it, send to Gemini, parse JSON
    for filename in image_files:
        path = os.path.join(viz_directory, filename)
        try:
            part = _make_inline_part(path, mime="image/png")
            response = model.generate_content([part, prompt])
            raw_text = response.text.strip()

            try:
                obj = json.loads(raw_text)
                if isinstance(obj, dict):
                    explanations.update(obj)
                else:
                    explanations[filename] = raw_text
            except json.JSONDecodeError:
                explanations[filename] = raw_text

        except Exception as e:
            print(f"Error processing {filename}: {e}", file=sys.stderr)

    if not explanations:
        print("No explanations were generated", file=sys.stderr)
        return False

    # 4. Save explanations to JSON
    output_path = os.path.join(viz_directory, "plot_explanations.json")
    try:
        with open(output_path, "w", encoding="utf-8") as out_file:
            json.dump(explanations, out_file, indent=2)
        print(f"Saved plot explanations to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving explanations: {e}", file=sys.stderr)
        return False
