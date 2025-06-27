import os
import json
from flask import Blueprint, session, request, jsonify, current_app as app
import google.generativeai as genai
from constants import FRONTEND_JSON_PATH, EDA_LOGS_FILE_PATH, ML_OUTPUT_LOGS_FILE

# --- Configure Gemini client ---
genai.configure(api_key="AIzaSyBF8Ik7v2Uwy_cRVzoDEj30g2oNpXPPlrQ")
model = genai.GenerativeModel("gemini-2.0-flash")

chat_bp = Blueprint("chatlm", __name__)

def format_suggestion(raw_text):
    """
    Format Gemini response to look clean using markdown-style output.
    """
    raw_text = raw_text.strip()
    if "Problem:" in raw_text or "EDA Findings:" in raw_text:
        lines = raw_text.split("* ")
        formatted = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("**") and line.endswith("**:"):
                # Section heading
                formatted.append(f"\n### 🔹 {line.strip('*: ')}\n")
            elif line.startswith("**") and "**:" in line:
                # Subpoint with bold label
                title, rest = line.split("**:", 1)
                formatted.append(f"**{title.strip('*')}**:{rest.strip()}")
            else:
                # Bullet point
                formatted.append(f"- {line}")
        return "\n".join(formatted)
    else:
        return raw_text

@chat_bp.route("/chatlm", methods=["POST"])
def initialize_chat():
    final_problem = session.get("final_problem")
    if not final_problem:
        return jsonify(error="Business problem not found in session."), 400

    # Load EDA logs
    eda_logs = ""
    if os.path.exists(EDA_LOGS_FILE_PATH):
        with open(EDA_LOGS_FILE_PATH, "r", encoding="utf-8") as f:
            eda_logs = f.read()

    # Load ML logs
    ml_logs = ""
    if os.path.exists(ML_OUTPUT_LOGS_FILE):
        with open(ML_OUTPUT_LOGS_FILE, "r", encoding="utf-8") as f:
            ml_logs = f.read()

    # Load JSON outputs
    frontend_path = FRONTEND_JSON_PATH
    os.makedirs(frontend_path, exist_ok=True)
    json_data = {}
    try:
        for fname in os.listdir(frontend_path):
            if fname.endswith(".json"):
                with open(os.path.join(frontend_path, fname), "r", encoding="utf-8") as f:
                    json_data[fname] = json.load(f)
    except Exception as e:
        app.logger.error("Error reading JSON files: %s", e)
        return jsonify(error=f"Error reading JSON files: {e}"), 500

    system_prompt = (
    "You are a business consultant AI. Based on the provided business problem, EDA logs, "
    "ML logs, and any supporting data outputs, give a very short, clear recommendation focused "
    "only on business action or decision. No technical suggestion. <IMPORTANT> use all the information to give some insight on business and dont talk about datset or ML models. <IMPORTANT>Avoid technical terms, jargon, or lengthy explanation. Focus on solution mainly"
)

    user_content = (
        f"Business problem:\n{final_problem}\n\n"
        f"EDA logs:\n{eda_logs or '[no EDA logs found]'}\n\n"
        f"ML logs:\n{ml_logs or '[no ML logs found]'}\n\n"
        f"Additional data outputs (JSON files):\n{json.dumps(json_data, indent=2)}"
    )

    messages = [
        {"role": "user", "parts": [{"text": f"{system_prompt}\n\n{user_content}"}]}
    ]

    try:
        response = model.generate_content(messages)
        raw_suggestion = response.text.strip()
        suggestion = format_suggestion(raw_suggestion)
    except Exception as e:
        app.logger.error("Gemini API error: %s", e)
        return jsonify(error=f"LLM generation failed: {e}"), 500

    session["conversation"] = messages + [{"role": "model", "parts": [{"text": suggestion}]}]
    return jsonify(status="success", suggestion=suggestion), 200

@chat_bp.route("/chatlm/message", methods=["POST"])
def chat_message():
    data = request.get_json(silent=True) or {}
    user_msg = data.get("user_message", "").strip()
    if not user_msg:
        return jsonify(error="Missing 'user_message' in request body"), 400

    conversation = session.get("conversation")
    if not isinstance(conversation, list):
        return jsonify(error="Chat not initialized. Call /chatlm first."), 400

    conversation.append({"role": "user", "parts": [{"text": user_msg}]})

    try:
        response = model.generate_content(conversation)
        raw_response = response.text.strip()
        assistant_msg = format_suggestion(raw_response)
    except Exception as e:
        app.logger.error("Gemini API error on follow-up: %s", e)
        return jsonify(error=f"LLM generation failed: {e}"), 500

    conversation.append({"role": "model", "parts": [{"text": assistant_msg}]})
    session["conversation"] = conversation
    return jsonify(status="success", response=assistant_msg), 200
