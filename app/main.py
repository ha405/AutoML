from flask import (
    Flask, request, session, redirect,
    url_for, jsonify
)
from routes import feedback, MachineLearning, data_analysis, AnalysisPlanner
from utils import filepreprocess, load_code_from_file, load_logs_from_file
from constants import (
    EDA_CODE_FILE_PATH,
    EDA_LOGS_FILE_PATH,
    OUTPUT_PLAN_FILE
)
import os

app = Flask(
    __name__,
    static_folder="frontend/build",
    static_url_path=""  # serve React build at root
)
app.secret_key = "super_secret_key_98765"

# Globals
final_business_problem = None
filedetails = None


@app.route("/api/home", methods=["POST"])
def api_home():
    global final_business_problem, filedetails
    # React sends FormData, so use request.form + request.files
    prompt = request.form.get("queryInput")
    file = request.files.get("file")

    if file:
        filedetails = filepreprocess(file)
    else:
        filedetails = None

    conversation = [
        f"User: {prompt}",
        f"System: {feedback.SYSTEM_PROMPT_INITIAL}"
    ]
    session["conversation"] = conversation
    final_business_problem = None
    return jsonify({"conversation": conversation})


@app.route("/api/conversation", methods=["GET", "POST"])
def api_conversation():
    global final_business_problem
    conversation = session.get("conversation", [])
    final_problem = None

    if request.method == "GET":
        if len(conversation) == 2:
            conversation, final_problem = feedback.process_feedback(conversation)
            session["conversation"] = conversation

    else:
        data = request.get_json()
        user_response = data.get("user_response")
        conversation.append(f"User: {user_response}")
        conversation.append(f"System: {feedback.SYSTEM_PROMPT_NEXT_ITERATION}")
        conversation, final_problem = feedback.process_feedback(conversation)
        session["conversation"] = conversation

    return jsonify({
        "conversation": conversation,
        "final_problem": final_problem
    })


@app.route("/api/dataanalysis", methods=["GET"])
def api_dataanalysis():
    global final_business_problem, filedetails
    eda_code = data_analysis.generate_data_analysis_code(filedetails, final_business_problem)
    data_analysis.save_code_to_file(eda_code)
    return jsonify({"status": "EDA generated"})


@app.route("/api/superllm", methods=["GET"])
def api_superllm():
    global final_business_problem, filedetails
    eda_code = load_code_from_file(EDA_CODE_FILE_PATH)
    eda_logs = load_logs_from_file(EDA_LOGS_FILE_PATH)

    ml_plan = AnalysisPlanner.generate_ml_plan(
        business_problem=final_business_problem,
        file_details=filedetails,
        eda_code=eda_code,
        eda_output_logs=eda_logs
    )
    if OUTPUT_PLAN_FILE:
        with open(OUTPUT_PLAN_FILE, 'w', encoding='utf-8') as f:
            f.write(ml_plan)
    return jsonify({"status": "ML plan generated"})


@app.route("/api/ml", methods=["GET"])
def api_ml():
    global final_business_problem
    # load plan and run ML code generator
    with open(OUTPUT_PLAN_FILE, 'r', encoding='utf-8') as f:
        ml_plan = f.read()
    try:
        with open(r"E:\AutoML\processed_sales_data_sample.csv", "rb") as f:
            new_filedetails = filepreprocess(f)
    except Exception as e:
        new_filedetails = {"error": str(e)}

    ml_code = MachineLearning.generate_ml_code(
        final_business_problem, new_filedetails, ml_plan
    )
    MachineLearning.save_ml_code_to_file(ml_code)
    return jsonify({"status": "ML code generated"})


# Serve React App
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return app.send_static_file(path)
    return app.send_static_file("index.html")


if __name__ == "__main__":
    app.run(debug=True)