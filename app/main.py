import os
import sys
import subprocess

from flask import Flask, request, session, jsonify, send_from_directory
from routes import feedback, MachineLearning, data_analysis, AnalysisPlanner, visualization_planner, visualization_generator, eda_fix
from utils import filepreprocess, load_code_from_file, load_logs_from_file
from constants import (
    DATASET_PATH,
    PROCESSED_DATASET_PATH,
    EDA_CODE_FILE_PATH,
    EDA_LOGS_FILE_PATH,
    OUTPUT_PLAN_FILE,
    ML_CODE_FILE_PATH,
    ML_OUTPUT_LOGS_FILE,
    VISUALIZATION_PLAN_FILE,
    VISUALIZATION_CODE_FILE_PATH,
    VISUALIZATION_OUTPUT_DIR
)

app = Flask(__name__, static_folder="frontend/build", static_url_path="")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "super_secret_key_98765")

@app.route("/api/home", methods=["POST"])
def api_home():
    prompt = request.form.get("queryInput")
    upload = request.files.get("file-upload")

    if upload and upload.filename:
        os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
        upload.save(DATASET_PATH)
        filedetails = filepreprocess(DATASET_PATH)
        if filedetails.get("error"):
            return jsonify(error=filedetails["error"]), 400
    elif prompt:
        filedetails = {"info": "Proceeding with text prompt only."}
    else:
        return jsonify(error="Provide a prompt or a CSV file."), 400

    # Initialize conversation
    session["filedetails"] = filedetails
    session["conversation"] = [
        f"User: {prompt or 'Analyze uploaded file.'}",
        f"System: {feedback.SYSTEM_PROMPT_INITIAL}",
        f"FileDetails: {filedetails}"
    ]
    session.pop("final_problem", None)
    try:
        conversation = session["conversation"]
        conversation, final_problem = feedback.process_feedback(conversation)
        session["conversation"] = conversation
        if final_problem:
            session["final_problem"] = final_problem
    except Exception as e:
        return jsonify(error="Failed to process initial feedback.", details=str(e)), 500

    return jsonify({
        "conversation": session["conversation"],
        "filedetails": filedetails,
        "final_problem": session.get("final_problem")
    }), 200

@app.route("/api/conversation", methods=["GET", "POST"])
def api_conversation():
    conversation = session.get("conversation", [])
    final_problem = session.get("final_problem")
    status_code = 200

    if not conversation:
        return jsonify({"error": "No active conversation found in session."}), 404

    if request.method == "GET":
        # Assume GET is only called for refreshing conversation or on first render
        if not final_problem:
            try:
                conversation, final_problem = feedback.process_feedback(conversation)
                session["conversation"] = conversation
                if final_problem:
                    session["final_problem"] = final_problem
            except Exception as e:
                return jsonify({"error": "Failed to process feedback.", "details": str(e)}), 500

    elif request.method == "POST":
        data = request.get_json()
        if not data or "user_response" not in data:
            return jsonify({"error": "Missing 'user_response' in request body."}), 400

        user_response = data["user_response"]
        conversation.append(f"User: {user_response}")
        conversation.append(f"System: {feedback.SYSTEM_PROMPT_NEXT_ITERATION}")

        try:
            conversation, final_problem = feedback.process_feedback(conversation)
            session["conversation"] = conversation
            if final_problem:
                session["final_problem"] = final_problem
        except Exception as e:
            return jsonify({"error": "Failed to process feedback.", "details": str(e)}), 500

    return jsonify({
        "conversation": conversation,
        "final_problem": final_problem,
        "final_problem_determined": bool(final_problem)
    }), status_code

MAX_ATTEMPTS = 5  # Maximum number of execution attempts
MAX_FIX_ATTEMPTS = 3  # Maximum number of fix attempts per execution failure

@app.route("/api/dataanalysis", methods=["POST"])
def api_dataanalysis():
    filedetails = session.get("filedetails")
    final_problem = session.get("final_problem")

    dataset_full_path = os.path.abspath(DATASET_PATH)
    eda_code_full_path = os.path.abspath(EDA_CODE_FILE_PATH)

    # Validate inputs
    if not os.path.exists(dataset_full_path):
        app.logger.error(f"Dataset not found at specified path: {dataset_full_path}")
        return jsonify(error=f"Dataset not found at specified path: {dataset_full_path}"), 400
    if not final_problem or not filedetails:
        app.logger.error("Missing business problem or file details in session.")
        return jsonify(error="Missing business problem or file details."), 400

    attempts = 0

    # Generate and save initial EDA code
    app.logger.info("Generating initial EDA code...")
    eda_code = data_analysis.generate_data_analysis_code(
        filedetails, final_problem, dataset_full_path
    )
    if not eda_code or eda_code.startswith("# Error"):
        app.logger.error("Initial EDA code generation failed.")
        return jsonify(error="EDA code generation failed."), 500

    os.makedirs(os.path.dirname(eda_code_full_path), exist_ok=True)
    app.logger.info(f"Saving initial EDA code to: {eda_code_full_path}")
    if not data_analysis.save_code_to_file(eda_code, eda_code_full_path):
        app.logger.error(f"Failed to save generated code to {eda_code_full_path}")
        return jsonify(error="Failed to save generated EDA code."), 500

    # Execute and auto-fix loop
    while attempts < MAX_ATTEMPTS:
        app.logger.info(f"Executing EDA code (attempt #{attempts + 1}/{MAX_ATTEMPTS}). Path: {eda_code_full_path}")
        try:
            result = subprocess.run(
                [sys.executable, eda_code_full_path],
                capture_output=True,
                text=True,
                check=False,
                timeout=120
            )
        except subprocess.TimeoutExpired:
            app.logger.error(f"Attempt #{attempts + 1}: EDA script execution timed out.")
            return jsonify(
                error="Code execution timed out.",
                details=f"Script at {eda_code_full_path} exceeded timeout."
            ), 500
        except Exception as subproc_err:
            app.logger.error(f"Attempt #{attempts + 1}: Error running subprocess: {subproc_err}")
            return jsonify(
                error="Failed to execute generated code via subprocess.",
                details=str(subproc_err)
            ), 500

        if result.returncode == 0:
            app.logger.info(f"EDA code executed successfully after {attempts} fix attempt(s).")
            app.logger.debug(f"STDOUT:\n{result.stdout}")
            return jsonify(status="success", output=result.stdout), 200

        # Execution failed
        attempts += 1
        app.logger.warning(f"Attempt #{attempts}: Execution failed with return code {result.returncode}.")
        app.logger.debug(f"Failed STDOUT:\n{result.stdout}")
        app.logger.debug(f"Failed STDERR:\n{result.stderr}")

        if attempts >= MAX_ATTEMPTS:
            app.logger.error(f"Execution failed after maximum ({MAX_ATTEMPTS}) attempts.")
            return jsonify(
                error=f"Code execution failed after {MAX_ATTEMPTS} attempts.",
                details=result.stderr
            ), 500

        # Attempt to fix the code with retries
        fix_attempts = 0
        fixed_code = None
        while fix_attempts < MAX_FIX_ATTEMPTS:
            try:
                with open(eda_code_full_path, "r", encoding='utf-8') as f:
                    prev_code = f.read()
                prev_err = result.stderr if result.stderr else result.stdout

                app.logger.info(f"Attempting code correction (attempt #{attempts}, fix attempt #{fix_attempts + 1})...")
                fixed_code = eda_fix.attempt_code_fix(
                    prev_code,
                    prev_err,
                    dataset_full_path,
                    final_problem,
                    str(filedetails)
                )
                if fixed_code:
                    break
                else:
                    fix_attempts += 1
                    app.logger.warning(f"Fix attempt {fix_attempts} returned no code.")
            except Exception as fix_err:
                fix_attempts += 1
                app.logger.error(f"Fix attempt {fix_attempts} failed with error: {fix_err}")
                time.sleep(5)  # Wait 5 seconds before retrying

        if not fixed_code:
            app.logger.error(f"Failed to obtain a fix after {MAX_FIX_ATTEMPTS} attempts for execution attempt #{attempts}.")
            # Continue to next attempt instead of stopping
            continue

        # Overwrite with fixed code and retry
        try:
            with open(eda_code_full_path, "w", encoding='utf-8') as f:
                f.write(fixed_code)
            app.logger.info(f"Attempt #{attempts}: Fixed code written to {eda_code_full_path}. Retrying execution.")
        except Exception as write_err:
            app.logger.error(f"Attempt #{attempts}: Failed to write fixed code to {eda_code_full_path}: {write_err}")
            return jsonify(
                error="Failed to save the corrected code.",
                details=str(write_err)
            ), 500

    app.logger.error(f"Code execution failed after maximum ({MAX_ATTEMPTS}) attempts.")
    return jsonify(error=f"Code execution failed after {MAX_ATTEMPTS} attempts."), 500

@app.route("/api/superllm", methods=["POST"])
def api_superllm():
    final_problem = session.get("final_problem")
    filedetails = session.get("filedetails")
    if not final_problem or not filedetails:
        return jsonify(error="Business problem or file details unavailable."), 400

    eda_code = load_code_from_file(EDA_CODE_FILE_PATH)
    plan = AnalysisPlanner.generate_ml_plan(
        business_problem=final_problem,
        file_details=filedetails,
        eda_code=eda_code,
        eda_output_logs=EDA_LOGS_FILE_PATH
    )
    if not plan:
        return jsonify(error="ML plan generation failed."), 500
    with open(OUTPUT_PLAN_FILE, "w") as f:
        f.write(plan)
    return jsonify(status="plan_generated")

@app.route("/api/ml", methods=["POST"])
def api_ml():
    final_problem = session.get("final_problem")
    if not final_problem or not os.path.exists(OUTPUT_PLAN_FILE):
        return jsonify(error="No ML plan available."), 400

    new_details = filepreprocess(PROCESSED_DATASET_PATH)
    if new_details.get("error"):
        return jsonify(error=new_details["error"]), 500

    plan = open(OUTPUT_PLAN_FILE).read()
    ml_code = MachineLearning.generate_and_refine_ml_code(
        business_problem=final_problem,
        file_path=DATASET_PATH,
        ML_PLAN=plan
    )
    if not ml_code or ml_code.startswith("# Error"):
        return jsonify(error="ML code generation failed."), 500
    MachineLearning.save_ml_code_to_file(ml_code, ML_CODE_FILE_PATH)

    result = subprocess.run(
        [sys.executable, ML_CODE_FILE_PATH],
        capture_output=True, text=True
    )
    os.makedirs(os.path.dirname(ML_OUTPUT_LOGS_FILE), exist_ok=True)
    with open(ML_OUTPUT_LOGS_FILE, 'w') as log:
        log.write(f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}")
    session["ml_output_logs"] = result.stdout + result.stderr
    status = "success" if result.returncode == 0 else "error"
    return jsonify(status=status)

@app.route("/api/visualizationplanning", methods=["POST"])
def api_visualization_planning():
    final_problem = session.get("final_problem")
    filedetails = session.get("filedetails")
    ml_code = load_code_from_file(ML_CODE_FILE_PATH)
    ml_logs = open(ML_OUTPUT_LOGS_FILE).read() if os.path.exists(ML_OUTPUT_LOGS_FILE) else session.get("ml_output_logs", "")
    if not final_problem or not filedetails or not ml_code:
        return jsonify(error="Missing inputs for visualization planning."), 400

    details_str = "\n".join(f"- {k}: {v}" for k, v in filedetails.items() if k != 'sample_data')
    plan = visualization_planner.generate_visualization_plan(
        business_problem_str=final_problem,
        file_details_str=details_str,
        ml_code_str=ml_code,
        ml_output_str=ml_logs
    )
    if not plan or plan.startswith("# Error"):
        return jsonify(error="Visualization plan generation failed."), 500
    os.makedirs(os.path.dirname(VISUALIZATION_PLAN_FILE), exist_ok=True)
    with open(VISUALIZATION_PLAN_FILE, 'w') as f:
        f.write(plan)
    session["visualization_plan"] = plan
    return jsonify(status="plan_generated")

@app.route("/api/visualizations", methods=["POST"])
def api_visualizations():
    final_problem = session.get("final_problem")
    plan = open(VISUALIZATION_PLAN_FILE).read() if os.path.exists(VISUALIZATION_PLAN_FILE) else session.get("visualization_plan", "")
    if not final_problem or not plan:
        return jsonify(error="Visualization plan unavailable."), 400

    viz_code = visualization_generator.generate_visualization_code(
        visualization_plan_str=plan,
        processed_data_path_str=DATASET_PATH,
        business_problem_str=final_problem,
        visualization_output_dir_str=VISUALIZATION_OUTPUT_DIR
    )
    if not viz_code or viz_code.startswith("# Error"):
        return jsonify(error="Visualization code generation failed."), 500
    visualization_generator.save_visualization_code(viz_code, VISUALIZATION_CODE_FILE_PATH)

    res = subprocess.run(
        [sys.executable, VISUALIZATION_CODE_FILE_PATH],
        capture_output=True, text=True
    )
    status = "success" if res.returncode == 0 else "error"
    return jsonify(status=status)

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    full_path = os.path.join(app.static_folder, path)
    if path and os.path.exists(full_path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
