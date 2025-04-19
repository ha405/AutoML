from flask import Flask, request, session, jsonify
from routes import feedback
from utils import filepreprocess, load_code_from_file, load_logs_from_file
from routes import MachineLearning, data_analysis, AnalysisPlanner
from constants import (
    EDA_CODE_FILE_PATH,
    EDA_LOGS_FILE_PATH,
    DATASET_PATH,
    OUTPUT_PLAN_FILE,
    ML_CODE_FILE_PATH # Assumed constant for saving ML code
)
import os
import subprocess
import sys
import traceback

app = Flask(
    __name__,
    static_folder="frontend/build",
    static_url_path=""
)
app.secret_key = "super_secret_key_98765"

final_business_problem = None
filedetails = None


@app.route("/api/home", methods=["POST"])
def api_home():
    global final_business_problem, filedetails
    prompt = request.form.get("queryInput")
    file = request.files.get("file-upload")

    filedetails = None
    error_message = None
    file_saved = False

    if file and file.filename:
        try:
            target_dir = os.path.dirname(DATASET_PATH)
            os.makedirs(target_dir, exist_ok=True)
            file.save(DATASET_PATH)
            print(f"File '{file.filename}' saved successfully as '{DATASET_PATH}'")
            filedetails = filepreprocess(DATASET_PATH)
            file_saved = True
            if filedetails and 'error' in filedetails:
                error_message = filedetails.get('error')
                print(f"Error processing uploaded file: {error_message}")

        except Exception as e:
            print(f"Error saving or processing uploaded file: {e}")
            error_message = f"Failed to save or process file: {str(e)}"
            filedetails = {"error": error_message}

    elif prompt:
        print(f"Received text prompt only: {prompt}")
        if not file_saved:
             filedetails = {"info": "Proceeding with text prompt only as no valid file was provided."}

    else:
        return jsonify({"error": "Please provide a prompt or upload a CSV file."}), 400

    if (filedetails and 'error' not in filedetails) or (prompt and not file_saved):
         user_input_for_conversation = prompt if prompt else "Analyze uploaded file."
         conversation = [f"User: {user_input_for_conversation}", f"System: {feedback.SYSTEM_PROMPT_INITIAL}"]
         session["conversation"] = conversation
         final_business_problem = None
         return jsonify({"conversation": conversation, "filedetails": filedetails if filedetails else {}})
    else:
         error_message = error_message or (filedetails.get('error') if filedetails else 'An unknown error occurred during file processing.')
         return jsonify({"error": error_message}), 400


@app.route("/api/conversation", methods=["GET", "POST"])
def api_conversation():
    global final_business_problem
    conversation = session.get("conversation", [])
    final_problem = None
    status_code = 200

    if not conversation:
         return jsonify({"error": "No active conversation found in session."}), 404

    if request.method == "GET":
        if len(conversation) == 2:
            try:
                conversation, final_problem = feedback.process_feedback(conversation)
                session["conversation"] = conversation
                if final_problem:
                    final_business_problem = final_problem
            except Exception as e:
                print(f"Error processing feedback (GET): {e}")
                return jsonify({"error": "Failed to process initial feedback.", "details": str(e)}), 500

    elif request.method == "POST":
        data = request.get_json()
        if not data or "user_response" not in data:
            return jsonify({"error": "Missing 'user_response' in request body."}), 400

        user_response = data.get("user_response")
        conversation.append(f"User: {user_response}")
        conversation.append(f"System: {feedback.SYSTEM_PROMPT_NEXT_ITERATION}")
        try:
            conversation, final_problem = feedback.process_feedback(conversation)
            session["conversation"] = conversation
            if final_problem:
                final_business_problem = final_problem
        except Exception as e:
            print(f"Error processing feedback (POST): {e}")
            return jsonify({"error": "Failed to process feedback.", "details": str(e)}), 500

    response_data = {
        "conversation": conversation,
        "final_problem": final_business_problem
    }
    if final_problem:
        response_data["final_problem_determined"] = True

    return jsonify(response_data), status_code


@app.route("/api/dataanalysis", methods=["POST"])
def api_dataanalysis():
    global final_business_problem, filedetails, DATASET_PATH

    if os.path.exists(DATASET_PATH):
        try:
            current_filedetails = filepreprocess(DATASET_PATH)
            if 'error' in current_filedetails:
                 print(f"Error re-processing file at {DATASET_PATH}: {current_filedetails['error']}")
                 return jsonify({"error": f"Failed to re-process dataset: {current_filedetails['error']}"}), 500
            filedetails = current_filedetails # Update global
        except Exception as e:
            print(f"Critical error re-processing file at {DATASET_PATH}: {e}")
            return jsonify({"error": f"Failed to access or process dataset file: {str(e)}"}), 500
    else:
        print(f"Error: Dataset file not found at {DATASET_PATH}")
        return jsonify({"error": "Dataset file not found. Please upload again."}), 404

    if not final_business_problem or not filedetails or 'error' in filedetails:
         error_msg = "Missing business problem or valid file details for data analysis."
         print(f"Error: {error_msg}")
         return jsonify({"error": error_msg}), 400

    print("Generating EDA code...")
    eda_code = data_analysis.generate_data_analysis_code(filedetails, final_business_problem, DATASET_PATH)

    if not eda_code or eda_code.startswith("# Error"):
        error_msg = "EDA code generation failed."
        print(error_msg)
        return jsonify({"error": error_msg, "details": eda_code if eda_code else "No code generated."}), 500

    try:
        data_analysis.save_code_to_file(eda_code, EDA_CODE_FILE_PATH)
        print(f"EDA code saved to {EDA_CODE_FILE_PATH}")
    except Exception as e:
        error_msg = f"Failed to save EDA code to {EDA_CODE_FILE_PATH}: {e}"
        print(f"ERROR: {error_msg}")
        return jsonify({"error": error_msg}), 500

    print(f"Executing generated EDA script: {EDA_CODE_FILE_PATH}")
    logs_content = ""
    execution_success = False
    try:
        python_executable = sys.executable
        process = subprocess.run(
            [python_executable, EDA_CODE_FILE_PATH],
            capture_output=True, text=True, encoding='utf-8', check=False
        )
        logs_content = f"--- stdout ---\n{process.stdout}\n\n--- stderr ---\n{process.stderr}"
        if process.returncode == 0:
            execution_success = True
            print("EDA script executed successfully.")
        else:
            print(f"EDA script execution failed with return code {process.returncode}.")

    except Exception as e:
        logs_content = f"# EXECUTION FAILED\n# Error: {e}\n\n{traceback.format_exc()}"
        print(f"ERROR: An exception occurred during EDA script execution: {e}")
        execution_success = False

    try:
        with open(EDA_LOGS_FILE_PATH, 'w', encoding='utf-8') as log_file:
            log_file.write(logs_content)
        print(f"EDA execution logs saved to: {EDA_LOGS_FILE_PATH}")
    except Exception as e:
        print(f"ERROR: Failed to save EDA logs: {e}")

    return jsonify({
        "status": "EDA generated and executed" if execution_success else "EDA generated, execution failed",
        "eda_code_path": EDA_CODE_FILE_PATH,
        "logs_path": EDA_LOGS_FILE_PATH,
        "logs_content": logs_content,
        "execution_successful": execution_success
    })


@app.route("/api/superllm", methods=["POST"])
def api_superllm():
    global final_business_problem, filedetails

    if not final_business_problem:
        return jsonify({"error": "Business problem not defined yet."}), 400

    if not filedetails:
         if os.path.exists(DATASET_PATH):
             try:
                 filedetails = filepreprocess(DATASET_PATH)
                 if 'error' in filedetails:
                     raise ValueError(filedetails['error'])
             except Exception as e:
                 return jsonify({"error": f"Failed to load file details for planning: {str(e)}"}), 500
         else:
              return jsonify({"error": "File details not available for planning."}), 400
    elif 'error' in filedetails:
         return jsonify({"error": f"File details error: {filedetails['error']}"}), 400

    try:
        eda_code = load_code_from_file(EDA_CODE_FILE_PATH)
    except Exception as e:
        return jsonify({"error": f"Failed to load EDA code: {str(e)}"}), 500

    try:
        eda_logs = load_logs_from_file(EDA_LOGS_FILE_PATH)
    except Exception as e:
        return jsonify({"error": f"Failed to load EDA logs: {str(e)}"}), 500

    print("Generating ML plan...")
    try:
        ml_plan = AnalysisPlanner.generate_ml_plan(
            business_problem=final_business_problem,
            file_details=filedetails,
            eda_code=eda_code,
            eda_output_logs=eda_logs
        )
    except Exception as e:
        print(f"Error generating ML plan: {e}")
        return jsonify({"error": "Failed to generate ML plan", "details": str(e)}), 500

    if not ml_plan:
         return jsonify({"error": "ML plan generation resulted in empty plan."}), 500

    try:
        if OUTPUT_PLAN_FILE:
            with open(OUTPUT_PLAN_FILE, 'w', encoding='utf-8') as f:
                f.write(ml_plan)
            print(f"ML plan saved to {OUTPUT_PLAN_FILE}")
        else:
            print("OUTPUT_PLAN_FILE not defined, ML plan not saved to file.")

        return jsonify({
            "status": "ML plan generated",
            "plan_path": OUTPUT_PLAN_FILE if OUTPUT_PLAN_FILE else None,
            "ml_plan": ml_plan
        })
    except Exception as e:
         print(f"Error saving ML plan: {e}")
         return jsonify({"error": "Failed to save ML plan", "details": str(e)}), 500


@app.route("/api/ml", methods=["POST"])
def api_ml():
    global final_business_problem

    if not final_business_problem:
        return jsonify({"error": "Business problem not defined yet."}), 400

    ml_plan = None
    if OUTPUT_PLAN_FILE and os.path.exists(OUTPUT_PLAN_FILE):
        try:
            with open(OUTPUT_PLAN_FILE, 'r', encoding='utf-8') as f:
                ml_plan = f.read()
        except Exception as e:
            return jsonify({"error": f"Failed to load ML plan from {OUTPUT_PLAN_FILE}: {str(e)}"}), 500
    else:
        return jsonify({"error": f"ML plan file not found or path not configured ({OUTPUT_PLAN_FILE})."}), 404

    if not ml_plan:
        return jsonify({"error": "Loaded ML plan is empty."}), 500

    # Define or import the path for processed data if needed by ML generation
    # PROCESSED_DATASET_PATH = constants.PROCESSED_DATASET_PATH
    PROCESSED_DATASET_PATH = r"E:\AutoML\processed_sales_data_sample.csv" # Keep hardcoded path based on original example

    new_filedetails = None
    try:
        if os.path.exists(PROCESSED_DATASET_PATH):
            new_filedetails = filepreprocess(PROCESSED_DATASET_PATH)
            if 'error' in new_filedetails:
                 raise ValueError(new_filedetails['error'])
        else:
            # If processed data isn't strictly needed, maybe use original 'filedetails'?
            # Or assume EDA *should* have created it.
            raise FileNotFoundError(f"Processed dataset not found at {PROCESSED_DATASET_PATH}")
    except Exception as e:
        print(f"Error loading processed file details: {e}")
        # Decide if this is critical. If ML code gen *needs* processed details:
        return jsonify({"error": f"Unable to read or process the processed dataset file needed for ML: {str(e)}"}), 500
        # If it can fallback to original 'filedetails', use that instead. For now, assume it's needed.
        # new_filedetails = filedetails # Example fallback


    print("Generating ML code...")
    try:
        # Use the same function name as in the original script provided by user
        ml_code = MachineLearning.generate_and_refine_ml_code(
            business_problem=final_business_problem,
            file_details=new_filedetails, # Using details from processed data path
            ml_plan=ml_plan
            # Add dataset_path=PROCESSED_DATASET_PATH if the function requires it
        )
    except Exception as e:
         print(f"Error generating ML code: {e}\n{traceback.format_exc()}")
         return jsonify({"error": "Failed to generate ML code", "details": str(e)}), 500

    if not ml_code or ml_code.startswith("# Error"):
         error_msg = "ML code generation failed."
         print(error_msg)
         return jsonify({"error": error_msg, "details": ml_code if ml_code else "No code generated."}), 500

    try:
        MachineLearning.save_ml_code_to_file(ml_code, ML_CODE_FILE_PATH)
        print(f"ML code saved to {ML_CODE_FILE_PATH}")
    except Exception as e:
        error_msg = f"Failed to save ML code to {ML_CODE_FILE_PATH}: {e}"
        print(f"ERROR: {error_msg}")
        return jsonify({"error": error_msg}), 500

    return jsonify({
        "status": "ML code generated",
        "ml_code_path": ML_CODE_FILE_PATH,
        "ml_code": ml_code
    })


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    static_folder = app.static_folder
    if path != "" and os.path.exists(os.path.join(static_folder, path)):
        return app.send_static_file(path)
    else:
        index_path = os.path.join(static_folder, "index.html")
        if os.path.exists(index_path):
            return app.send_static_file("index.html")
        else:
            return jsonify({"error": "React application not found. Build the frontend and ensure 'index.html' is in the 'frontend/build' directory."}), 404


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)