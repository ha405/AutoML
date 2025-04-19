from flask import Flask, render_template, request, session, redirect, url_for
from routes import feedback
from utils import filepreprocess, load_code_from_file, load_logs_from_file
from routes import MachineLearning, data_analysis, AnalysisPlanner
from constants import EDA_CODE_FILE_PATH,EDA_LOGS_FILE_PATH,DATASET_PATH,OUTPUT_PLAN_FILE
import os

import subprocess
import sys
import traceback # Needed for detailed error logging

app = Flask(__name__)
app.secret_key = "super_secret_key_98765"
final_business_problem = None
filedetails = None
# @app.route("/", methods=["GET", "POST"])
# def home():
#     global final_business_problem
#     global filedetails
#     if request.method == "POST":
#         prompt = request.form.get("queryInput")
#         file = request.files.get("file-upload") 
#         if file:
#             print("file")
#         else:
#             print(prompt)
#         filedetails = filepreprocess("E:\AutoML\app\scripts" + file)
#         # print(filedetails)
#         conversation = [f"User: {prompt}", f"System: {feedback.SYSTEM_PROMPT_INITIAL}"]
#         session["conversation"] = conversation
#         final_business_problem = None
#         return redirect(url_for("conversation"))
#     return render_template("index.html")

@app.route("/", methods=["GET", "POST"])
def home():
    global final_business_problem
    global filedetails
    if request.method == "POST":
        prompt = request.form.get("queryInput")
        file = request.files.get("file-upload") 
        filedetails = None

        if file and file.filename: 
            try:
                target_dir = os.path.dirname(DATASET_PATH) 
                os.makedirs(target_dir, exist_ok=True)
                file.save(DATASET_PATH)
                print(f"File '{file.filename}' saved successfully as '{DATASET_PATH}'")
                filedetails = filepreprocess(DATASET_PATH)

            except Exception as e:
                print(f"Error saving or processing uploaded file: {e}")
                filedetails = {"error": f"Failed to save or process file: {str(e)}"}

        elif prompt:
            print(f"Received text prompt only: {prompt}")
            filedetails = {"error": "No file uploaded, proceeding with text prompt only."}
        else:
             return render_template("index.html", error="Please provide a prompt or upload a CSV file.")

        if filedetails and 'error' not in filedetails or (prompt and not file):
             user_input_for_conversation = prompt if prompt else "Analyze uploaded file."
             conversation = [f"User: {user_input_for_conversation}", f"System: {feedback.SYSTEM_PROMPT_INITIAL}"]
             session["conversation"] = conversation
             final_business_problem = None 
             return redirect(url_for("conversation"))
        else:
             error_message = filedetails.get('error', 'An unknown error occurred during file processing.')
             return render_template("index.html", error=error_message)
    return render_template("index.html")


@app.route("/conversation", methods=["GET", "POST"])
def conversation():
    global final_business_problem
    conversation = session.get("conversation", [])
    final_problem = None
    if request.method == "GET":
        if len(conversation) == 2:
            conversation, final_problem = feedback.process_feedback(conversation)
            session["conversation"] = conversation
        if final_problem:
            final_business_problem = final_problem
            return redirect(url_for("next_step"))
        return render_template("conversation.html", conversation=conversation, final_problem=final_problem)
    if request.method == "POST":
        user_response = request.form.get("user_response")
        conversation.append(f"User: {user_response}")
        conversation.append(f"System: {feedback.SYSTEM_PROMPT_NEXT_ITERATION}")
        conversation, final_problem = feedback.process_feedback(conversation)
        session["conversation"] = conversation
        if final_problem:
            final_business_problem = final_problem
            return redirect(url_for("dataanalysis"))
        return redirect(url_for("conversation"))


# @app.route("/dataanalysis", methods=["GET"])
# def dataanalysis():
#     global final_business_problem, filedetails
#     from routes import data_analysis
#     eda_code = data_analysis.generate_and_refine_eda_code(filedetails, final_business_problem, DATASET_PATH)
#     data_analysis.save_code_to_file(eda_code)
#     return redirect(url_for("SuperLLM"))

@app.route("/dataanalysis", methods=["GET"])
def dataanalysis():
    global final_business_problem, filedetails, DATASET_PATH
    filedetails = filepreprocess(DATASET_PATH)
    # print(filedetails)
    # print(final_business_problem)

    if not final_business_problem or not filedetails or 'error' in filedetails:
         print("Error: Missing business problem or valid file details for data analysis.")
         return redirect(url_for('home', error='Missing business problem or file details'))

    print("Generating EDA code...")
    eda_code = data_analysis.generate_data_analysis_code(filedetails, final_business_problem, DATASET_PATH)

    if not eda_code or eda_code.startswith("# Error"):
        print("EDA code generation failed.")

    data_analysis.save_code_to_file(eda_code, os.path.basename(EDA_CODE_FILE_PATH))
    print(f"EDA code saved to {EDA_CODE_FILE_PATH}")
    print(f"Executing generated EDA script: {EDA_CODE_FILE_PATH}")

    logs_content = ""
    try:
        python_executable = sys.executable
        process = subprocess.run(
            [python_executable, EDA_CODE_FILE_PATH],
            capture_output=True, text=True, encoding='utf-8', check=False
        )
        logs_content = f"--- stdout ---\n{process.stdout}\n\n--- stderr ---\n{process.stderr}"
    except Exception as e:
        logs_content = f"# EXECUTION FAILED\n# Error: {e}\n\n{traceback.format_exc()}"
        print(f"ERROR: An exception occurred during EDA script execution: {e}")

    try:
        with open(EDA_LOGS_FILE_PATH, 'w', encoding='utf-8') as log_file:
            log_file.write(logs_content)
        print(f"EDA execution logs saved to: {EDA_LOGS_FILE_PATH}")
    except Exception as e:
        print(f"ERROR: Failed to save EDA logs: {e}")

    print("Redirecting to SuperLLM route...")
    return redirect(url_for("SuperLLM"))

@app.route("/SuperLLM", methods=["GET"])
def SuperLLM():
    global final_business_problem
    global filedetails
    eda_code = load_code_from_file(EDA_CODE_FILE_PATH)
    eda_logs = load_logs_from_file(EDA_LOGS_FILE_PATH) 
    ml_plan_output = AnalysisPlanner.generate_ml_plan(
        business_problem=final_business_problem,
        file_details=filedetails,
        eda_code=eda_code,
        eda_output_logs=eda_logs
    )
    if OUTPUT_PLAN_FILE:
        try:
            with open(OUTPUT_PLAN_FILE, 'w', encoding='utf-8') as f:
                f.write(ml_plan_output)
            print(f"\nML Plan saved to: {OUTPUT_PLAN_FILE}")
        except Exception as e:
            print(f"❌ Error saving ML Plan to file: {e}")
    return redirect(url_for("ml"))

@app.route("/ml", methods=["GET"])
def ml():
    global final_business_problem
    with open(OUTPUT_PLAN_FILE, "r", encoding="utf-8") as plan_file:
        ML_PLAN = plan_file.read()
    try:
        csv_file_path = r"E:\AutoML\processed_sales_data_sample.csv" 
        with open(csv_file_path, "rb") as f:
            new_filedetails = filepreprocess(f)
    except Exception as e:
        new_filedetails = {"error": f"Unable to read processed CSV file: {str(e)}"}
    ml_code = MachineLearning.generate_and_refine_ml_code(final_business_problem,new_filedetails, ML_PLAN)
    MachineLearning.save_ml_code_to_file(ml_code)
    return (
        f"Business Problem for ML (from file): {final_business_problem}<br>"
        f"ML code generated and saved to 'ML.py' in the scripts folder."
    )

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
