import os
import sys
import subprocess
import time
import json

from flask import Flask, request, session, jsonify, send_from_directory
from flask_session import Session
from routes import feedback, MachineLearning, data_analysis, visualization_planner, visualization_generator, eda_fix, ml_fix, AnalysisPlanner, vision_model
from routes.chatbp import chat_bp
from utils import filepreprocess, load_code_from_file, load_logs_from_file, _serialize
from constants import (
    DATASET_PATH,
    PROCESSED_DATASET_PATH,
    EDA_CODE_FILE_PATH,
    EDA_LOGS_FILE_PATH,
    ML_PLAN,
    EDA_GUIDANCE_PLAN,
    ML_CODE_FILE_PATH,
    ML_OUTPUT_LOGS_FILE,
    VISUALIZATION_PLAN_FILE,
    VISUALIZATION_CODE_FILE_PATH,
    VISUALIZATION_OUTPUT_DIR,
    FRONTEND_JSON_PATH,
    MAX_ATTEMPTS,MAX_FIX_ATTEMPTS,MAX_ML_EXEC_ATTEMPTS,MAX_ML_FIX_ATTEMPTS
)

app = Flask(__name__, static_folder="frontend/build", static_url_path="")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "super_secret_key_98765")

app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = os.path.join(os.getcwd(), ".flask_session")
app.config["SESSION_PERMANENT"] = False
Session(app)


@app.route("/api/home", methods=["POST"])
def api_home():
    prompt = request.form.get("queryInput")
    upload = request.files.get("file-upload")
    session.clear()
    filedetails = {}
    try:
        if upload:
            if not upload.filename.lower().endswith('.csv'):
                 # app.logger.warning(f"Invalid file type uploaded: {upload.filename}") # Removed logging
                 return jsonify(error="Only CSV files are allowed."), 400
            os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
            upload.save(DATASET_PATH)
            filedetails = filepreprocess(DATASET_PATH)
            if filedetails.get("error"):
                # app.logger.error(f"File preprocessing error: {filedetails['error']}") # Removed logging
                return jsonify(error=filedetails["error"]), 400
        elif prompt:
            filedetails = {"info": "Proceeding with text prompt only."}
            if os.path.exists(DATASET_PATH):
                try: os.remove(DATASET_PATH)
                except OSError as e: pass # app.logger.warning(f"Could not remove previous dataset file {DATASET_PATH}: {e}") # Removed logging
        else:
            return jsonify(error="Provide a prompt or a CSV file."), 400
    except Exception as e:
        # app.logger.error(f"Error during file upload/processing: {e}", exc_info=True) # Removed logging
        return jsonify(error="Failed to process input.", details=str(e)), 500

    serialized_filedetails = _serialize(filedetails)
    session["filedetails"] = serialized_filedetails

    initial_context = [
        f"User: {prompt or 'Analyze uploaded file.'}",
        f"System: {feedback.SYSTEM_PROMPT_INITIAL}",
        f"FileDetails: {serialized_filedetails}"
    ]

    try:
        initial_conversation_state, _initial_final_problem = feedback.process_feedback(initial_context)
        session["conversation"] = initial_conversation_state
        session["final_problem"] = None
    except Exception as e:
         session["conversation"] = [
             f"User: {prompt or 'Analyze uploaded file.'}",
             "System: Error generating initial AI response."
         ]
         session["final_problem"] = None
         return jsonify({
             "conversation": session.get("conversation", []),
             "filedetails": session.get("filedetails", {}),
             "final_problem": None,
             "error": "AI failed to generate initial response."
         }), 500

    return jsonify({
        "conversation": session.get("conversation", []),
        "filedetails": session.get("filedetails", {}),
        "final_problem": None
    }), 200


@app.route("/api/conversation", methods=["GET", "POST"])
def api_conversation():
    conversation = session.get("conversation", [])
    final_problem = session.get("final_problem")

    if not conversation:
        return jsonify({"error": "No active conversation found. Please start over."}), 404

    if request.method == "POST":
        data = request.get_json()
        if not data or "user_response" not in data:
            return jsonify({"error": "Missing 'user_response' in request body."}), 400

        user_response = data["user_response"].strip()
        if not user_response:
            return jsonify({"error": "User response cannot be empty."}), 400

        current_conversation = list(session.get("conversation", []))
        current_conversation.append(f"User: {user_response}")

        try:
            updated_conversation, new_final_problem = feedback.process_feedback(current_conversation)

            session["conversation"] = updated_conversation
            if new_final_problem:
                session["final_problem"] = new_final_problem
                final_problem = new_final_problem
            else:
                session.pop("final_problem", None)
                final_problem = None

        except Exception as e:
            return jsonify({
                "error": "AI failed to process your response.",
                "conversation": current_conversation,
                "final_problem": session.get("final_problem"),
                "final_problem_determined": bool(session.get("final_problem"))
            }), 500

    elif request.method == "GET":
        # app.logger.debug(f"GET /api/conversation retrieving state: Conversation length {len(conversation)}, Final problem set: {bool(final_problem)}") # Removed logging
        pass

    return jsonify({
        "conversation": session.get("conversation", []),
        "final_problem": final_problem,
        "final_problem_determined": bool(final_problem)
    }), 200

@app.route("/api/superllm", methods=["POST"])
def api_superllm():
    final_problem = session.get("final_problem")
    nfiledetails = filepreprocess(PROCESSED_DATASET_PATH)
    viz_output_prompt_path = VISUALIZATION_OUTPUT_DIR.replace("\\", "/")
    eda_code = load_code_from_file(EDA_CODE_FILE_PATH)
    plan = AnalysisPlanner.generate_ml_plan(
        business_problem=final_problem,
        file_details=nfiledetails,
        vis_output_dir=viz_output_prompt_path
    )
    if not plan:
        return jsonify(error="ML plan generation failed."), 500
    return jsonify(status="plan_generated")


@app.route("/api/dataanalysis", methods=["POST"])
def api_dataanalysis():
    filedetails = session.get("filedetails")
    final_problem = session.get("final_problem")
    dataset_full_path = os.path.abspath(DATASET_PATH)
    eda_code_full_path = os.path.abspath(EDA_CODE_FILE_PATH)
    eda_guidance_full_path = os.path.abspath(EDA_GUIDANCE_PLAN) 
    viz_output_prompt_path = VISUALIZATION_OUTPUT_DIR.replace("\\", "/")
    if not final_problem or not filedetails:
        app.logger.error("Data analysis request missing problem or file details in session.")
        return jsonify(error="Missing business problem or file details."), 400
    if 'info' not in filedetails and not os.path.exists(dataset_full_path):
         app.logger.error(f"Dataset file specified but not found: {dataset_full_path}")
         return jsonify(error=f"Dataset file not found. Please re-upload."), 400

    eda_guidance = ""
    if os.path.exists(eda_guidance_full_path):
        try:
            with open(eda_guidance_full_path, 'r', encoding='utf-8') as f:
                eda_guidance = f.read()
            app.logger.info(f"Successfully loaded EDA guidance from {eda_guidance_full_path}")
        except Exception as e:
            app.logger.warning(f"Could not read EDA guidance file at {eda_guidance_full_path}: {e}. Proceeding without guidance.")
            eda_guidance = "# Warning: Could not read EDA guidance plan file."
    else:
        app.logger.warning(f"EDA guidance file not found at {eda_guidance_full_path}. Proceeding without guidance.")
        eda_guidance = "# Info: EDA guidance plan file not found."

    os.makedirs(os.path.dirname(eda_code_full_path), exist_ok=True)
    os.makedirs(os.path.dirname(EDA_LOGS_FILE_PATH), exist_ok=True)

    try:
        app.logger.info("Generating initial EDA code with guidance...")
        eda_code = data_analysis.generate_data_analysis_code(
            filedetails=filedetails,
            business_problem=final_problem,
            file_path=dataset_full_path,
            eda_guidance=eda_guidance,
            viz_directory=viz_output_prompt_path,
        )
        if not eda_code or eda_code.strip().startswith("# Error"):
            app.logger.error(f"Initial EDA code generation failed. Response: {eda_code}")
            return jsonify(error="EDA code generation failed.", details=eda_code), 500
        if not data_analysis.save_code_to_file(eda_code, eda_code_full_path):
            app.logger.error(f"Failed to save generated EDA code to {eda_code_full_path}")
            return jsonify(error="Failed to save generated EDA code."), 500
    except TypeError as te:
         if 'eda_guidance' in str(te):
              app.logger.error(f"TypeError: data_analysis.generate_data_analysis_code does not accept 'eda_guidance'. Update the function definition. Error: {te}", exc_info=True)
              return jsonify(error="Internal Server Error: EDA code generator needs update.", details=str(te)), 500
         else:
              app.logger.error(f"TypeError during initial EDA code generation/saving: {te}", exc_info=True)
              return jsonify(error="Failed to generate or save EDA code.", details=str(te)), 500
    except Exception as e:
        app.logger.error(f"Error generating/saving initial EDA code: {e}", exc_info=True)
        return jsonify(error="Failed to generate or save EDA code.", details=str(e)), 500


    attempts = 0
    last_error_output = ""
    current_code = eda_code 

    while attempts < MAX_ATTEMPTS:
        app.logger.info(f"Executing EDA code (attempt #{attempts + 1}/{MAX_ATTEMPTS}). Path: {eda_code_full_path}")
        try:
            result = subprocess.run(
                [sys.executable, eda_code_full_path], capture_output=True, text=True, check=False, timeout=120
            )

            log_content = f"--- EDA Attempt #{attempts + 1} ---\nRC: {result.returncode}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}\n"

            with open(EDA_LOGS_FILE_PATH, 'w' if attempts == 0 else 'a', encoding='utf-8') as log:
                log.write(log_content + "="*20 + "\n")
            session["eda_output_logs"] = log_content 

            if result.returncode == 0:
                app.logger.info(f"EDA code executed successfully on attempt #{attempts + 1}.")
                return jsonify(status="success", output=result.stdout), 200

            last_error_output = result.stderr if result.stderr else result.stdout
            app.logger.warning(f"EDA Attempt #{attempts + 1} failed. RC: {result.returncode}. Error: {last_error_output[:500]}...")
            attempts += 1
            if attempts >= MAX_ATTEMPTS:
                 app.logger.error(f"EDA execution failed after maximum ({MAX_ATTEMPTS}) attempts.")
                 break 

            app.logger.info(f"Attempting EDA code correction (for execution attempt #{attempts})...")
            try:
                fixed_code = eda_fix.attempt_code_fix(
                    failing_code_str=current_code,
                    error_output_str=last_error_output,
                    dataset_path_str=dataset_full_path,
                    business_problem_str=final_problem,
                    file_details_str=str(filedetails),
                    eda_guidance=eda_guidance,
                )
                if fixed_code and fixed_code.strip() and not fixed_code.strip().startswith("# Error"):
                    app.logger.info(f"Fix obtained for attempt #{attempts}. Saving corrected code.")
                    if data_analysis.save_code_to_file(fixed_code, eda_code_full_path):
                         current_code = fixed_code 
                    else:
                        app.logger.error(f"Attempt #{attempts}: Failed to save corrected EDA code. Retrying with previous code.")
                else:
                    app.logger.warning(f"EDA Fix attempt after execution attempt #{attempts} did not yield usable code. Retrying with previous code.")
                    # Loop continues, retrying with the old code

            except TypeError as te_fix:
                 if 'eda_guidance' in str(te_fix):
                      app.logger.error(f"TypeError: eda_fix.attempt_code_fix does not accept 'eda_guidance'. Update the function definition. Error: {te_fix}", exc_info=True)
                      # Cannot fix without guidance potentially, maybe break or just log and continue? Let's log and continue.
                 else:
                      app.logger.error(f"TypeError during EDA code fix attempt #{attempts}: {te_fix}", exc_info=True)
            except Exception as fix_err:
                app.logger.error(f"Error during EDA code fix attempt #{attempts}: {fix_err}", exc_info=True)
                # Loop continues, retrying with the old code

        except subprocess.TimeoutExpired:
            last_error_output = f"EDA script execution timed out after 120s."
            app.logger.error(f"EDA Attempt #{attempts + 1} failed: Timeout") # Log before incrementing attempt counter
            log_content = f"--- EDA Attempt #{attempts + 1} ---\nRC: TIMEOUT\nError: {last_error_output}\n"
            with open(EDA_LOGS_FILE_PATH, 'a', encoding='utf-8') as log: log.write(log_content + "="*20 + "\n")
            session["eda_output_logs"] = log_content
            attempts += 1 
            if attempts >= MAX_ATTEMPTS:
                app.logger.error(f"EDA execution failed due to timeout after maximum ({MAX_ATTEMPTS}) attempts.")
                break

        except Exception as subproc_err:
            last_error_output = f"Subprocess execution error: {str(subproc_err)}"
            app.logger.error(f"EDA Attempt #{attempts + 1} failed: Subprocess error: {subproc_err}", exc_info=True) # Log before incrementing
            log_content = f"--- EDA Attempt #{attempts + 1} ---\nRC: SUBPROCESS_ERROR\nError: {last_error_output}\n"
            with open(EDA_LOGS_FILE_PATH, 'a', encoding='utf-8') as log: log.write(log_content + "="*20 + "\n")
            session["eda_output_logs"] = log_content
            attempts += 1 
            if attempts >= MAX_ATTEMPTS:
                app.logger.error(f"EDA execution failed due to subprocess error after maximum ({MAX_ATTEMPTS}) attempts.")
                break # Exit loop

    app.logger.error(f"EDA code execution failed permanently after {attempts} attempts.")
    return jsonify(error=f"EDA code execution failed after {attempts} attempts.", details=last_error_output), 500


@app.route("/api/ml", methods=["POST"])
def api_ml():
    final_problem = session.get("final_problem")
    ml_plan = None
    processed_dataset_details = None
    viz_output_prompt_path = VISUALIZATION_OUTPUT_DIR.replace("\\", "/")
    processed_dataset_full_path = os.path.abspath(PROCESSED_DATASET_PATH)
    ml_plan_full_path = os.path.abspath(ML_PLAN) 
    ml_code_full_path = os.path.abspath(ML_CODE_FILE_PATH)
    ml_logs_full_path = os.path.abspath(ML_OUTPUT_LOGS_FILE)
    eda_logs_full_path = os.path.abspath(EDA_LOGS_FILE_PATH) 

    if not final_problem:
        app.logger.error("ML execution request failed: Final business problem not found in session.")
        return jsonify(error="Business problem definition unavailable."), 400

    if os.path.exists(ml_plan_full_path):
        try:
            with open(ml_plan_full_path, 'r', encoding='utf-8') as f:
                ml_plan = f.read()
            if not ml_plan:
                 app.logger.error(f"ML Plan file is empty: {ml_plan_full_path}")
                 return jsonify(error="ML Plan is empty."), 500
            app.logger.info("Successfully loaded ML plan.")
            session["ml_plan"] = ml_plan 
        except Exception as e:
            app.logger.error(f"Failed to read ML plan file: {e}", exc_info=True)
            return jsonify(error="Failed to read ML plan file.", details=str(e)), 500
        except Exception as e:
            app.logger.error(f"Failed to read ML plan file: {e}", exc_info=True)
            return jsonify(error="Failed to read ML plan file.", details=str(e)), 500
    else:
        app.logger.error(f"ML execution request failed: ML plan file not found at {ml_plan_full_path}")
        return jsonify(error="No ML plan available. Ensure planning step was successful."), 404

    # Check for processed dataset
    app.logger.info(f"Checking for processed dataset at: {processed_dataset_full_path}")
    if not os.path.exists(processed_dataset_full_path):
         app.logger.error(f"ML execution failed: Processed dataset not found at {processed_dataset_full_path}. Did EDA run and save correctly?")
         return jsonify(error="Processed dataset not found. Please ensure EDA completed successfully."), 400

    # --- Read EDA Output Logs ---
    eda_output_logs = ""
    if os.path.exists(eda_logs_full_path):
        try:
            with open(eda_logs_full_path, 'r', encoding='utf-8') as f:
                eda_output_logs = f.read()
            app.logger.info(f"Successfully loaded EDA logs from {eda_logs_full_path}")
        except Exception as e:
            app.logger.warning(f"Could not read EDA logs file at {eda_logs_full_path}: {e}. Proceeding without EDA log context.")
            eda_output_logs = "# Warning: Could not read EDA logs file."
    else:
        app.logger.warning(f"EDA logs file not found at {eda_logs_full_path}. Proceeding without EDA log context.")
        eda_output_logs = "# Info: EDA logs file not found."

    os.makedirs(os.path.dirname(ml_code_full_path), exist_ok=True)
    os.makedirs(os.path.dirname(ml_logs_full_path), exist_ok=True)

    app.logger.info("Generating initial ML code with EDA log context...")
    try:
        ml_code = MachineLearning.generate_and_refine_ml_code(
            business_problem=final_problem,
            file_path=processed_dataset_full_path,
            ml_plan=ml_plan, # Pass the plan content
            eda_output_logs=eda_output_logs, #,
            viz_directory = viz_output_prompt_path
        )
        # --- End Pass eda_output_logs ---
        if not ml_code or ml_code.strip().startswith("# Error"):
            app.logger.error(f"ML code generation failed. Reason: {ml_code}")
            return jsonify(error="ML code generation failed.", details=ml_code), 500
        if not MachineLearning.save_ml_code_to_file(ml_code, ml_code_full_path):
            app.logger.error(f"Failed to save initial ML code to {ml_code_full_path}")
            return jsonify(error="Failed to save generated ML code."), 500
    except TypeError as te_ml:
        if 'eda_output_logs' in str(te_ml):
            app.logger.error(f"TypeError: MachineLearning.generate_and_refine_ml_code does not accept 'eda_output_logs'. Update the function definition. Error: {te_ml}", exc_info=True)
            return jsonify(error="Internal Server Error: ML code generator needs update.", details=str(te_ml)), 500
        elif 'ml_plan' in str(te_ml) and 'ML_PLAN' in str(te_ml.__traceback__): 
            app.logger.error(f"TypeError: MachineLearning.generate_and_refine_ml_code potentially requires 'ML_PLAN' instead of 'ml_plan', or signature mismatch. Error: {te_ml}", exc_info=True)
            return jsonify(error="Internal Server Error: ML code generator needs update (check plan argument name).", details=str(te_ml)), 500
        else:
            app.logger.error(f"TypeError during initial ML code generation/saving: {te_ml}", exc_info=True)
            return jsonify(error="Failed to generate or save ML code.", details=str(te_ml)), 500
    except Exception as e:
        app.logger.error(f"Error generating/saving initial ML code: {e}", exc_info=True)
        return jsonify(error="Failed to generate or save ML code.", details=str(e)), 500

    app.logger.info(f"Initial ML code saved to {ml_code_full_path}")

    attempts = 0
    last_error_output = "No execution attempts made yet."
    current_code = ml_code

    while attempts < MAX_ML_EXEC_ATTEMPTS:
        app.logger.info(f"--- Starting ML Code Execution Attempt #{attempts + 1}/{MAX_ML_EXEC_ATTEMPTS} ---")
        app.logger.info(f"Running script: {ml_code_full_path}")

        result = None
        try:
            result = subprocess.run(
                [sys.executable, ml_code_full_path],
                capture_output=True, text=True, check=False,
                timeout=300
            )

            log_content = f"--- Execution Attempt #{attempts + 1} ---\nReturn Code: {result.returncode}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}\n" + "="*20 + "\n"
            # Overwrite log on first attempt, append afterwards
            with open(ml_logs_full_path, 'w' if attempts == 0 else 'a', encoding='utf-8') as log:
                log.write(log_content)
            session["ml_output_logs"] = f"Attempt #{attempts + 1}:\n" + result.stdout + "\n" + result.stderr
            app.logger.info(f"Execution attempt #{attempts + 1} completed. Logs saved.")

            if result.returncode == 0:
                app.logger.info(f"ML code executed successfully on attempt #{attempts + 1}.")
                return jsonify(status="success", output=result.stdout, logs_path=ml_logs_full_path), 200

            # --- Execution Failed ---
            app.logger.warning(f"Attempt #{attempts + 1}: ML execution failed (Code: {result.returncode}).")
            last_error_output = result.stderr if result.stderr else result.stdout
            attempts += 1 # Increment attempts only after failure
            if attempts >= MAX_ML_EXEC_ATTEMPTS:
                app.logger.error(f"ML execution failed after maximum ({MAX_ML_EXEC_ATTEMPTS}) attempts.")
                break

            app.logger.info(f"Attempting ML code correction (Exec Attempt #{attempts})...") 
            try:
                fixed_code = ml_fix.attempt_ml_code_fix(
                    broken_ml_code=current_code, # Use code that just failed
                    error_message=last_error_output,
                    dataset_path=processed_dataset_full_path,
                    business_goal=final_problem,
                    ml_plan=ml_plan, # Pass plan
                    eda_output_logs=eda_output_logs 
                )
                print(fixed_code)
                if fixed_code and fixed_code.strip() and not fixed_code.strip().startswith("# Error"):
                     app.logger.info(f"ML code fix obtained for attempt #{attempts}.")
                     if not MachineLearning.save_ml_code_to_file(fixed_code, ml_code_full_path):
                         app.logger.error(f"Attempt #{attempts}: Critical error - failed to save the fixed ML code. Retrying with previous code.")
                     else:
                         app.logger.info(f"Attempt #{attempts}: Fixed ML code saved. Retrying execution.")
                         current_code = fixed_code # Update code for next loop
                else:
                     app.logger.warning(f"ML fix attempt #{attempts} returned no usable code. Retrying with previous code.")
            except TypeError as te_ml_fix:
                 if 'eda_output_logs' in str(te_ml_fix):
                      app.logger.error(f"TypeError: ml_fix.attempt_ml_code_fix does not accept 'eda_output_logs'. Update the function definition. Error: {te_ml_fix}", exc_info=True)
                 elif 'ml_plan' in str(te_ml_fix):
                      app.logger.error(f"TypeError: ml_fix.attempt_ml_code_fix does not accept 'ml_plan'. Update the function definition. Error: {te_ml_fix}", exc_info=True)
                 else:
                      app.logger.error(f"TypeError during ML code fix attempt #{attempts}: {te_ml_fix}", exc_info=True)
                 # Continue loop with old code if fixer signature is wrong
            except Exception as fix_call_err:
                app.logger.error(f"Attempt #{attempts}: Error during call to attempt_ml_code_fix: {fix_call_err}", exc_info=True)
                # Continue loop with old code

        except subprocess.TimeoutExpired:
            error_details = f"Script at {ml_code_full_path} exceeded timeout ({300}s)."
            app.logger.error(f"Attempt #{attempts + 1}: ML script execution timed out.") # Log before increment
            last_error_output = error_details
            log_content = f"--- Execution Attempt #{attempts + 1} ---\nReturn Code: TIMEOUT\nError: {error_details}\n" + "="*20 + "\n"
            with open(ml_logs_full_path, 'a', encoding='utf-8') as log: log.write(log_content)
            session["ml_output_logs"] = f"Attempt #{attempts + 1}: Timeout - {error_details}"
            attempts += 1
            if attempts >= MAX_ML_EXEC_ATTEMPTS:
                 app.logger.error(f"ML code execution failed due to timeout after maximum ({MAX_ML_EXEC_ATTEMPTS}) attempts.")
                 break
            app.logger.info("Continuing to next attempt after timeout...")
            continue

        except Exception as subproc_err:
            error_details = str(subproc_err)
            app.logger.error(f"Attempt #{attempts + 1}: Error running ML subprocess: {subproc_err}", exc_info=True) # Log before increment
            last_error_output = f"Subprocess execution error: {error_details}"
            log_content = f"--- Execution Attempt #{attempts + 1} ---\nReturn Code: SUBPROCESS_ERROR\nError: {error_details}\n" + "="*20 + "\n"
            with open(ml_logs_full_path, 'a', encoding='utf-8') as log: log.write(log_content)
            session["ml_output_logs"] = f"Attempt #{attempts + 1}: Subprocess Error - {error_details}"
            attempts += 1 
            if attempts >= MAX_ML_EXEC_ATTEMPTS:
                 app.logger.error(f"ML execution failed due to subprocess error after maximum ({MAX_ML_EXEC_ATTEMPTS}) attempts.")
                 break
    app.logger.error(f"ML code execution failed permanently after {attempts} attempt(s). Check logs: {ml_logs_full_path}")
    return jsonify(
        error=f"ML code execution failed after {attempts} attempt(s).",
        details=last_error_output
    ), 500

@app.route("/api/vlm", methods=["POST"])
def api_vlm():
    final_problem = session.get("final_problem")
    if not final_problem:
        return jsonify(error="Business problem not found in session."), 400
    viz_output_prompt_path = VISUALIZATION_OUTPUT_DIR.replace("\\", "/")
    success = vision_model.generate_plot_explanations(
        business_problem=final_problem,
        viz_directory=viz_output_prompt_path
    )
    if not success:
        return jsonify(error="Failed to generate plot explanations."), 500
    return jsonify(status="success"), 200

app.register_blueprint(chat_bp, url_prefix="/api")



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