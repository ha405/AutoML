from flask import Flask, render_template, request, session, redirect, url_for
from routes import feedback, data_analysis, MachineLearning
from utils import filepreprocess

app = Flask(__name__)
app.secret_key = "super_secret_key_98765"

final_business_problem = None
filedetails = None

@app.route("/", methods=["GET", "POST"])
def home():
    global final_business_problem, filedetails
    if request.method == "POST":
        prompt = request.form.get("queryInput")
        file = request.files.get("file-upload")
        print(file)
        if file:
            filedetails = filepreprocess(file)
        else:
            print(f"User Prompt: {prompt}")

        conversation = [
            f"User: {prompt}",
            f"File Details: {filedetails}",
            f"System: {feedback.SYSTEM_PROMPT_INITIAL}"
        ]
        
        session["conversation"] = conversation
        final_business_problem = None
        
        return redirect(url_for("conversation"))
    
    return render_template("index.html")

@app.route("/conversation", methods=["GET", "POST"])
def conversation():
    global final_business_problem
    conversation = session.get("conversation", [])
    final_problem = None

    if request.method == "GET":
        if len(conversation) == 3:
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
            return redirect(url_for("next_step"))
        return redirect(url_for("conversation"))

@app.route("/next", methods=["GET"])
def next_step():
    global final_business_problem, filedetails
    # LLM -> ML Model -> Selection -> Context;
    eda_code = data_analysis.generate_data_analysis_code(filedetails, final_business_problem)
    data_analysis.save_code_to_file(eda_code)
    return redirect(url_for("ml"))

@app.route("/ml", methods=["GET"])
def ml():
    global final_business_problem
    try:
        with open(r"E:\AutoML\processed_sales_data_sample.csv", "rb") as f:
            new_filedetails = filepreprocess(f)
    except Exception as e:
        new_filedetails = {"error": f"Unable to read processed CSV file: {str(e)}"}
    
    ml_code = MachineLearning.generate_ml_code(final_business_problem, new_filedetails)
    MachineLearning.save_ml_code_to_file(ml_code)
    
    return (
        f"Final Business Problem for ML: {final_business_problem}<br>"
        f"ML code generated and saved to 'ML.py' in the scripts folder."
    )

if __name__ == "__main__":
    app.run(debug=True)
