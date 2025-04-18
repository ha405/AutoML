from flask import Flask, render_template, request, session, redirect, url_for
from routes import feedback
from utils import filepreprocess

app = Flask(__name__)
app.secret_key = "super_secret_key_98765"
final_business_problem = None
filedetails = None
@app.route("/", methods=["GET", "POST"])
def home():
    global final_business_problem
    global filedetails
    if request.method == "POST":
        prompt = request.form.get("queryInput")
        file = request.files.get("file-upload") 
        if file:
            print("file")
        else:
            print(prompt)
        filedetails = filepreprocess(file)
        # print(filedetails)
        conversation = [f"User: {prompt}", f"System: {feedback.SYSTEM_PROMPT_INITIAL}"]
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
            return redirect(url_for("next_step"))
        return redirect(url_for("conversation"))


@app.route("/next", methods=["GET"])
def next_step():
    global final_business_problem, filedetails
    from routes import data_analysis
    eda_code = data_analysis.generate_data_analysis_code(filedetails, final_business_problem)
    data_analysis.save_code_to_file(eda_code)
    return (f"Final Business Problem Stored: {final_business_problem}<br>"
            f"EDA code generated and saved to 'generated_eda.py'.")

if __name__ == "__main__":
    app.run(debug=True)
