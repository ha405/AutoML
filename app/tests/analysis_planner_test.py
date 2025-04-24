# File: test_analysis_planner.py
import os
import sys
import utils

# Adjust path if AnalysisPlanner.py is not in the same directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # If AnalysisPlanner is in parent dir
from routes import AnalysisPlanner  # Import your AnalysisPlanner script

# --- Configuration ---
EDA_CODE_FILE_PATH = r'E:\AutoML\app\scripts\credit_churn_eda.py'
EDA_LOGS_FILE_PATH = r'E:\AutoML\app\scripts\data_analysis_logs.txt' 
OUTPUT_PLAN_FILE = r"E:\AutoML\app\scripts\ml_plan_output.txt"
DATASET_PATH = "E:\AutoML\TestDatasets\credit_card_churn.csv" 

class MockFile:
    def __init__(self, path):
        self.filename = path
        # Read content during initialization to handle potential file not found early
        try:
            # We don't actually need to read the content here anymore
            # if filepreprocess uses the filename. Just check existence.
            if not os.path.exists(path):
                 raise FileNotFoundError(f"No file found at {path}")
            print(f"MockFile initialized for: {path}")
            self.content = b'' # Placeholder, not strictly needed if read() isn't used by pandas
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            self.content = None # Mark as invalid

    # Keep read() in case something else expects it, but pandas won't use it now
    def read(self):
        if self.content is None:
             raise FileNotFoundError(f"Content not available, file not found at {self.filename}")
        # This won't actually be called by pd.read_csv(file.filename)
        return self.content

file_obj = MockFile(DATASET_PATH)

# --- Dummy Business Problem and File Details (Replace with your actual data) ---
BUSINESS_PROBLEM = "A business manager of a consumer credit card bank is facing the problem of customer attrition. They want to analyze the data to find out the reason behind this and leverage the same to predict customers who are likely to drop off."
FILE_DETAILS = utils.filepreprocess(file_obj)

def load_code_from_file(file_path):
    """Loads Python code from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        return code
    except Exception as e:
        print(f"Error reading code file: {e}")
        return None

def load_logs_from_file(file_path):
    """Loads logs from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            logs = f.read()
        return logs
    except FileNotFoundError:
        print(f"Warning: Logs file not found at {file_path}. Proceeding with empty logs.")
        return "# No EDA logs file found at specified path."
    except Exception as e:
        print(f"Error reading logs file: {e}")
        return f"# Error reading EDA logs: {e}"

def main():
    print("--- Starting Analysis Planner Test ---")

    # 1. Load EDA Code
    print("\n--- Loading EDA Code ---")
    eda_code = load_code_from_file(EDA_CODE_FILE_PATH)
    if not eda_code:
        print("❌ Failed to load EDA code. Exiting.")
        return
    print("✅ EDA code loaded.")

    # 2. Load EDA Logs (Create a dummy log file if you don't have actual logs yet)
    print("\n--- Loading EDA Logs ---")
    eda_logs = load_logs_from_file(EDA_LOGS_FILE_PATH) # Assuming logs are in 'data_analysis_logs.txt'
    print("✅ EDA logs loaded (or warning displayed if file not found).")
    #print("--- EDA Logs Content Snippet ---") # Optional: print a snippet of logs for review
    #print(eda_logs[:500] + "..." if len(eda_logs) > 500 else eda_logs)

    # 3. Generate ML Plan using AnalysisPlanner
    print("\n--- Generating ML Plan ---")
    ml_plan_output = AnalysisPlanner.generate_ml_plan(
        business_problem=BUSINESS_PROBLEM,
        file_details=FILE_DETAILS,
        eda_code=eda_code,
        eda_output_logs=eda_logs
    )

    if ml_plan_output.startswith("# Error"):
        print("❌ ERROR generating ML Plan:")
        print(ml_plan_output)
        return

    print("✅ ML Plan generated successfully!")
    print("\n--- ML Plan Output ---")
    print(ml_plan_output)

    # 4. Save ML Plan to file (optional)
    if OUTPUT_PLAN_FILE:
        try:
            with open(OUTPUT_PLAN_FILE, 'w', encoding='utf-8') as f:
                f.write(ml_plan_output)
            print(f"\nML Plan saved to: {OUTPUT_PLAN_FILE}")
        except Exception as e:
            print(f"❌ Error saving ML Plan to file: {e}")

    print("\n--- Analysis Planner Test Completed ---")

if __name__ == "__main__":
    main()