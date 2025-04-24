# test_analysis.py
from routes import data_analysis
# from data_analysis import generate_data_analysis_code, save_code_to_file
from utils import filepreprocess
import os

# Define dataset path and business problem
DATASET_PATH = "E:\AutoML\TestDatasets\credit_card_churn.csv" # Make sure this file exists in /content/
BUSINESS_PROBLEM = "A business manager of a consumer credit card bank is facing the problem of customer attrition. They want to analyze the data to find out the reason behind this and leverage the same to predict customers who are likely to drop off."
OUTPUT_FILENAME = "credit_churn_eda.py" # Define the output filename 

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

print(f"Starting analysis for dataset: {DATASET_PATH}")
print(f"Business Problem: {BUSINESS_PROBLEM}")

# 1. Create MockFile and Preprocess
print("\n--- Step 1: Preprocessing File ---")
file_obj = MockFile(DATASET_PATH)

# Check if file was found during MockFile init
if file_obj.content is None:
    print("❌ ERROR: Cannot proceed, dataset file not found during MockFile initialization.")
else:
    filedetails = filepreprocess(file_obj)

    # Check for errors during preprocessing
    if 'error' in filedetails:
         print(f"❌ ERROR in filepreprocess: {filedetails['error']}")
    else:
        print("File details extracted successfully.")
        # print(filedetails) # Optional: print extracted details

        # 2. Generate and Refine EDA Code
        print("\n--- Step 2: Generating and Refining EDA Code ---")
        refined_eda_code = data_analysis.generate_and_refine_eda_code(
            filedetails=filedetails,
            business_problem=BUSINESS_PROBLEM,
            file_path=DATASET_PATH,
            max_refinements=5 # You can adjust the number of refinement loops
        )

        # 3. Save the Final Code
        print("\n--- Step 3: Saving Generated Code ---")
        if refined_eda_code and not refined_eda_code.startswith("# Error"):
             data_analysis.save_code_to_file(refined_eda_code, OUTPUT_FILENAME)
             print(f"\n✅ Process complete! Refined EDA script saved as {OUTPUT_FILENAME}")
             # You can optionally display the first few lines of the generated code
             print("\n--- Generated Code Snippet ---")
             print('\n'.join(refined_eda_code.splitlines()[:20]))
             print("...")
             print("-----------------------------")

        else:
             print(f"❌ ERROR: Failed to generate valid EDA code. Final code not saved.")
             print(f"Last generated content:\n{refined_eda_code}")