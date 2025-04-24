# test_ml.py
from routes import MachineLearning # Import the module we just defined
import os

# --- Configuration ---
# Use raw string for Windows paths
CSV_FILE_PATH = r'E:\AutoML\TestDatasets\CarPrice_Assignment.csv'
PROBLEM_DESCRIPTION = "We are required to model the price of cars with the available independent variables. It will be used by the management to understand how exactly the prices vary with the independent variables. They can accordingly manipulate the design of the cars, the business strategy etc. to meet certain price levels. Further, the model will be a good way for management to understand the pricing dynamics of a new market."
OUTPUT_FILENAME = "CarPrice_ML_Pipeline.py"
MAX_REFINEMENTS = 3 # Number of reflection cycles

# --- Execution ---

print(f"Starting ML pipeline generation for dataset: {CSV_FILE_PATH}")
print(f"Business Problem: {PROBLEM_DESCRIPTION}")

# Check if the input file exists before starting
if not os.path.exists(CSV_FILE_PATH):
    print(f"❌ ERROR: Input dataset file not found at {CSV_FILE_PATH}")
else:
    # Generate and refine the ML code
    print("\n--- Generating and Refining ML Code ---")
    final_ml_code = MachineLearning.generate_and_refine_ml_code(
        business_problem=PROBLEM_DESCRIPTION,
        file_path=CSV_FILE_PATH,
        max_refinements=MAX_REFINEMENTS
    )

    # Save the final code
    print("\n--- Saving Generated Code ---")
    if final_ml_code and not final_ml_code.startswith("# Error"):
        # Define where to save the output script
        save_dir = r"E:\AutoML\app\scripts" # Define your desired output directory
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, OUTPUT_FILENAME)

        try:
            with open(output_path, "w", encoding='utf-8') as f:
                f.write(final_ml_code)
            print(f"✅ Final ML script saved successfully at: {output_path}")
            print(f"\n✅ Process complete!")

            # Optionally display a snippet
            print("\n--- Generated Code Snippet ---")
            print('\n'.join(final_ml_code.splitlines()[:20]))
            print("...")
            print("-----------------------------")

        except OSError as e:
            print(f"❌ Error saving file to {output_path}: [Errno {e.errno}] {e.strerror}")
        except Exception as e:
            print(f"❌ An unexpected error occurred during file saving: {e}")
    else:
        print(f"❌ ERROR: Failed to generate valid ML code. Final code not saved.")
        print(f"Last generated content:\n{final_ml_code}")