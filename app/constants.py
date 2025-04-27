import os
APP_DIR = os.path.dirname(os.path.abspath(__file__))
AUTOML_ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, os.pardir))
DATASET_PATH = os.path.join(AUTOML_ROOT_DIR, "TestDatasets", "Input.csv")
# print(DATASET_PATH)
PROCESSED_DATASET_PATH = os.path.join(AUTOML_ROOT_DIR, "TestDatasets", "Input_processed.csv") # Assuming name

SCRIPTS_PATH_REL = os.path.join("app", "scripts")
VISUALIZATIONS_PATH_REL = os.path.join("app", "visualizations")

EDA_CODE_FILE_PATH = os.path.join(AUTOML_ROOT_DIR, SCRIPTS_PATH_REL, "Eda_code.py")
EDA_GUIDANCE_PLAN = os.path.join(AUTOML_ROOT_DIR, SCRIPTS_PATH_REL, "eda_guidance_plan.txt")
EDA_LOGS_FILE_PATH = os.path.join(AUTOML_ROOT_DIR, SCRIPTS_PATH_REL, "data_analysis_logs.txt")
ML_PLAN = os.path.join(AUTOML_ROOT_DIR, SCRIPTS_PATH_REL, "ml_plan.txt")
ML_CODE_FILE_PATH = os.path.join(AUTOML_ROOT_DIR, SCRIPTS_PATH_REL, "ML.py")
ML_OUTPUT_LOGS_FILE = os.path.join(AUTOML_ROOT_DIR, SCRIPTS_PATH_REL, "ml_output_logs.txt")
VISUALIZATION_PLAN_FILE = os.path.join(AUTOML_ROOT_DIR, SCRIPTS_PATH_REL, "visualization_plan_output.txt")
VISUALIZATION_CODE_FILE_PATH = os.path.join(AUTOML_ROOT_DIR, SCRIPTS_PATH_REL, "Visualizations.py")
VISUALIZATION_OUTPUT_DIR = os.path.join(AUTOML_ROOT_DIR, VISUALIZATIONS_PATH_REL)


# ----------------------Code Fix Attempts -----------------------------
MAX_ATTEMPTS = 5
MAX_FIX_ATTEMPTS = 3
MAX_ML_EXEC_ATTEMPTS = 5
MAX_ML_FIX_ATTEMPTS = 3
