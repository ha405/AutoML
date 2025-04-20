import os
APP_DIR = os.path.dirname(os.path.abspath(__file__))
AUTOML_ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, os.pardir))

DATASET_PATH = os.path.join(AUTOML_ROOT_DIR, "TestDatasets", "Input.csv")
PROCESSED_DATASET_PATH = os.path.join(AUTOML_ROOT_DIR, "TestDatasets", "Input_processed.csv") # Assuming name

SCRIPTS_PATH_REL = os.path.join("app", "scripts")
VISUALIZATIONS_PATH_REL = os.path.join("app", "visualizations")

EDA_CODE_FILE_PATH = os.path.join(AUTOML_ROOT_DIR, SCRIPTS_PATH_REL, "Eda_code.py")
EDA_LOGS_FILE_PATH = os.path.join(AUTOML_ROOT_DIR, SCRIPTS_PATH_REL, "data_analysis_logs.txt")
OUTPUT_PLAN_FILE = os.path.join(AUTOML_ROOT_DIR, SCRIPTS_PATH_REL, "ml_plan_output.txt")
ML_CODE_FILE_PATH = os.path.join(AUTOML_ROOT_DIR, SCRIPTS_PATH_REL, "ML.py")
ML_OUTPUT_LOGS_FILE = os.path.join(AUTOML_ROOT_DIR, SCRIPTS_PATH_REL, "ml_output_logs.txt")
VISUALIZATION_PLAN_FILE = os.path.join(AUTOML_ROOT_DIR, SCRIPTS_PATH_REL, "visualization_plan_output.txt")
VISUALIZATION_CODE_FILE_PATH = os.path.join(AUTOML_ROOT_DIR, SCRIPTS_PATH_REL, "Visualizations.py")
VISUALIZATION_OUTPUT_DIR = os.path.join(AUTOML_ROOT_DIR, VISUALIZATIONS_PATH_REL)
