import os

# ========================
# Directory Structure
# ========================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
AUTOML_ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, os.pardir))

# Single output directory for ALL runtime artifacts
OUTPUT_DIR = os.path.join(AUTOML_ROOT_DIR, "output")

# Sub-directories
DATASETS_DIR = os.path.join(OUTPUT_DIR, "datasets")
SCRIPTS_DIR = os.path.join(OUTPUT_DIR, "scripts")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
PLANS_DIR = os.path.join(OUTPUT_DIR, "plans")
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
SESSIONS_DIR = os.path.join(OUTPUT_DIR, "sessions")

# Create all directories on import
for d in [DATASETS_DIR, SCRIPTS_DIR, LOGS_DIR, PLANS_DIR, VISUALIZATIONS_DIR, SESSIONS_DIR]:
    os.makedirs(d, exist_ok=True)

# ========================
# File Paths — Datasets
# ========================
DATASET_PATH = os.path.join(DATASETS_DIR, "Input.csv")
PROCESSED_DATASET_PATH = os.path.join(DATASETS_DIR, "Input_processed.csv")

# ========================
# File Paths — Scripts (LLM-generated code)
# ========================
EDA_CODE_FILE_PATH = os.path.join(SCRIPTS_DIR, "Eda_code.py")
ML_CODE_FILE_PATH = os.path.join(SCRIPTS_DIR, "ML.py")
VISUALIZATION_CODE_FILE_PATH = os.path.join(SCRIPTS_DIR, "Visualizations.py")

# ========================
# File Paths — Plans (LLM-generated strategy docs)
# ========================
EDA_GUIDANCE_PLAN = os.path.join(PLANS_DIR, "eda_guidance_plan.txt")
ML_PLAN = os.path.join(PLANS_DIR, "ml_plan.txt")
VISUALIZATION_PLAN_FILE = os.path.join(PLANS_DIR, "visualization_plan.txt")

# ========================
# File Paths — Logs (execution output)
# ========================
EDA_LOGS_FILE_PATH = os.path.join(LOGS_DIR, "data_analysis_logs.txt")
ML_OUTPUT_LOGS_FILE = os.path.join(LOGS_DIR, "ml_output_logs.txt")

# ========================
# File Paths — Visualizations
# ========================
VISUALIZATION_OUTPUT_DIR = VISUALIZATIONS_DIR

# ========================
# Frontend paths
# ========================
FRONTEND_PUBLIC_DIR = os.path.join(APP_DIR, "frontend", "public")
FRONTEND_JSON_PATH = os.path.join(FRONTEND_PUBLIC_DIR, "data")

# ========================
# Retry Configuration
# ========================
MAX_ATTEMPTS = 5
MAX_FIX_ATTEMPTS = 3
MAX_ML_EXEC_ATTEMPTS = 5
MAX_ML_FIX_ATTEMPTS = 3

# ========================
# Legacy aliases (for AnalysisPlanner backward compat)
# ========================
SCRIPTS_PATH_REL = os.path.relpath(PLANS_DIR, AUTOML_ROOT_DIR)
