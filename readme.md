# DataWise AI
## Overview
DataWise AI is a full-stack data analysis and machine learning platform designed to automate complex analytical tasks and generate actionable visualizations. It empowers SMEs to make data-informed decisions through an interactive dashboard and robust backend ML pipeline.

## Setup Instructions

### Files setup
Download and copy all the files from the following link into the folder app/routes/filtered  
https://drive.google.com/drive/folders/1oXV7d5Mzohfndjb8zxRTN9IV1DhvpwZk?usp=sharing

### Backend Setup
Ensure Python 3.9+ is installed.  
Run in Powershell or CMD: set GOOGLE_API_KEY=AIzaSyBF8Ik7v2Uwy_cRVzoDEj30g2oNpXPPlrQ
Create and activate a virtual environment (recommended):  
python -m venv venv  
source venv/bin/activate (On Windows: venv\Scripts\activate)

Install required Python dependencies:  
pip install -r requirements.txt

Run the backend server:  
python main.py

### Frontend Setup
Navigate to the frontend directory: 
cd frontend

Install Node.js dependencies:  
npm install

Start the frontend development server:  
npm start

Access the dashboard via http://localhost:3000

## Project Structure
frontend/ — React-based frontend for dashboard and visualization.

Backend root directory contains core Python modules and scripts:

main.py — backend server entry point.

config.py, constants.py — configuration files.

utils.py — utility functions (Analysis Planner, feedback loops, ML, Data Analysis, Visualizations).

gradient_boosting_model.joblib, trained_model.joblib — saved ML models.

Supporting directories:

routes/, scripts/, visualizations/ — backend API routes, helper scripts, and visualization generation.

static/, templates/ — static assets and HTML templates.

tests/ — test files.

Additional components:

Models/, Voice/, app/, TestDatasets/ — ML models, voice-related modules, app components, and datasets.

experiments/ — experiment tracking.

.flask_session/ — Flask session data.

## Notes
The system uses accessible APIs like Gemini for ML and data processing.

Modular design emphasizes Analysis Planning, Machine Learning, and Visualization.

Supports expansion for new ML tasks, data types, and interactive dashboard features.

Run both backend and frontend servers simultaneously to use the platform fully.

## Example Prompts for RAG Pipeline
Prompt 1: "Outline the main steps of honey farming."
Prompt 2: "How much space do I need to open a laundry business?"
Prompt 3: "What SOPs do I need to follow to open a restaurant in Lahore, Pakistan?"
Prompt 4: "Summarize what steps I would need to follow to open a library in Lahore."

