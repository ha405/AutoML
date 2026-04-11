# DataWise AI

DataWise AI is an automated machine learning and data analysis platform that converts CSV data into insights, models, and visualizations.

## Quick Start

The platform is fully containerized using Docker.

### 1. Configuration
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 2. Execution
Launch the ecosystem using Docker Compose:
```bash
docker-compose up -d --build
```
Access the dashboard at http://localhost:3000.

### Running Backend Only
If you prefer to run the backend without Docker:
1. Ensure Python 3.12+ is installed.
2. Install dependencies: `pip install -r requirements.txt`
3. Set your environment variable: `GOOGLE_API_KEY=your_key`
4. Run: `python app/main.py`

## Datasets

The platform accepts any standard CSV file up to 50MB.

### Using Sample Data
Several pre-verified datasets are available in the `TestDatasets/` directory for immediate testing:
- Car price prediction (Regression)
- Amazon sales trends (Time-series/Sales)
- Supermarket analytics (Retail)
- Credit card churn (Classification)

### Custom Data
Upload your own CSV directly through the dashboard interface once the system is running.

## Project Architecture

- **Frontend**: React-based dashboard served via Nginx.
- **Backend**: Flask API orchestrating ML and Data Analysis logic.
- **Output**: All runtime artifacts (scripts, logs, and plots) are stored in the `output/` directory.

## Verification

To verify the pipeline independently of the UI, run the automated test suite. It will automatically upload the sample car price dataset to the backend and execute the full ML flow:

```bash
# Basic test against Docker (localhost:3000) or local (localhost:5000)
python test_pipeline.py --host http://localhost:5000
```

### Manual API Usage
To pass a dataset manually to the backend API, send a `POST` request to `/api/home` with:
- **Header**: `multipart/form-data`
- **Body**: A file field named `file-upload` containing your CSV.
- **Body**: A text field named `queryInput` describing your analysis goal.

## Tech Stack
- **Backend**: Python 3.12 (Flask, Pandas, Scikit-Learn).
- **Frontend**: React 18 (MUI).
- **AI**: Google Gemini (2.0 Flash).
