import pandas as pd

def filepreprocess(file):
    if not file.filename.endswith(".csv"):
        return {"error": "Unsupported file format. Currently, only CSV files are processed."}
    
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return {"error": f"Error reading CSV file: {str(e)}"}
    
    file_info = {}
    
    try:
        file_info["file_type"] = "CSV"
    except Exception as e:
        file_info["file_type"] = f"Error: {str(e)}"
    
    try:
        file_info["num_rows"] = df.shape[0]
    except Exception as e:
        file_info["num_rows"] = f"Error: {str(e)}"
    
    try:
        file_info["num_columns"] = df.shape[1]
    except Exception as e:
        file_info["num_columns"] = f"Error: {str(e)}"
    
    try:
        file_info["column_names"] = list(df.columns)
    except Exception as e:
        file_info["column_names"] = f"Error: {str(e)}"
    
    try:
        file_info["data_types"] = df.dtypes.astype(str).to_dict()
    except Exception as e:
        file_info["data_types"] = f"Error: {str(e)}"
    
    try:
        file_info["null_entries"] = df.isnull().sum().to_dict()
    except Exception as e:
        file_info["null_entries"] = f"Error: {str(e)}"
    
    try:
        file_info["duplicates"] = int(df.duplicated().sum())
    except Exception as e:
        file_info["duplicates"] = f"Error: {str(e)}"
    
    try:
        file_info["unique_values"] = df.nunique().to_dict()
    except Exception as e:
        file_info["unique_values"] = f"Error: {str(e)}"
    
    try:
        file_info["sample_data"] = df.head(5).to_dict(orient="records")
    except Exception as e:
        file_info["sample_data"] = f"Error: {str(e)}"
    
    return file_info
