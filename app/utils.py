import pandas as pd

def filepreprocess(file):
    """
    Reads a CSV file using its filename and extracts metadata.
    Args:
        file: An object with a 'filename' attribute containing the path to the CSV.
    Returns:
        A dictionary containing file metadata or an error dictionary.
    """
    # Check if the filename attribute exists and ends with .csv
    if hasattr(file, 'filename') and file.filename.endswith(".csv"):
        try:
            # --- CHANGE HERE: Use file.filename instead of file ---
            print(f"Attempting to read CSV from path: {file.filename}")
            df = pd.read_csv(file.filename)
            # ------------------------------------------------------
            file_info = {
                "file_type": "CSV",
                "num_rows": df.shape[0],
                "num_columns": df.shape[1],
                "column_names": list(df.columns),
                # Using .name preserves more specific dtype info (e.g., 'int64', 'float64')
                "data_types": df.dtypes.apply(lambda x: x.name).to_dict(),
                "null_entries": df.isnull().sum().to_dict(),
                "duplicates": df.duplicated().sum(),
                "unique_values": df.nunique().to_dict(),
                "sample_data": df.head(5).to_dict(orient="records")
            }
            print("File preprocessing successful.")
            return file_info
        except FileNotFoundError:
             print(f"ERROR: File not found at path: {file.filename}")
             return {"error": f"File not found at path: {file.filename}"}
        except Exception as e:
            print(f"Error during file preprocessing: {e}")
            return {"error": f"Error processing CSV file: {str(e)}"}

    elif not hasattr(file, 'filename'):
         print("Error: Input object does not have a 'filename' attribute.")
         return {"error": "Input object lacks 'filename' attribute."}
    else:
        print(f"Unsupported file format: {file.filename}")
        return {"error": "Unsupported file format. Currently, only CSV files are processed."}