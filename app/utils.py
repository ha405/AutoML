import pandas as pd
import os

def filepreprocess(file_input):
    """
    Reads a CSV file (given by path) and extracts metadata.
    Args:
        file_input: Either a string path to the CSV file or a direct object with filename

    Returns:
        A dictionary containing file metadata or an error dictionary.
    """
    # Check if the filename is being sent correctly 
    print(f"Received file input: {file_input}")

    # If we have a path, read from the path

    file_path = file_input

    if not isinstance(file_path, str):
         print(f"File format is not correct, must be file path!")

    if not os.path.exists(file_path):
            return {"error": "Uploaded file not found."}
    # Check the file exist path to make sure it is CSV is running correctly 

    elif not file_path.lower().endswith(".csv"):
         return {"error": "Uploaded CSV file is not in proper format."}
    try:
        # Handle successful processing
        #Load CSV through processing steps
        df = pd.read_csv(file_path)
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
        return file_info

    # Handle common exceptions such as incorrect path
    except FileNotFoundError:
          return {"error": "Uploaded file not found."}

    # Handle potential errors during file reading.

    except Exception as e:
          # Report and exception has occured during reading of the file

          return {"error": f"Uploaded file could not be processed : " + str(e)}

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
    """Loads logs from a text file, attempting different encodings."""
    encodings_to_try = ['utf-8', 'utf-16', 'utf-8-sig', 'latin-1'] # Add more if needed
    logs = None
    error_message = None

    if not os.path.exists(file_path):
         print(f"Warning: Logs file not found at {file_path}. Proceeding with empty logs.")
         return "# No EDA logs file found at specified path."

    for enc in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                logs = f.read()
            # print(f"Successfully read logs file with encoding: {enc}")
            return logs # Return as soon as successful read occurs
        except UnicodeDecodeError:
            # print(f"Failed to decode logs file with encoding: {enc}")
            error_message = f"Failed to decode with attempted encodings: {encodings_to_try}"
            continue # Try the next encoding
        except FileNotFoundError:
            error_message = f"Logs file not found at {file_path}"
            break # No point trying other encodings if file not found
        except Exception as e:
            error_message = f"Error reading logs file: {e}"
            break # Stop on other errors

    # If loop completes without success
    print(f"Warning: Could not read logs file {file_path}. {error_message}")
    return f"# Error reading EDA logs: {error_message}"

# Keep load_code_from_file as it is (usually Python scripts are UTF-8)
def load_code_from_file(file_path):
    """Loads Python code from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        return code
    except Exception as e:
        print(f"Error reading code file: {e}")
        return None
    


# Add this near the top of main.py or in a separate utils file
import json
import numpy as np
from flask.json.provider import JSONProvider # Import the provider base

class NumpyJSONProvider(JSONProvider):
    def default(self, obj):
        if isinstance(obj, np.integer): # Handle numpy integers (int32, int64, etc.)
            return int(obj)
        elif isinstance(obj, np.floating): # Handle numpy floats (float32, float64, etc.)
            return float(obj)
        elif isinstance(obj, np.ndarray): # Handle numpy arrays
            return obj.tolist() # Convert arrays to Python lists
        elif isinstance(obj, np.bool_): # Handle numpy booleans
            return bool(obj)
        elif isinstance(obj, (np.void, np.generic)): # Handle other numpy scalar types if needed
             # Decide how to handle these - often converting to string or int/float is reasonable
             # Example: return str(obj) or handle specific types
             return str(obj) # Fallback to string for other numpy types
        # Let the base class default method raise the TypeError for unsupported types
        return super().default(obj)

    # Need to implement dumps and loads using the custom default
    def dumps(self, obj, **kwargs):
        kwargs.setdefault("default", self.default)
        # Use ensure_ascii=False for better Unicode handling if needed
        return json.dumps(obj, **kwargs)

    def loads(self, s, **kwargs):
        return json.loads(s, **kwargs)