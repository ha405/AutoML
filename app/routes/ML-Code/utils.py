import pandas as pd

def filepreprocess(file):
    
    try:
        df = pd.read_csv(file)
        file_info = {
            "file_type": "CSV",
            "num_rows": df.shape[0],  
            "num_columns": df.shape[1],  
            "column_names": list(df.columns),  
            "data_types": df.dtypes.apply(lambda x: str()).to_dict(),  
            "null_entries": df.isnull().sum().to_dict(),  
            "duplicates": df.duplicated().sum(),  
            "unique_values": df.nunique().to_dict(),  
            "sample_data": df.head(5).to_dict(orient="records")  
        }
        return file_info
    except Exception as e:
        return {"error": f"Error processing CSV file: {str(e)}"}
    
    
