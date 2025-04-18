from groq import Groq
from utils import filepreprocess
import os

# Set Groq API key
os.environ["GROQ_API_KEY"] = "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"
client = Groq()

def generate_reflection(ml_code: str, csv_file_path: str):
    """
    Takes the ML code and CSV file, preprocesses the file, then generates reflection feedback
    using Groq, and provides suggestions for improving the ML code.
    
    Parameters:
        ml_code (str): The generated ML code from the original model.
        csv_file_path (str): The path to the CSV file that contains data.
    
    Returns:
        str: The reflection feedback from the Groq model.
    """
    
    #Preprocess using utils.py function
    preprocessed_file_path = filepreprocess(csv_file_path)
    
    # Prepare the reflection input for Groq
    reflection_prompt = (
        "You are a senior data scientist and ML expert. Critique the ML code provided. Focus on:\n"
        "- Algorithm appropriateness\n"
        "- Feature selection\n"
        "- Evaluation strategy\n"
        "- Possible improvements\n\n"
        "ML Code:\n"
        f"{ml_code}\n\n"
        "Data File (Preprocessed):\n"
        f"{preprocessed_file_path}\n"
        "Provide detailed feedback and suggest concrete improvements."
    )
    
    # Request reflection 
    reflection_response = client.chat.completions.create(
        messages=[{"role": "system", "content": reflection_prompt}],
        model="llama3-70b-8192"
    ).choices[0].message.content
    
    # Extract and return the response
    return reflection_response
