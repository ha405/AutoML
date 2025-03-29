from google import genai
client = genai.Client(api_key="AIzaSyBoAFOxBSX1nxEF8lNuhJudPiHVTCRNK8Q")


SYSTEM_PROMPT_INITIAL = """
You are an AI assisting with business problem definition. 
The user will provide a brief business problem statement.
Your task is to **ask exactly one clarifying question** to refine the problem.

Your response should strictly follow this format:
Question 1) <your first question>
"""

SYSTEM_PROMPT_NEXT_QUESTION = """
Now, based on the user's response, ask the **next clarifying question**.

Strictly follow this format:
Question 2) <your second question>
"""

SYSTEM_PROMPT_FINAL_QUESTION = """
Now, based on the user's response, ask the **final clarifying question**.

Strictly follow this format:
Question 3) <your third question>
"""

SYSTEM_PROMPT_FINAL_PROBLEM = """
Now that you have gathered enough details, structure a **well-defined business problem statement** based on the conversation.
Your response should NOT include any additional suggestions or next steps—only the final business problem.
"""

def generate_response(messages):
    """Generate AI response using Gemini API."""
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=messages
    )
    return response.text.strip()  

def feedback_loop():
    """Runs a structured 3-question feedback loop using Gemini API with user input."""
    messages = []

    # Initial business problem statement
    user_input = input("\n🟢 Start by describing your business problem:\nYou: ")
    messages.append(f"User: {user_input}")

    # Iterative questioning process 
    for i, system_prompt in enumerate([SYSTEM_PROMPT_INITIAL, SYSTEM_PROMPT_NEXT_QUESTION, SYSTEM_PROMPT_FINAL_QUESTION]):
        messages.append(f"System: {system_prompt}")
        response = generate_response("\n".join(messages))

       
        if not response.startswith(f"Question {i+1})"):
            response = f"Question {i+1}) " + response

        print(f"AI: {response}")
        messages.append(f"AI: {response}")

        # User input 
        user_response = input("You: ")
        messages.append(f"User: {user_response}")

    # Business Problem Statement
    messages.append(f"System: {SYSTEM_PROMPT_FINAL_PROBLEM}")
    final_response = generate_response("\n".join(messages))
    
    print("\n✅ Final Business Problem Formulated:\n", final_response)

# Run the feedback loop with user input
feedback_loop()