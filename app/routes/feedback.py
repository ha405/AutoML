import google.generativeai as genai

GOOGLE_API_KEY = "AIzaSyBF8Ik7v2Uwy_cRVzoDEj30g2oNpXPPlrQ"  
MODEL_NAME = "gemini-2.0-flash"

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

SYSTEM_PROMPT_INITIAL = """
You are an AI assisting with business problem definition.
The user will provide a brief business problem statement.
Your task is to:
1. Analyze the user's problem statement.
2. Decide if you need more information to understand the problem deeply enough to formulate a well-defined business problem statement.
3. If you need more information, ask ONE clarifying question.  The question should be specific and help you understand a crucial aspect of the problem.
4. If you believe you have enough information to formulate a problem statement, your response MUST START with the EXACT phrase: "Ready to formulate problem." and nothing else before it. Do not ask a question in this case.
Your response should follow one of these formats:
Format 1: Asking a Clarifying Question (if more info needed):
<Your clarifying question>
Format 2: Ready to formulate problem (if enough info):
Ready to formulate problem.  (Your response MUST START with this EXACT phrase)
Example:
What specific metrics are you currently tracking to measure this problem?
Start the process now.
"""

SYSTEM_PROMPT_NEXT_ITERATION = """
You are continuing to assist with business problem definition.
The user has responded to your previous question.
Your task is to:
1. Review the entire conversation so far, including the initial problem statement and the user's responses to your questions.
2. Decide if you now have enough information to formulate a well-defined business problem statement.
3. If you still need more information, ask ONE more clarifying question.
4. If you believe you have enough information NOW to formulate a problem statement, your response MUST START with the EXACT phrase: "Ready to formulate problem." and nothing else before it.
Your response should follow one of these formats:
Format 1: Asking a Clarifying Question:
<Your clarifying question>
Format 2: Ready to formulate problem:
Ready to formulate problem. (Your response MUST START with this EXACT phrase)
Continue the process.
"""

SYSTEM_PROMPT_FINAL_PROBLEM = """
Now that you have indicated you are ready by responding with "Ready to formulate problem.", structure a **well-defined business problem statement** based on the entire conversation.
Your response should ONLY include the final business problem statement and nothing else.
Do not include any additional suggestions, next steps, or conversational text—only the final business problem.
"""

def generate_response(messages):
    response = model.generate_content(contents=messages)
    return response.text.strip()

def process_feedback(conversation):
    conversation_text = "\n".join(conversation)
    ai_response = generate_response(conversation_text)
    ai_response = ai_response.strip()
    if ai_response.startswith("Ready to formulate problem"):
        conversation.append(f"AI: {ai_response}")
        conversation.append(f"System: {SYSTEM_PROMPT_FINAL_PROBLEM}")
        final_response = generate_response("\n".join(conversation))
        final_response = final_response.strip()
        return conversation, final_response
    else:
        if not ai_response.startswith("Question"):
            question_number = sum(1 for msg in conversation if msg.startswith("AI: Question"))
            ai_response = f"Question {question_number + 1}) " + ai_response
        conversation.append(f"AI: {ai_response}")
        return conversation, None