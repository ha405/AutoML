from google import genai
client = genai.Client(api_key="AIzaSyBoAFOxBSX1nxEF8lNuhJudPiHVTCRNK8Q")


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
Question <number>) <Your clarifying question>

Format 2: Ready to Formulate Problem (if enough info):
Ready to formulate problem.  (Your response MUST START with this EXACT phrase)

Example of asking a question:
Question 1) What specific metrics are you currently tracking to measure this problem?


Start the process now.
"""

SYSTEM_PROMPT_NEXT_ITERATION = """
You are continuing to assist with business problem definition.
The user has responded to your previous question.

Your task is to:
1. Review the entire conversation so far, including the initial problem statement and the user's responses to your questions.
2. Based on the conversation, decide if you now have enough information to formulate a well-defined business problem statement.
3. If you still need more information, ask ONE more clarifying question. The question should be specific and address any remaining ambiguities or critical unknowns.
4. If you believe you have enough information NOW to formulate a problem statement, your response MUST START with the EXACT phrase: "Ready to formulate problem." and nothing else before it. Do not ask a question in this case.

Your response should follow one of these formats:

Format 1: Asking a Clarifying Question (if more info still needed):
Question <number>) <Your clarifying question>

Format 2: Ready to Formulate Problem (if enough info now):
Ready to formulate problem. (Your response MUST START with this EXACT phrase)

Continue the process.
"""


SYSTEM_PROMPT_FINAL_PROBLEM = """
Now that you have indicated you are ready by responding with "Ready to formulate problem.", structure a **well-defined business problem statement** based on the entire conversation.
Your response should ONLY include the final business problem statement and nothing else.
Do not include any additional suggestions, next steps, or conversational text—only the final business problem.
"""

def generate_response(messages):
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=messages
    )
    return response.text.strip()

def feedback_loop():
    messages = []
    question_count = 0
    ask_more_questions = True

    user_input = input("\n🟢 Start by describing your business problem:\nYou: ")
    messages.append(f"User: {user_input}")
    messages.append(f"System: {SYSTEM_PROMPT_INITIAL}")

    while ask_more_questions:
        ai_response = generate_response("\n".join(messages))
        ai_response_processed = ai_response.strip()

        if ai_response_processed.startswith("Ready to formulate problem"):
            print(f"AI: {ai_response_processed}")
            ask_more_questions = False
            messages.append(f"AI: {ai_response_processed}")
            break

        else:
            question_count += 1
            if not ai_response_processed.startswith(f"Question {question_count})"):
                ai_response_processed = f"Question {question_count}) " + ai_response_processed

            print(f"AI: {ai_response_processed}")
            messages.append(f"AI: {ai_response_processed}")

            user_response = input("You: ")
            messages.append(f"User: {user_response}")

            messages.append(f"System: {SYSTEM_PROMPT_NEXT_ITERATION}")


    if not ask_more_questions:
        messages.append(f"System: {SYSTEM_PROMPT_FINAL_PROBLEM}")
        final_response = generate_response("\n".join(messages))

        print("\n✅ Final Business Problem Formulated:\n", final_response)
    else:
        print("\n⚠️  Process interrupted before formulating final problem.")

feedback_loop()