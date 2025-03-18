from together import Together

API_KEY = "e9e351b1305aac351a602fe3d58d4c9f389acc2b74c3eff0dc4009cd986c2f50"
client = Together(api_key=API_KEY)

def chat_with_bot():
    """Continuously chat with the model until the user types 'exit'."""
    messages = []

    print("🟢 Chatbot is running. Type 'exit' to stop.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("🔴 Chat ended.")
            break

        messages.append({"role": "user", "content": user_input})
        
        response = client.chat.completions.create(
            model="meta-llama/Llama-2-13b-chat-hf",
            messages=messages,
            max_tokens=500,
        )

        print(f"AI: {response.choices[0].message.content.strip()}")
        print("Full API Response:\n", response)
        messages.append({"role": "assistant", "content": response})

# Start the chatbot
chat_with_bot()
