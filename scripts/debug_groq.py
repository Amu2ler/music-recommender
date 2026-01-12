from groq import Groq
import os

# Use environment variable instead of hardcoded key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("❌ Error: GROQ_API_KEY environment variable not set.")
    exit(1)

print(f"Testing Groq Connection with key: {api_key[:10]}...")

try:
    client = Groq(api_key=api_key)
    print("Requesting model: llama-3.1-8b-instant")
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Hello",
            }
        ],
        model="llama-3.1-8b-instant",
    )
    print("Success!")
    print(chat_completion.choices[0].message.content)
except Exception as e:
    print("FAILED!")
    print(f"Error type: {type(e)}")
    print(f"Error message: {e}")
