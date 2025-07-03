import os
import sys
import cohere

API_KEY = os.getenv("COHERE_API_KEY")
if not API_KEY:
    print("ERROR: COHERE_API_KEY not set in environment")
    sys.exit(1)

co = cohere.Client(API_KEY)
MODEL_ID = "command-xlarge-nightly"

# 1) Call the chat API
resp = co.chat(
    model=MODEL_ID,
    message="Say hello to students in a cheerful tone.",
    max_tokens=20,
    temperature=0.7
)

# 2) Extract text from any of the possible response shapes
if hasattr(resp, "generations") and resp.generations:
    text = resp.generations[0].text
elif hasattr(resp, "message"):
    text = resp.message
elif hasattr(resp, "text"):
    text = resp.text
else:
    print("Unexpected response shape:", resp)
    sys.exit(1)

# 3) Print it out
print(f"Model used: {MODEL_ID}")
print("Response:", text.strip())
