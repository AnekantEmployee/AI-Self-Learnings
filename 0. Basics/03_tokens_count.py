import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()                          # loads GEMINI_API_KEY from .env

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_ID       = "gemini-2.5-flash"
URL            = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:countTokens"

# ------------------------------------------------------------------
# 1. Build the exact payload you would send to generateContent
# ------------------------------------------------------------------
pdf_text = "Paste the full document text here"
question = "What is the candidate's highest degree?"

payload = {
    "contents": [
        {
            "parts": [
                {
                    "text": (
                        "You are a helpful assistant. Answer the following "
                        "question using only the document below.\n\n"
                        f"Document:\n{pdf_text}\n\n"
                        f"Question: {question}"
                    )
                }
            ]
        }
    ]
}

# ------------------------------------------------------------------
# 2. Call the countTokens endpoint
# ------------------------------------------------------------------
headers = {
    "Content-Type": "application/json",
    "x-goog-api-key": GEMINI_API_KEY,
}

response = requests.post(URL, headers=headers, data=json.dumps(payload))
response.raise_for_status()

token_info = response.json()
input_tokens = token_info["totalTokens"]

print(f"Input tokens: {input_tokens}")