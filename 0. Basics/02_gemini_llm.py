from google import genai
from google.genai.types import GenerateContentConfig
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

response = client.models.generate_content(
    model="gemini-2.5-flash", 
    contents="Predit the next word: I love  ...",
    config=GenerateContentConfig(
        temperature=1,         
        top_p=0.95,
        top_k=30,
        # candidate_count=1,
        # response_mime_type="application/json",
        # top_p=0.95,
        # top_k=20,
        # seed=5,
        # max_output_tokens=500,
        stop_sequences=["you"],
        # presence_penalty=0.0,
        # frequency_penalty=0.0,
    )
)
print(response.text)