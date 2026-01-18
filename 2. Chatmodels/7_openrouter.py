import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

llm = ChatOpenAI(
    model="nvidia/nemotron-3-nano-30b-a3b:free",  # Example model
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
    temperature=0.7,
)

import time
start_time = time.time()
result = llm.invoke("Capital of india in 1000 words")
print(time.time() - start_time)
print(result.content[:10])
