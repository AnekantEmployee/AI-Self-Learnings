import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

load_dotenv()

@tool
def currency_convert(base_currency: str):
    """Currency converter from Base Currency Version to different versions of other currencies"""
    response = requests.get(f'https://v6.exchangerate-api.com/v6/95a313af8a9c0deb7be94410/latest/{base_currency}')
    response = response.json()
    return response

query = HumanMessage("Convert 163 USD to INR & Convertion Rate")
messages = [query]

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
# model = model.bind_tools([currency_convert])
response = model.invoke(messages)
messages.append(response)

if response.content:
    print(response.content)
else:
    currency_response = currency_convert.invoke(response.tool_calls[0])
    messages.append(currency_response)

    final_response = model.invoke(messages)
    print(final_response.content)