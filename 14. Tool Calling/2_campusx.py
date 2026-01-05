import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_core.tools import InjectedToolArg
from typing import Annotated

load_dotenv()

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
  """
  This function fetches the currency conversion factor between a given base currency and a target currency
  """
  url = f'https://v6.exchangerate-api.com/v6/c754eab14ffab33112e380ca/pair/{base_currency}/{target_currency}'

  response = requests.get(url)

  return response.json()

@tool
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
  """
  given a currency conversion rate this function calculates the target currency value from a given base currency value
  """

  return base_currency_value * conversion_rate

query = HumanMessage("Convert 163 USD to INR & Convertion Rate")
messages = [query]

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
llm_with_tools = model.bind_tools([get_conversion_factor, convert])

ai_message = llm_with_tools.invoke(messages)
messages.append(ai_message)
print(ai_message.tool_calls)