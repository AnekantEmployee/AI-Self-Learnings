from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()

@tool
def multiply(a: int, b: int) -> int:
    """Given two numbers return their product"""
    return a * b

# Binding
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
model_with_tools = model.bind_tools([multiply])

# Tool Binding
messages = [HumanMessage(content="What is 3 times 4?")]
result = model_with_tools.invoke(messages)
messages.append(result)
# print(result.tool_calls[0])

# Tool Execution
final_result = multiply.invoke(result.tool_calls[0]) # Returns a Tool Message
messages.append(final_result)
# print(final_result)

last_result = model_with_tools.invoke(messages)
print(last_result)