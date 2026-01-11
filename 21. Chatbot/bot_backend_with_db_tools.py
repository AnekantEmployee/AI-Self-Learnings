import sqlite3
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import tool

import requests
import os

load_dotenv("../.env")


class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


llm = ChatGroq(model="llama-3.1-8b-instant")


conn = sqlite3.connect("chatbot_tools.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

@tool
def search_web(query: str) -> str:
    """Search the web for information using Google Serper API
    
    Args:
        query (str): The search query
        
    Returns:
        str: Search results
    """
    search = GoogleSerperAPIWrapper()
    return search.run(query)


@tool
def calculator(first_num: float, second_num: float, operator: str) -> dict:
    """Do the mathematical operation using this function

    Args:
        first_num (float): first number for the operation
        second_num (float): second number for the operation
        operator (str): operator like add, sub, mul, div, pow

    Returns:
        float: final output
    """

    if operator == "add":
        result = first_num + second_num
    elif operator == "sub":
        result = first_num - second_num
    elif operator == "mul":
        result = first_num * second_num
    elif operator == "pow":
        result = first_num**second_num
    elif operator == "div":
        if second_num == 0:
            return {"error": "Division by zero is not allowed"}
        else:
            result = first_num / second_num
    else:
        return {"error": f"Unsupported operation {operator}"}
    return {
        "first_num": first_num,
        "second_num": second_num,
        "operator": operator,
        "result": result,
    }


@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch stock price of the company by it's symbol

    Args:
        symbol (str): like AAPL, TSLA

    Returns:
        dict: Alpha advantage api output
    """
    api_key = os.getenv("ALPHA_ADVANTAGE")
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    results = requests.get(url)
    return results.json()


tools_list = [search_web, calculator, get_stock_price]
llm = llm.bind_tools(tools=tools_list)

tool_node = ToolNode(tools=tools_list)


def chat_node(state: ChatState):
    messages = state["messages"]

    # Add system message if not present
    if not messages or not isinstance(messages[0], SystemMessage):
        system_msg = SystemMessage(
            "You are a helpful AI assistant. You can have normal conversations and also help with calculations, web searches, and stock information when needed. "
            "Only use tools when the user specifically asks for calculations, searches, or stock prices. For general conversation, respond normally without using any tools."
        )
        messages = [system_msg] + messages

    response = llm.invoke(messages)
    return {"messages": [response]}


def get_chatbot():
    graph = StateGraph(ChatState)
    graph.add_node("chat", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat")
    graph.add_conditional_edges("chat", tools_condition)
    graph.add_edge("tools", "chat")  # crucial step to make it a loop
    graph.add_edge("chat", END)

    return graph.compile(checkpointer=checkpointer)


def get_threads():
    threads = set()
    for checkpoint in list(checkpointer.list(None)):
        threads.add(checkpoint.config["configurable"]["thread_id"])

    return list(threads)


### NORMAL WORKFLOW
# chatbot = get_chatbot()
# config = {"configurable": {"thread_id": "1"}}
# result = chatbot.invoke(
#     {
#         "messages": [
#             HumanMessage(
#                 "what is stock price of apple and how much it would cost to purchase 50 shares"
#             )
#         ]
#     },
#     config,
# )
# print(result["messages"][-1].content)


#### STREAMLING
# chatbot = get_chatbot()
# config = {"configurable": {"thread_id": "1"}}
# result = chatbot.stream(
#     {"messages": [HumanMessage("Tell me a 100 words esssay on AI")]},
#     config=config,
#     stream_mode="messages",
# )
# print(result)  # Gives a generator object
# for chunk, metadata in result:
#     if chunk.content:
#         print(chunk.content)
