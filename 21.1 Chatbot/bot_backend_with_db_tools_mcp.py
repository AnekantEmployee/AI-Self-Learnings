import aiosqlite
import asyncio
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import tool

import requests
import os

load_dotenv("../.env")

FASTMCP_API_KEY = os.getenv("FASTMCP_API_KEY")

# PATCH: Add is_alive method to aiosqlite.Connection
if not hasattr(aiosqlite.Connection, "is_alive"):

    def is_alive(self):
        """Check if the connection is alive"""
        return self._conn is not None

    aiosqlite.Connection.is_alive = is_alive


class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


llm = ChatGroq(model="llama-3.1-8b-instant")


SERVERS = {
    "math": {
        "transport": "stdio",
        "command": "C:/Users/anekant.jain/AppData/Local/Programs/Python/Python311/Scripts/uv.exe",
        "args": [
            "run",
            "fastmcp",
            "run",
            "D:/Agentic AI/Self-Learning/MCP_THE_BEAST/main.py",
        ],
    },
    "expense-tracker-anekant": {
        "transport": "streamable_http",
        "url": "https://expense-tracker-anekant.fastmcp.app/mcp",
        "headers": {"Authorization": f"Bearer {FASTMCP_API_KEY}"},
    },
}


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


async def chat_node(state: ChatState):
    messages = state["messages"]

    # Add system message if not present
    if not messages or not isinstance(messages[0], SystemMessage):
        system_msg = SystemMessage(
            "You are a helpful AI assistant. You can have normal conversations and also help with calculations, web searches, and stock information when needed. "
            "Only use tools when the user specifically asks for calculations, searches, or stock prices. For general conversation, respond normally without using any tools."
        )
        messages = [system_msg] + messages

    response = await llm.ainvoke(messages)
    return {"messages": [response]}


def build_graph():
    """Build the graph without checkpointer"""
    tool_node = ToolNode(tools=tools_list)

    graph = StateGraph(ChatState)

    graph.add_node("chat", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat")
    graph.add_conditional_edges("chat", tools_condition)
    graph.add_edge("tools", "chat")
    graph.add_edge("chat", END)
    
    return graph


async def get_threads():
    """Get list of all thread IDs"""
    async with AsyncSqliteSaver.from_conn_string("chatbot_mcp.db") as checkpointer:
        threads = set()
        async for checkpoint in checkpointer.list(None):
            threads.add(checkpoint.config["configurable"]["thread_id"])
        return list(threads)


## NORMAL WORKFLOW
async def main():
    # Use the context manager to keep the checkpointer alive during execution
    async with AsyncSqliteSaver.from_conn_string("chatbot_mcp.db") as checkpointer:
        graph = build_graph()
        chatbot = graph.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "1"}}
        result = await chatbot.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        "what is stock price of apple and how much it would cost to purchase 50 shares"
                    )
                ]
            },
            config,
        )
        print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
