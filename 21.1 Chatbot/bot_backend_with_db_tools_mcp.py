from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool, BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_community.utilities import GoogleSerperAPIWrapper
from dotenv import load_dotenv
import aiosqlite
import requests
import asyncio
import threading
import os
from typing import List

load_dotenv()

# Dedicated async loop for backend tasks
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)


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
client = MultiServerMCPClient(SERVERS)


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
def get_stock_price(symbol: str) -> str:
    """Fetch current stock price for a company symbol like TSLA, AAPL, etc.

    Args:
        symbol (str): Stock symbol (e.g., TSLA for Tesla, AAPL for Apple)

    Returns:
        str: Current stock price information
    """
    try:
        api_key = os.getenv("ALPHA_ADVANTAGE")
        if not api_key:
            return f"Error: Alpha Vantage API key not found. Please set ALPHA_ADVANTAGE environment variable."

        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        # Check for API errors
        if "Error Message" in data:
            return f"Error: {data['Error Message']}"

        if "Note" in data:
            return f"API Limit: {data['Note']}"

        # Extract quote data
        quote = data.get("Global Quote", {})
        if not quote:
            return f"No data found for symbol {symbol}"

        price = quote.get("05. price", "N/A")
        change = quote.get("09. change", "N/A")
        change_percent = quote.get("10. change percent", "N/A")

        return f"Stock: {symbol}\nPrice: ${price}\nChange: {change} ({change_percent})"

    except requests.exceptions.RequestException as e:
        return f"Network error fetching stock data: {str(e)}"
    except Exception as e:
        return f"Error fetching stock price: {str(e)}"


def load_mcp_tools() -> list[BaseTool]:
    try:
        tools = run_async(client.get_tools())
        print(f"Loaded {len(tools)} MCP tools: {[tool.name for tool in tools]}")
        return tools
    except Exception as e:
        print(f"Failed to load MCP tools: {e}")
        return []


mcp_tools = load_mcp_tools()
tools_list = [search_web, get_stock_price, *mcp_tools]
print(f"Total tools available: {len(tools_list)}")
print(f"Tool names: {[getattr(tool, 'name', str(tool)) for tool in tools_list]}")
llm_with_tools = llm.bind_tools(tools=tools_list)

tool_node = ToolNode(tools_list, handle_tool_errors=True) if tools_list else None


async def chat_node(state: ChatState):
    messages = state["messages"]

    # Add system message if not present
    if not messages or not isinstance(messages[0], SystemMessage):
        system_msg = SystemMessage(
            "You are a helpful AI assistant. You can have normal conversations and also help with calculations, web searches, and stock information when needed. "
            "Only use tools when the user specifically asks for calculations, searches, or stock prices. For general conversation, respond normally without using any tools."
            "For expense tracking, use the available MCP tools. When listing expenses, try different parameter formats if one fails."
            "Always use date format we are currently in 2026. Convert natural language dates to DD-MM-YYYY format."
        )
        messages = [system_msg] + messages

    try:
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}
    except Exception as e:
        # If tool calling fails, try without tools
        try:
            fallback_llm = ChatGroq(model="llama-3.1-8b-instant")
            response = await fallback_llm.ainvoke(messages)
            return {"messages": [response]}
        except Exception as fallback_error:
            # Return error message as AI response
            from langchain_core.messages import AIMessage

            error_msg = AIMessage(
                content=f"I apologize, but I'm experiencing technical difficulties. Error: {str(e)}"
            )
            return {"messages": [error_msg]}


async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot_mcp.db")
    return AsyncSqliteSaver(conn)


checkpointer = run_async(_init_checkpointer())

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def retrieve_all_threads():
    return run_async(_alist_threads())
