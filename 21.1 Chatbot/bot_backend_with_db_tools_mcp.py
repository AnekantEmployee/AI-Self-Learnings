from typing import List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage

from helpers.mcp_client import load_mcp_tools
from helpers.tools import search_web, get_stock_price
from helpers.rag_tool import rag_search, initialize_rag
from helpers.database import get_checkpointer, retrieve_all_threads

load_dotenv()


class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


llm = ChatGroq(model="llama-3.1-8b-instant")


mcp_tools = load_mcp_tools()
tools_list = [search_web, get_stock_price, rag_search, *mcp_tools]
print(f"Total tools available: {len(tools_list)}")
print(f"Tool names: {[getattr(tool, 'name', str(tool)) for tool in tools_list]}")
llm_with_tools = llm.bind_tools(tools=tools_list)

tool_node = ToolNode(tools_list, handle_tool_errors=True) if tools_list else None


async def chat_node(state: ChatState):
    messages = state["messages"]

    # Add system message if not present
    if not messages or not isinstance(messages[0], SystemMessage):
        system_msg = SystemMessage(
            "You are a helpful AI assistant. You can have normal conversations and also help with calculations, web searches, stock information, document retrieval, and expense tracking when needed. "
            "Only use tools when the user specifically asks for calculations, searches, stock prices, document information, or expense operations. For general conversation, respond normally without using any tools."
            "For document retrieval, use the rag_search tool when users ask about information that might be in documents. "
            "For expense tracking, use the available MCP tools (list_expenses, add_expense, delete_expense, summarize_expenses). "
            "IMPORTANT: For dates, always use YYYY-MM-DD format (e.g., 2026-01-11 for January 11th, 2026). We are currently in 2026. "
            "When a user mentions dates like '11th jan' or 'January 11th', convert to 2026-01-11 format. "
            "For list_expenses tool: DO NOT use 'date' parameter - it's not supported. Use 'start_date' and 'end_date' instead. "
            "For single day queries, set both start_date and end_date to the same date."
            "If one date format fails, try alternative formats like MM-DD-YYYY or DD-MM-YYYY."
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
            error_msg = AIMessage(
                content=f"I apologize, but I'm experiencing technical difficulties. Error: {str(e)}"
            )
            return {"messages": [error_msg]}


checkpointer = get_checkpointer()

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


def get_all_threads():
    return list(reversed(retrieve_all_threads(checkpointer)))
