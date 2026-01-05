import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


search_tool = TavilySearchResults(max_results=2)
tools = [search_tool]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm_with_tools = llm.bind_tools(tools=tools)


def model(state: AgentState):
    return {
        "messages": [llm_with_tools.invoke(state["messages"])],
    }


def tools_router(state: AgentState):
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tool_node"
    else:
        return END


tool_node = ToolNode(tools=tools)
graph = StateGraph(AgentState)

graph.add_node("model", model)
graph.add_node("tool_node", tool_node)
graph.set_entry_point("model")

graph.add_conditional_edges("model", tools_router)
graph.add_edge("tool_node", "model")

app = graph.compile()


async def run_agent():
    input_data = {"messages": ["Hi, how are you?"]}

    # Option 1: Using astream_events for detailed streaming
    events = app.astream_events(input=input_data, version="v2")

    async for event in events:
        if event["event"] == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                print(content, end="", flush=True)

    print()  # New line after streaming


# Alternative simpler approach
async def run_agent_simple():
    input_data = {"messages": ["Hi, how are you?"]}

    # Simple streaming approach
    async for chunk in app.astream(input_data):
        for node_name, node_output in chunk.items():
            if "messages" in node_output:
                last_message = node_output["messages"][-1]
                if hasattr(last_message, "content"):
                    print(f"[{node_name}]: {last_message.content}")


# Run the agent
if __name__ == "__main__":
    print("Running with detailed streaming:")
    asyncio.run(run_agent())

    print("\nRunning with simple approach:")
    asyncio.run(run_agent_simple())
