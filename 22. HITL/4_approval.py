from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch
from typing import TypedDict, Annotated, List
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END, add_messages

load_dotenv()

memory = MemorySaver()

search_tool = TavilySearch(max_results=2)
tools = [search_tool]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm_with_tools = llm.bind_tools(tools=tools)


class BasicState(TypedDict):
    messages: Annotated[List, add_messages]


def model(state: BasicState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def tools_router(state: BasicState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tools"
    else:
        return END


graph = StateGraph(BasicState)
graph.add_node(model, "model")
graph.add_node("tools", ToolNode(tools=tools))

graph.set_entry_point("model")
graph.add_conditional_edges("model", tools_router)

graph.add_edge("tools", "model")

app = graph.compile(checkpointer=memory, interrupt_before=["tools"])

config = {"configurable": {"thread_id": 1}}

events = app.stream(
    {"messages": [HumanMessage(content="What is the current weather in Chennai?")]},
    config=config,
    stream_mode="values",
)

for event in events:
    event["messages"][-1].pretty_print()

# print("\n", "*" * 100, "\n")

snapshot = app.get_state(config=config)
print(snapshot.next)
# print("\n", "*" * 100, "\n")

events = app.stream(None, config, stream_mode="values")
for event in events:
    event["messages"][-1].pretty_print()
