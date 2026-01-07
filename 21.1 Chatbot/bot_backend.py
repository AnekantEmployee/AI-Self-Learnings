from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()


class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


llm = ChatGroq(model="llama-3.1-8b-instant")
checkpointer = MemorySaver()


def chat_node(state: ChatState):
    messages = state["messages"]

    # Add system message if not present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage("You're a helpful chatbot")] + messages

    response = llm.invoke(messages)
    return {"messages": [response]}


def get_chatbot():
    graph = StateGraph(ChatState)
    graph.add_node("chat", chat_node)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    return graph.compile(checkpointer=checkpointer)


### NORMAL WORKFLOW
# chatbot = get_chatbot()
# config = {"configurable": {"thread_id": "1"}}
# result = chatbot.invoke(
#     {"messages": [HumanMessage("Tell me a 100 words esssay on AI")]}, config
# )
# print(result)


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