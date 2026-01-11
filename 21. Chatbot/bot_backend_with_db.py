import sqlite3
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

load_dotenv("../.env")


class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


llm = ChatGroq(model="llama-3.1-8b-instant")


conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


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


def get_threads():
    threads = set()
    for checkpoint in list(checkpointer.list(None)):
        threads.add(checkpoint.config["configurable"]["thread_id"])

    return list(threads)


### NORMAL WORKFLOW
# chatbot = get_chatbot()
# config = {"configurable": {"thread_id": "1"}}
# result = chatbot.invoke(
#     {"messages": [HumanMessage("what is my name")]}, config
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
