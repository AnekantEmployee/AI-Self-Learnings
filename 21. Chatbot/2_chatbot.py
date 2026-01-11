from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()


class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


def chat_node(state: ChatState):
    messages = state["messages"]

    response = llm.invoke(messages)

    return {"messages": [response]}


checkpointer = MemorySaver()
graph = StateGraph(ChatState)

graph.add_node("chat", chat_node)

graph.add_edge(START, "chat")
graph.add_edge("chat", END)

chatbot = graph.compile(checkpointer=checkpointer)

thread_id = "1"
while True:
    text = input("User: ")
    text = text.strip()

    if text.lower() == "exit":
        print("Bye")
        break

    config = {"configurable": {"thread_id": thread_id}}
    results = chatbot.invoke({"messages": [HumanMessage(text)]}, config=config)
    print(f"AI:", results["messages"][-1].content)
