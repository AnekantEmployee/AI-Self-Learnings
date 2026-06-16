from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

load_dotenv()


class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages] # Add messages is the reducer to store the values


prompt_template = ChatPromptTemplate(
    [
        (SystemMessage, "You're a helpful chatbot"),
        (HumanMessage, "Write a response to this message: {message}"),
    ]
)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
output_parser = StrOutputParser()
chain = prompt_template | llm | output_parser


def chat_node(state: ChatState):
    messages = state["messages"]
    print()
    print(messages)
    print()
    response = chain.invoke(messages)

    return {"messages": [AIMessage(response)]}


graph = StateGraph(ChatState)

graph.add_node("chat", chat_node)

graph.add_edge(START, "chat")
graph.add_edge("chat", END)

chatbot = graph.compile()

while True:
    text = input("User: ")
    text = text.strip()
    
    if text.lower() == 'exit':
        print("Bye")
        break
    
    results = chatbot.invoke({"messages": [HumanMessage(text)]})
    print(f"AI:", results["messages"][-1].content)
    
    