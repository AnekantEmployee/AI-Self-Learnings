from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import atexit

load_dotenv()

# Fix for gRPC cleanup issue
def cleanup_grpc():
    pass  # Simple no-op cleanup

atexit.register(cleanup_grpc)


class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


llm = ChatGroq(model="llama-3.1-8b-instant")

pdf_path = os.path.join(os.path.dirname(__file__), "..", "data", "Resume.pdf")

loader = PyPDFLoader(pdf_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = FAISS.from_documents(chunks, embeddings)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})


@tool
def rag_tool(query: str):
    """Retrieve relevent information from the documents
    Use this tool when the user asks about the facts of concepts
    that might be stored in this documents"""
    result = retriever.invoke(query)

    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {"query": query, "context": context, "metadata": metadata}


tools = [rag_tool]
llm = llm.bind_tools(tools)
tool_node = ToolNode(tools=tools)


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

    return graph.compile()


chatbot = get_chatbot()
result = chatbot.invoke(
    {"messages": [HumanMessage("Tell me about yash technologies from the document")]},
)
print(result["messages"][-1].content)
