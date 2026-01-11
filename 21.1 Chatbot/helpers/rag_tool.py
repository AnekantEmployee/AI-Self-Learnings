from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import atexit
import asyncio

# Fix for gRPC cleanup issue
def cleanup_grpc():
    pass

atexit.register(cleanup_grpc)

# Global variables for RAG components
embeddings = None
vector_store = None
retriever = None


def initialize_rag(pdf_path: str):
    """Initialize RAG components with a PDF file"""
    global vector_store, retriever, embeddings
    
    try:
        # Ensure event loop exists
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Initialize embeddings only when needed
        if embeddings is None:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        
        return True
    except Exception as e:
        print(f"Error initializing RAG: {e}")
        return False


@tool
def rag_search(query: str) -> str:
    """Retrieve relevant information from documents
    Use this tool when the user asks about facts or concepts that might be stored in documents"""
    global retriever
    
    if retriever is None:
        return "No documents loaded. Please upload a PDF document first."
    
    try:
        result = retriever.invoke(query)
        context = [doc.page_content for doc in result]
        
        if not context:
            return "No relevant information found in the documents."
        
        return f"Retrieved information:\n" + "\n\n".join(context)
    except Exception as e:
        return f"Error retrieving information: {str(e)}"