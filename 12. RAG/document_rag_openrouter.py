import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda,
    RunnablePassthrough,
)

# Configuration
TXT_FILE_PATH = "document.txt"
QUESTION = "When I watched movie?"

load_dotenv()

# Setup OpenRouter
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"


def format_text(text):
    return "\n\n".join(doc.page_content for doc in text)


def load_text_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


import time

start_time = time.time()


def hierarchical_retrieval(query, vector_store):
    # Stage 1: Broad retrieval (20 candidates)
    broad_docs = vector_store.similarity_search(query, k=20)
    
    # Stage 2: Re-ranking with cross-encoder (top 4)
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [[query, doc.page_content] for doc in broad_docs]
    scores = cross_encoder.predict(pairs)
    
    # Stage 3: Get top 4 re-ranked documents
    ranked_docs = sorted(zip(broad_docs, scores), key=lambda x: x[1], reverse=True)[:4]
    return [doc for doc, score in ranked_docs]

# Load and process document
text_content = load_text_file(TXT_FILE_PATH)

if text_content:
    # Create chunks and vector store
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.create_documents([text_content])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-l6-v2"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Setup OpenRouter model
    model = ChatOpenAI(
        model="nvidia/nemotron-3-nano-30b-a3b:free",
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        temperature=0.7,
    )

    # Create prompt and chain
    prompt = PromptTemplate(
        template="""Answer based only on the provided document content.

Document Content:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"],
    )


    # Build hierarchical RAG chain with context printing
    def print_context(data):
        print("=" * 50)
        print("HIERARCHICAL RETRIEVED CONTEXT:")
        print("=" * 50)
        print(data["context"])
        print("=" * 50)
        return data

    chain = (
        RunnableParallel(
            {
                "context": RunnableLambda(lambda q: format_text(hierarchical_retrieval(q, vector_store))),
                "question": RunnablePassthrough(),
            }
        )
        | RunnableLambda(print_context)
        | prompt
        | model
        | StrOutputParser()
    )

    # Get answer
    result = chain.invoke(QUESTION)
    print(time.time() - start_time)
    print("=" * 50)
    print("ANSWER:")
    print("=" * 50)
    print(result)
    print("=" * 50)
else:
    print("Failed to load document.txt")
