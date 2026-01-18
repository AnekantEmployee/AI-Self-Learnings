from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough

# Configuration
TXT_FILE_PATH = "document.txt"  # Change this to your text file path
QUESTION = "What is the main topic discussed in this document?"

load_dotenv()

def format_text(text):
    """Format retrieved documents into a single context string"""
    context_text = "\n\n".join(doc.page_content for doc in text)
    return context_text

def load_text_file(file_path):
    """Load text content from a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

# Load the text file
text_content = load_text_file(TXT_FILE_PATH)

if text_content:
    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text_content])
    
    # Create embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Create retriever
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    
    # Define prompt template
    prompt = PromptTemplate(
        template="""
            You're a helpful assistant for document Q&A.
            Answer only from the provided document content. Don't use external knowledge.
            If the content is not sufficient to answer the question, say so clearly.

            Document Content:
            {context}
            
            Question: {question}
            
            Answer in English:
        """,
        input_variables=['context', 'question']
    )
    
    # Create model and chain
    model = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp')
    parser = StrOutputParser()
    model_chain = prompt | model | parser
    
    # Create parallel chain for context retrieval
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_text),
        'question': RunnablePassthrough()
    })
    
    # Final chain
    final_chain = parallel_chain | model_chain
    
    # Get answer
    final_results = final_chain.invoke(QUESTION)
    print("\n" + "="*50)
    print("ANSWER:")
    print("="*50)
    print(final_results)
    print("="*50)
else:
    print("Failed to load text file. Please check the file path.")