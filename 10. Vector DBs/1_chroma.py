from documents import docs
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-exp-03-07')

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory='my_chroma_db',
    collection_name='sample'
)
vector_store.add_documents(docs)
results = vector_store.get(include=['embeddings','documents', 'metadatas'])
print(results)

# search documents
search_results = vector_store.similarity_search(
    query='Who among these are a bowler?',
    k=2
)
print(search_results)

# search with similarity score
search_results = vector_store.similarity_search_with_score(
    query='Who among these are a bowler?',
    k=2
)
print(search_results)

# meta-data filtering
search_results = vector_store.similarity_search_with_score(
    query="",
    filter={"team": "Chennai Super Kings"}
)
print(search_results)

# update documents
updated_doc1 = Document(
    page_content="Virat Kohli, the former captain of Royal Challengers Bangalore (RCB), is renowned for his aggressive leadership and consistent batting performances. He holds the record for the most runs in IPL history, including multiple centuries in a single season. Despite RCB not winning an IPL title under his captaincy, Kohli's passion and fitness set a benchmark for the league. His ability to chase targets and anchor innings has made him one of the most dependable players in T20 cricket.",
    metadata={"team": "Royal Challengers Bangalore"}
)
vector_store.update_document(document_id='09a39dc6-3ba6-4ea7-927e-fdda591da5e4', document=updated_doc1)

# delete document
vector_store.delete(ids=['09a39dc6-3ba6-4ea7-927e-fdda591da5e4'])