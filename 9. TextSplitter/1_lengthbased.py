from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

loader = PyPDFLoader('dl-curriculum.pdf')
docs = loader.load()


splitter = CharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separator=''
)
# result = splitter.split_text(docs)
result = splitter.split_documents(docs)
print(result[0].page_content)