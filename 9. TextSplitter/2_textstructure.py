from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader('dl-curriculum.pdf')
docs = loader.load()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
)
# result = splitter.split_text(docs)
result = splitter.split_documents(docs)

for i in range(10):
    print(result[i].page_content)
    print()