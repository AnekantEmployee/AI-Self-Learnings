from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('data/SOC.pdf')

docs = loader.load()

print(docs[2].page_content)