from langchain_community.document_loaders import TextLoader

loader = TextLoader('data/cricket_data.txt', encoding='UTF-8')

docs = loader.load()

print(docs)