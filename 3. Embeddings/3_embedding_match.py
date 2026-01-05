import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents=[
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]
question="Tell me about bumrah"

doc_embeddings = embedding.embed_documents(documents)
query_embeddings = embedding.embed_query(question)

match_score = cosine_similarity([query_embeddings], doc_embeddings)[0]
match_index = sorted(list(enumerate(match_score)), key=lambda x: x[1])[-1][0]
print(documents[match_index])