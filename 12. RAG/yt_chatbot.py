from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough

VIDEO_ID = "etnLX7m2MiA"
QUESTION = "Brief what he taught in the video"

load_dotenv()

# Transcript Fetcher
try:
    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.fetch(video_id=VIDEO_ID, languages=['hi'])
    transcript_list = transcript_list.to_raw_data()
    transcript = " ".join(dict(chunk)["text"] for chunk in transcript_list)    
    # print(transcript)
except TranscriptsDisabled:
    print("No captions available for this video.")
    

# Embeddings
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])
# print(chunks[0])
# print(len(chunks))

# Embed generation
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.from_documents(chunks, embeddings)
# print(vector_store.index_to_docstore_id)
# print(vector_store.get_by_ids(['9ea07e20-18ec-410c-a575-5ae1b10b2fcd']))




def format_text(text):
    context_text = "\n\n".join(doc.page_content for doc in text)
    return context_text




# Retreiver
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
# results = retriever.invoke(QUESTION)
# context_text = format_text(results)

# Augmentation
prompt = PromptTemplate(
    template=""""
        You're a helpful assisstent for a YT Video
        Answer only from this script that i am giving to you don't use your knowledge to answer
        If content is not sufficient don't hallucinate

        {context}
        Question: {question}
        Answer in english
    """,
    input_variables=['context', 'question']
)

# Generation
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
parser = StrOutputParser()
model_chain = prompt | model | parser
# final_results = model_chain.invoke({'context': context_text, 'question': QUESTION})
# print(final_results)

# Chains (Optional)
parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_text),
    'question': RunnablePassthrough()
})
final_chain = parallel_chain | model_chain
final_results = final_chain.invoke(QUESTION)
print(final_results)