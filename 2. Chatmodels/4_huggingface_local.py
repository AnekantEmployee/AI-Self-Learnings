from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
        task='text-generation',
        pipeline_kwargs=dict(
            temperature=0.5
        )
    )

model = ChatHuggingFace(llm=llm)

result = model.invoke("Tell me about Indore")

print(result.content)