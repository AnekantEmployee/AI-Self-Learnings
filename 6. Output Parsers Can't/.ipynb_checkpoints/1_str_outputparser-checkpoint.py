from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
        task='text-generation',
        pipeline_kwargs=dict(
            temperature=0.5
        )
    )
model = ChatHuggingFace(llm=llm)


template1 = PromptTemplate(
    template="""Write a detailed report on {topic}""",
    input_variables=['topic']
)
template2 = PromptTemplate(
    template="""Write a 5 line summary on. /n {text}""",
    input_variables=['text']
)

parser = StrOutputParser()

prompt1 = template1.invoke({'topic': "Blackhole"})
result = model.invoke(prompt1)
print(result.content)

prompt2 = template2.invoke({'text': result.content})
result = model.invoke(prompt2)
print(result.content)