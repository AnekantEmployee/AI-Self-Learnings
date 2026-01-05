from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

parser = StrOutputParser()
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
template = PromptTemplate(
    template='Tell me the meaning of this cyber security term in 5 points for {term}',
    input_variables=['term']
)

chain = template | model | parser

result = chain.invoke({'term':'GRC'})
chain.get_graph().print_ascii() # Visualize the steps
print("*" * 100)
print(result)