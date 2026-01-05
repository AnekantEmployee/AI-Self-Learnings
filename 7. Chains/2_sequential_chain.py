from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

parser = StrOutputParser()
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
template1 = PromptTemplate(
    template='Give me a detailed report on the current state of the {industry}.',
    input_variables=['industry']
)
template2 = PromptTemplate(
    template='Summarize the following text in 100 words: \n {text}',
    input_variables=['text']
)

chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({'industry': "IT"})
chain.get_graph().print_ascii()
print("*" * 100)
print(result) 