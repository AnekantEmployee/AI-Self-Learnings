from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
parser = JsonOutputParser()
template = PromptTemplate(
    template='Give me 5 facts about {topic} \n {format_instruction}',
    input_variable=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

print(parser.get_format_instructions())
print("*" * 100)
chain = template | model | parser
result = chain.invoke({'topic': "Black Hole"})
print(result)