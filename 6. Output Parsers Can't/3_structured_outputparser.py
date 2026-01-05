from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

load_dotenv()

schema = [
    ResponseSchema(name='Fact 1', description='Fact 1 about the topic'),
    ResponseSchema(name='Fact 2', description='Fact 2 about the topic'),
    ResponseSchema(name='Fact 3', description='Fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
prompt_template = PromptTemplate(
    template='Give me 3 topics on the {topic} \n {format_instructions}',
    input_variables=['topic'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

print(parser.get_format_instructions())
print("*" * 100)
chain = prompt_template | model | parser
response = chain.invoke({'topic': "Agentic AI"})
print(response)