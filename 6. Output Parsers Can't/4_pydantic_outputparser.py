from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

class Person(BaseModel):
    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person')
    movies: list[str] = Field("List of movies")

parser = PydanticOutputParser(pydantic_object=Person)
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
prompt_template = PromptTemplate(
    template='Generate the name, age and movies of a bollywood actor in {place} \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

print(parser.get_format_instructions())
print("*" * 100)
chain = prompt_template | model | parser
response = chain.invoke({'place': "India"})
print(response)