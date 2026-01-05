from dotenv import load_dotenv
from dummy_text import document
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

load_dotenv()

# Creating Parsers
parser = StrOutputParser()
json_parser = JsonOutputParser()

# Creating a GEMINI Model
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# Creating a Prompt Template
notes_template = PromptTemplate(
    template='Generate key points notes for this document \n {document}',
    input_variables=['document'],
)
quiz_template = PromptTemplate(
    template='Generate a Quiz for this document having 5 questions and 4 options in each with correct \n {document} \n {get_format_instructions}',
    input_variables=['document'],
    partial_variables={'get_format_instructions': json_parser.get_format_instructions()}
)
final_template = PromptTemplate(
    template='Combine both of these and give me the results \n {notes} & {quiz}',
    input_variables=['notes', 'quiz']
)

# Parallel Chain
parallel_chains = RunnableParallel({
    'notes': notes_template | model | parser,
    'quiz': quiz_template | model | json_parser,
})
results = parallel_chains.invoke({'document': document})
print(results)