from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class LLMQa(TypedDict):
    question: str
    answer: str
    
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

graph = StateGraph(LLMQa)

def get_answer(state: LLMQa) -> LLMQa:
    question = state['question']
    
    result = model.invoke(question)
    state['answer'] = result.content
    
    return state

graph.add_node('get_answer', get_answer)

graph.add_edge(START, 'get_answer')
graph.add_edge('get_answer', END)

workflow = graph.compile()
result = workflow.invoke({'question': "Capital of MP"})
print(result)