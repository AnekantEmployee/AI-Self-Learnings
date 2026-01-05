from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver, InMemorySaver
from typing import TypedDict 

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

class JokeState(TypedDict):
    topic: str
    joke: str
    explain: str
    
graph = StateGraph(JokeState)

def generate_joke(state: JokeState) -> JokeState:
    response = model.invoke(f"Generate a Hinglish Joke in short for: {state['topic']}")
    return {'joke': response.content}

def generate_explaination(state: JokeState) -> JokeState:
    response = model.invoke(f"Generate a explaination for the Joke in short: {state['joke']}")
    return {'explain': response.content}

graph.add_node('generate_joke', generate_joke)
graph.add_node('generate_explaination', generate_explaination)

graph.add_edge(START, 'generate_joke')
graph.add_edge('generate_joke', 'generate_explaination')
graph.add_edge('generate_explaination', END)

checkpointer = InMemorySaver()
bot = graph.compile(checkpointer=checkpointer)

config1 = {'configurable': {'thread_id': '1'}}
result = bot.invoke({'topic': 'Python'}, config=config1)

print(result)
print("*" * 100)
print(bot.get_state(config1))
print("*" * 100)
print(list(bot.get_state_history(config1)))