from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, BaseMessage

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

graph = StateGraph(ChatState)

def chat(state: ChatState) -> ChatState:
    messages = state['messages']
    response = model.invoke(messages)
    
    return {'messages': [response]}

graph.add_node('chat', chat)

graph.add_edge(START, 'chat')
graph.add_edge('chat', END)

checkpointer = MemorySaver()
chatbot = graph.compile(checkpointer=checkpointer)

thread_id = '1'
while True:
    input_text = input("User: ")
    config = {'configurable': {'thread_id': thread_id}}
    
    if input_text.strip().lower() == 'exit':
        print(chatbot.get_state(config=config))
        break
    
    result = chatbot.invoke({'messages': [HumanMessage(content=input_text)]}, config=config)
    
    print("AI: ", result['messages'][-1].content)
    
