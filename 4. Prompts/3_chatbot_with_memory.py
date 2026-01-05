from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

messages = [
    SystemMessage(content="You're a helpful AI Assistent")
]

input_val = input("You: ")

while input_val != "quit":
    messages.append(HumanMessage(content=input_val))   
    
    response = model.invoke(f'Always give the output in max 5 words for this prompt: {messages}')
    messages.append(AIMessage(content=response.content))
    print("AI:", response.content)
    
    input_val = input("You: ")

    
print(messages)