import time
from langchain import hub
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor


load_dotenv()


@tool
def get_current_time():
    """Getting current date and time"""
    return time.ctime()


prompt = hub.pull("hwchase17/react")
search_tool = TavilySearch(max_results=4)
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
tools = [search_tool, get_current_time]

agent = create_react_agent(model, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent, tools=tools, handle_parsing_errors=True, verbose=True
)


def get_agent_search_results(query):
    """Run the agent and return both the answer and the raw search data"""
    result = agent_executor.invoke({"input": query})
    search_results = search_tool.invoke({"query": query})

    return {"agent_answer": result["output"], "search_results": search_results}


result = get_agent_search_results(
    "after how much days in the next match of india and with which team"
)
print("\n")
print(result)
