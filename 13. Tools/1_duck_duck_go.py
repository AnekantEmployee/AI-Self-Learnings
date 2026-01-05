from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun(output_format="list")

results = search_tool.invoke('Today highest volume stock on bse')

print(results)