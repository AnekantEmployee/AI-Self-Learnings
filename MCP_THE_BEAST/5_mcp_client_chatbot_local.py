import asyncio
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import ToolMessage

load_dotenv()

SERVERS = {
    "math": {
        "transport": "stdio",
        "command": "C:/Users/anekant.jain/AppData/Local/Programs/Python/Python311/Scripts/uv.exe",
        "args": [
            "run",
            "fastmcp",
            "run",
            "D:/Agentic AI/Self-Learning/MCP_THE_BEAST/main.py",
        ],
    },
}


async def main():
    try:
        client = MultiServerMCPClient(SERVERS)
        tools_list = await client.get_tools()

        named_tools = {}
        for tool in tools_list:
            named_tools[tool.name] = tool

        llm = ChatGroq(model="llama-3.1-8b-instant")
        llm_with_tools = llm.bind_tools(tools_list)

        question = "Roll a dice and make the square of it"
        response = await llm_with_tools.ainvoke(question)

        tool_messages = []
        for tool in response.tool_calls:
            print("Calling Tool:", tool["name"])
            print("Tool Arguments:", tool["args"])
            results = await named_tools[tool["name"]].ainvoke(tool["args"])
            print(results)
            tool_message = ToolMessage(
                content=str(results), tool_name=tool["name"], tool_call_id=tool["id"]
            )
            tool_messages.append(tool_message)

        messages = [question, response] + tool_messages
        response = await llm_with_tools.ainvoke(messages)
        print(response.content)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
