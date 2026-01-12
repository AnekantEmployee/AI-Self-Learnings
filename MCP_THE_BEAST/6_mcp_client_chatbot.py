import os
import asyncio
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()
FASTMCP_API_KEY = os.getenv("FASTMCP_API_KEY")

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
    "expense-tracker-anekant": {
        "transport": "streamable_http",  # if fails sse
        "url": "https://expense-tracker-anekant.fastmcp.app/mcp",
        "headers": {"Authorization": f"Bearer {FASTMCP_API_KEY}"},
    },
}


async def main():

    client = None
    try:
        client = MultiServerMCPClient(SERVERS)
        tools_list = await client.get_tools()

        print(f"Successfully loaded {len(tools_list)} tools:")

        named_tools = {}
        for tool in tools_list:
            named_tools[tool.name] = tool

        print(f"List of available tools", list(named_tools.keys()))

        llm = ChatGroq(model="llama-3.1-8b-instant")
        llm_with_tools = llm.bind_tools(tools_list)

        system_msg = SystemMessage(
            content="Always use date format we are currently in 2026. Convert natural language dates to DD-MM-YYYY format."
        )
        question = "What all expenses I did on 10-01-2026 it's total and divide it in between of two ppls tool"

        messages = [system_msg, question]
        
        while True:
            response = await llm_with_tools.ainvoke(messages)
            
            if not response.tool_calls:
                print("Final Response:", response.content)
                break
                
            messages.append(response)
            
            for tool in response.tool_calls:
                print("Calling Tool:", tool["name"])
                print("Tool Arguments:", tool["args"])
                results = await named_tools[tool["name"]].ainvoke(tool["args"])
                print(results)
                tool_message = ToolMessage(
                    content=str(results), tool_name=tool["name"], tool_call_id=tool["id"]
                )
                messages.append(tool_message)

    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if client:
            try:
                await client.close()
            except:
                pass


if __name__ == "__main__":
    asyncio.run(main())
