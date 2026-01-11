import asyncio
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv

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
        "transport": "streamable_http",
        "url": "https://expense-tracker-anekant.fastmcp.app/mcp",
        "headers": {"Authorization": f"Bearer {FASTMCP_API_KEY}"},
    },
}

async def test_mcp():
    client = MultiServerMCPClient(SERVERS)
    
    try:
        print("Testing MCP connection...")
        tools = await client.get_tools()
        print(f"Successfully loaded {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
        return tools
    except Exception as e:
        print(f"MCP connection failed: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    asyncio.run(test_mcp())