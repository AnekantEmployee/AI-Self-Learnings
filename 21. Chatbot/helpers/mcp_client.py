from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool
from .async_utils import run_async
import os

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

try:
    client = MultiServerMCPClient(SERVERS)
except Exception as e:
    print(f"Failed to initialize MCP client: {e}")
    client = None


def load_mcp_tools() -> list[BaseTool]:
    if client is None:
        print("MCP client not initialized")
        return []
    
    try:
        # Add timeout and proper error handling
        import asyncio
        async def get_tools_with_timeout():
            try:
                return await asyncio.wait_for(client.get_tools(), timeout=10.0)
            except asyncio.TimeoutError:
                print("MCP tools loading timed out")
                return []
            except Exception as e:
                print(f"Error getting MCP tools: {e}")
                return []
        
        tools = run_async(get_tools_with_timeout())
        print(f"Loaded {len(tools)} MCP tools: {[tool.name for tool in tools]}")
        return tools
    except Exception as e:
        print(f"Failed to load MCP tools: {e}")
        return []