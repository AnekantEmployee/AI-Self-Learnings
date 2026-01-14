import os
import asyncio
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage

load_dotenv()

SERVERS = {
    "terminal_filesystem": {
        "transport": "stdio",
        "command": "C:/Users/anekant.jain/AppData/Local/Programs/Python/Python311/Scripts/uv.exe",
        "args": [
            "run",
            "fastmcp",
            "run",
            "D:/Agentic AI/Self-Learning/mcps_new_course/1_filesystem.py",
        ],
    },
}

BASE_DIR = "D:/Agentic AI/Self-Learning/mcps_new_course/"

async def main():
    client = MultiServerMCPClient(SERVERS)
    tools_list = await client.get_tools()
    named_tools = {tool.name: tool for tool in tools_list}
    llm_with_tools = ChatGoogleGenerativeAI(model="gemini-2.5-flash").bind_tools(tools_list)

    print("Filesystem Chat Assistant")
    print(f"Base Directory: {BASE_DIR}\n")

    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit"]:
            break

        system_msg = SystemMessage(content=f"You are a helpful assistant. Use Unix or shell commands to perform the operations. When base_dir is needed, use '{BASE_DIR}'.")
        messages = [system_msg, HumanMessage(content=question)]

        while True:
            response = await llm_with_tools.ainvoke(messages)
            if not response.tool_calls:
                print(f"\nAssistant: {response.content}\n")
                break
            messages.append(response)
            for tool in response.tool_calls:
                print(f"🔧 Calling: {tool['name']}")
                print(f"Args: {tool['args']}")
                results = await named_tools[tool["name"]].ainvoke(tool["args"])
                print(f"✅ Output: {results}\n")
                messages.append(ToolMessage(content=str(results), tool_name=tool["name"], tool_call_id=tool["id"]))

    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
