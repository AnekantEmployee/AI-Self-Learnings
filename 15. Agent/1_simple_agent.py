from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain import agents
from langchain_classic.agents import AgentExecutor
from langsmith import Client
import warnings

# Suppress the ddgs warning
warnings.filterwarnings("ignore", message=".*duckduckgo_search.*has been renamed.*")

load_dotenv()


class SimpleSearchAgent:
    """A simple search agent that shows detailed results"""

    def __init__(self):
        self.search_tool = DuckDuckGoSearchRun()
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", temperature=0.1
        )

        # Pull the ReAct prompt
        client = Client(api_key=LANGSMITH_AP_KEY)
        self.prompt = client.pull_prompt("hwchase17/react", include_model=True)

        # Create agent
        agent = agents.create_react_agent(self.model, [self.search_tool], self.prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=[self.search_tool],
            verbose=True,
            max_iterations=4,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
        )

    def format_detailed_results(self, raw_search_data):
        """Format the raw search data into detailed readable format"""
        if not raw_search_data:
            return "No detailed search data available."

        # Split into individual results and format
        lines = raw_search_data.strip().split("…")
        formatted_results = []

        for i, line in enumerate(lines, 1):
            line = line.strip()
            if line and len(line) > 20:  # Filter out very short fragments
                # Add proper formatting
                if not line.endswith("."):
                    line += "..."
                formatted_results.append(f"📰 Result {i}: {line}")

        return "\n\n".join(formatted_results) if formatted_results else raw_search_data

    def search(self, query: str) -> str:
        """Perform search and return both detailed and summary results"""
        try:
            print(f"\n🔍 Searching for: {query}")

            # Execute search
            result = self.agent_executor.invoke({"input": query})

            # Get the final answer
            final_answer = result.get("output", "No final answer provided")

            # Extract raw search data from intermediate steps
            raw_search_data = None
            intermediate_steps = result.get("intermediate_steps", [])

            for step in intermediate_steps:
                if len(step) >= 2:
                    action, observation = step[0], step[1]
                    if hasattr(action, "tool") and action.tool == "duckduckgo_search":
                        raw_search_data = str(observation)
                        break

            # Format response
            response = f"🎯 **SUMMARY:**\n{final_answer}\n\n"

            if raw_search_data:
                detailed_results = self.format_detailed_results(raw_search_data)
                response += f"📊 **DETAILED SEARCH RESULTS:**\n{detailed_results}"
            else:
                response += "ℹ️ No detailed search data captured."

            return response

        except Exception as e:
            return f"❌ Search failed: {str(e)}"


def main():
    """Main function"""
    print("🤖 Enhanced Search Agent")
    print("=" * 50)

    try:
        agent = SimpleSearchAgent()
        print("✅ Agent ready!")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return

    while True:
        try:
            query = input("\nEnter search query (or 'quit'): ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break

            if not query:
                continue

            print("-" * 60)
            result = agent.search(query)
            print(f"\n{result}")
            print("-" * 60)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
