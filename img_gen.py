import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import requests
from PIL import Image
from io import BytesIO

# Set your Groq API key here
os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"


# MCP-style tool configuration
class ImageGenTools:
    """MCP-inspired tool registry for image generation"""

    @staticmethod
    def enhance_prompt(prompt: str, llm) -> str:
        """Use LLM to enhance the image prompt"""
        messages = [
            SystemMessage(
                content="""You are an expert at writing image generation prompts.
            Enhance the user's prompt with artistic details, style, lighting, and composition.
            Keep it concise but descriptive. Return only the enhanced prompt. in english"""
            ),
            HumanMessage(content=f"Enhance this image prompt: {prompt}"),
        ]
        response = llm.invoke(messages)
        return response.content

    @staticmethod
    def generate_image(prompt: str, save_path: str = "generated_image.png") -> dict:
        """Generate image using Pollinations API (free)"""
        try:
            # Use Pollinations.ai free API
            url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(prompt)}"
            
            response = requests.get(url)
            
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                image.save(save_path)
                return {"status": "success", "path": save_path, "prompt": prompt}
            else:
                return {"status": "error", "message": f"HTTP {response.status_code}"}

        except Exception as e:
            return {"status": "error", "message": str(e)}


# LangGraph State
class ImageGenState(TypedDict):
    user_prompt: str
    enhanced_prompt: str
    image_path: str
    status: str
    messages: Annotated[list, "conversation history"]


# LangGraph Nodes
def enhance_prompt_node(state: ImageGenState) -> ImageGenState:
    """Node to enhance the user's prompt using LLM"""
    llm = ChatGroq(model="llama3-8b-8192", temperature=0.7)

    enhanced = ImageGenTools.enhance_prompt(state["user_prompt"], llm)

    return {
        **state,
        "enhanced_prompt": enhanced,
        "messages": state["messages"] + [f"Enhanced prompt: {enhanced}"],
        "status": "prompt_enhanced",
    }


def generate_image_node(state: ImageGenState) -> ImageGenState:
    """Node to generate the image"""
    result = ImageGenTools.generate_image(
        state["enhanced_prompt"], save_path=f"generated_{state['user_prompt'][:20]}.png"
    )

    return {
        **state,
        "image_path": result.get("path", ""),
        "status": result["status"],
        "messages": state["messages"] + [f"Image generation: {result['status']}"],
    }


def review_node(state: ImageGenState) -> ImageGenState:
    """Node to review and provide feedback"""
    llm = ChatGroq(model="llama3-8b-8192", temperature=0.7)

    messages = [
        SystemMessage(
            content="Provide brief feedback on the image generation process."
        ),
        HumanMessage(
            content=f"""
        Original prompt: {state['user_prompt']}
        Enhanced prompt: {state['enhanced_prompt']}
        Status: {state['status']}
        """
        ),
    ]

    response = llm.invoke(messages)

    return {**state, "messages": state["messages"] + [f"Review: {response.content}"]}


# Build LangGraph
def create_image_generation_graph():
    """Create the LangGraph workflow"""
    workflow = StateGraph(ImageGenState)

    # Add nodes
    workflow.add_node("enhance_prompt", enhance_prompt_node)
    workflow.add_node("generate_image", generate_image_node)
    workflow.add_node("review", review_node)

    # Add edges
    workflow.set_entry_point("enhance_prompt")
    workflow.add_edge("enhance_prompt", "generate_image")
    workflow.add_edge("generate_image", "review")
    workflow.add_edge("review", END)

    return workflow.compile()


# Main execution
def main():
    """Run the image generation system"""

    # Initialize the graph
    graph = create_image_generation_graph()

    # Initial state
    initial_state = {
        "user_prompt": "a serene sunset over mountains",
        "messages": [],
    }

    # Run the graph
    print("🎨 Starting Image Generation System...")
    print(f"📝 User Prompt: {initial_state['user_prompt']}\n")

    result = graph.invoke(initial_state)

    # Display results
    print("\n" + "=" * 60)
    print("📊 EXECUTION FLOW:")
    print("=" * 60)
    for msg in result["messages"]:
        print(f"  → {msg}")

    print(f"\n✅ Final Status: {result['status']}")
    if result["status"] == "success":
        print(f"🖼️  Image saved to: {result['image_path']}")

    return result


if __name__ == "__main__":
    # Run the main system
    result = main()
