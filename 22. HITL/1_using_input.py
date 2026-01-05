from typing import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.messages import HumanMessage

load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages]


model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


def generate_post(state: State) -> State:
    return {"messages": [model.invoke(state["messages"])]}


def get_review_decision(state: State):
    post_content = state["messages"][-1].content

    print("\n📢 Current LinkedIn Post:\n")
    print(post_content)
    print("\n")

    decision = input("Post to LinkedIn? (yes/no): ")

    if decision.lower() == "yes":
        return "post"
    else:
        return "collect_feedback"


def post(state: State):
    final_post = state["messages"][-1].content
    print("\n📢 Final LinkedIn Post:\n")
    print(final_post)
    print("\n✅ Post has been approved and is now live on LinkedIn!")


def collect_feedback(state: State):
    feedback = input("How can I improve this post?")
    return {"messages": [HumanMessage(content=feedback + "in 40 words")]}


graph = StateGraph(State)

graph.add_node("generate_post", generate_post)
graph.add_node("get_review_decision", get_review_decision)
graph.add_node("collect_feedback", collect_feedback)
graph.add_node("post", post)

graph.add_edge(START, "generate_post")
graph.add_conditional_edges("generate_post", get_review_decision)
graph.add_edge("collect_feedback", "generate_post")
graph.add_edge("post", END)

app = graph.compile()

response = app.invoke(
    {
        "messages": [
            HumanMessage(
                content="Write me a LinkedIn post on AI Agents taking over content creation in 40 words"
            )
        ]
    }
)

print(response)
