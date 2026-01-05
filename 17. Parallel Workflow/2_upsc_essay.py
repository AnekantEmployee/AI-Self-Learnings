from typing import TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

essay = """India in the Age of AI
As the world enters a transformative era defined by artificial intelligence (AI), India stands at a critical juncture — one where it can either emerge as a global leader in AI innovation or risk falling behind in the technology race. The age of AI brings with it immense promise as well as unprecedented challenges, and how India navigates this landscape will shape its socio-economic and geopolitical future.

India's strengths in the AI domain are rooted in its vast pool of skilled engineers, a thriving IT industry, and a growing startup ecosystem. With over 5 million STEM graduates annually and a burgeoning base of AI researchers, India possesses the intellectual capital required to build cutting-edge AI systems. Institutions like IITs, IIITs, and IISc have begun fostering AI research, while private players such as TCS, Infosys, and Wipro are integrating AI into their global services. In 2020, the government launched the National AI Strategy (AI for All) with a focus on inclusive growth, aiming to leverage AI in healthcare, agriculture, education, and smart mobility.

One of the most promising applications of AI in India lies in agriculture, where predictive analytics can guide farmers on optimal sowing times, weather forecasts, and pest control. In healthcare, AI-powered diagnostics can help address India's doctor-patient ratio crisis, particularly in rural areas. Educational platforms are increasingly using AI to personalize learning paths, while smart governance tools are helping improve public service delivery and fraud detection.

However, the path to AI-led growth is riddled with challenges. Chief among them is the digital divide. While metropolitan cities may embrace AI-driven solutions, rural India continues to struggle with basic internet access and digital literacy. The risk of job displacement due to automation also looms large, especially for low-skilled workers. Without effective skilling and re-skilling programs, AI could exacerbate existing socio-economic inequalities.

Another pressing concern is data privacy and ethics. As AI systems rely heavily on vast datasets, ensuring that personal data is used transparently and responsibly becomes vital. India is still shaping its data protection laws, and in the absence of a strong regulatory framework, AI systems may risk misuse or bias.

To harness AI responsibly, India must adopt a multi-stakeholder approach involving the government, academia, industry, and civil society. Policies should promote open datasets, encourage responsible innovation, and ensure ethical AI practices. There is also a need for international collaboration, particularly with countries leading in AI research, to gain strategic advantage and ensure interoperability in global systems.

India's demographic dividend, when paired with responsible AI adoption, can unlock massive economic growth, improve governance, and uplift marginalized communities. But this vision will only materialize if AI is seen not merely as a tool for automation, but as an enabler of human-centered development.

In conclusion, India in the age of AI is a story in the making — one of opportunity, responsibility, and transformation. The decisions we make today will not just determine India's AI trajectory, but also its future as an inclusive, equitable, and innovation-driven society."""


class FeedbackModel(BaseModel):
    text: str = Field(description="Text summary of the feedback")
    score: int = Field(description="Score of the feedback, 1-5")


class UPSCEssay(TypedDict):
    essay: str
    depth_of_topic: int
    depth_of_topic_text: str
    language: int
    language_text: str
    clarity_of_thought: int
    clarity_of_thought_text: str
    summary: str


model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
parser = PydanticOutputParser(pydantic_object=FeedbackModel)
str_parser = StrOutputParser()
pointer_template = PromptTemplate(
    template="Evaluate this essay on the basis of {point}: {essay} \n {format_instructions}",
    input_variables=["essay", "point"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
summary_template = PromptTemplate(
    template="Summarize the feedback for this essay based on the following criteria:\nLanguage: {language_text}\nClarity of Thought: {clarity_of_thought_text}\nDepth of Topic: {depth_of_topic_text}",
    input_variables=["language_text", "clarity_of_thought_text", "depth_of_topic_text"],
)
chain = pointer_template | model | parser
summary_chain = summary_template | model | str_parser

graph = StateGraph(UPSCEssay)

def language(state: UPSCEssay) -> UPSCEssay:
    result = chain.invoke({"essay": state["essay"], "point": "Language"})
    return {
        "language": result.score,
        "language_text": result.text,
    }


def clarity_of_thought(state: UPSCEssay) -> UPSCEssay:
    result = chain.invoke({"essay": state["essay"], "point": "Clarity of Thought"})
    return {
        "clarity_of_thought": result.score,
        "clarity_of_thought_text": result.text,
    }


def depth_of_topic(state: UPSCEssay) -> UPSCEssay:
    result = chain.invoke({"essay": state["essay"], "point": "Depth of Topic"})
    return {
        "depth_of_topic": result.score,
        "depth_of_topic_text": result.text,
    }


def summary(state: UPSCEssay) -> UPSCEssay:
    result = summary_chain.invoke({
        "language_text": state["language_text"],
        "clarity_of_thought_text": state["clarity_of_thought_text"],
        "depth_of_topic_text": state["depth_of_topic_text"],
    })
    return {"summary": result}


graph.add_node("language", language)
graph.add_node("depth_of_topic", depth_of_topic)
graph.add_node("clarity_of_thought", clarity_of_thought)
graph.add_node("summary", summary)

# Set up edges
graph.add_edge(START, "language")
graph.add_edge(START, "depth_of_topic")
graph.add_edge(START, "clarity_of_thought")

graph.add_edge("language", "summary")
graph.add_edge("depth_of_topic", "summary")
graph.add_edge("clarity_of_thought", "summary")

graph.add_edge("summary", END)


workflow = graph.compile()
result = workflow.invoke({"essay": essay})
print(result)