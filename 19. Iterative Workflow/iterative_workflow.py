import operator
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import TypedDict, Literal, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

class CaptionState(TypedDict):
    topic: str
    caption: str
    evaluation: Literal["approved", "needs_improvement"]
    feedback: str
    iteration: int
    max_iteration: int
    caption_history: Annotated[list[str], operator.add]
    feedback_history: Annotated[list[str], operator.add]

class CaptionEvaluation(BaseModel):
    evaluation: Literal["approved", "needs_improvement"] = Field(..., description="Final evaluation result.")
    feedback: str = Field(..., description="feedback for the Instagram caption.")
    
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
optimizer_model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
generator_model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
structured_evaluator_llm = model.with_structured_output(CaptionEvaluation)

def generate_caption(state: CaptionState):
    # prompt
    messages = [
        SystemMessage(content="""You are a witty and engaging Instagram influencer who creates viral-worthy content. 
        You understand Instagram culture, trends, and what makes people stop scrolling to engage with a post."""),
        HumanMessage(content=f"""
Create a funny and engaging Instagram caption about: "{state['topic']}"

Instagram Caption Rules:
- Write in a conversational, relatable tone (mix of English and Hinglish is perfect)
- Use observational humor, relatable situations, or cultural references
- Think about what would make someone double-tap, comment, or share
- Can be 1-3 sentences, doesn't need to be super short like Twitter
- Include relevant emojis naturally (but don't overdo it)
- Make it scroll-stopping and engaging
- Focus on relatability and humor that resonates with Indian millennials/Gen-Z
- Avoid question-answer format or generic setup-punchline jokes

Think like you're talking to your friends about this topic in a fun, witty way!
""")
    ]

    # send generator_llm
    response = generator_model.invoke(messages).content

    # return response
    return {'caption': response, 'caption_history': [response]}

def evaluate_caption(state: CaptionState):
    # prompt
    messages = [
        SystemMessage(content="""You are an Instagram content strategist who knows what makes captions go viral. 
        You evaluate captions based on engagement potential, relatability, humor, and Instagram best practices."""),
        HumanMessage(content=f"""
Evaluate this Instagram caption:

Caption: "{state['caption']}"
Topic: "{state['topic']}"

Evaluation Criteria:
1. **Engagement Potential** – Would people comment, share, or save this?
2. **Relatability** – Does it connect with the target audience's experiences?
3. **Humor Quality** – Is it genuinely funny, witty, or clever?
4. **Instagram Fit** – Does it feel native to Instagram culture and format?
5. **Scroll-Stopping Power** – Would this make someone pause while scrolling?
6. **Authenticity** – Does it sound natural and conversational?

Auto-reject if:
- Uses boring question-answer format ("Why do we..." "What happens when...")
- Sounds too formal or corporate
- Lacks personality or humor
- Feels generic or overused
- Doesn't relate to the topic meaningfully
- Too lengthy or wordy for social media attention spans

Rate the caption and provide specific feedback on how to improve engagement and humor.
""")
    ]

    response = structured_evaluator_llm.invoke(messages)

    return {'evaluation': response.evaluation, 'feedback': response.feedback, 'feedback_history': [response.feedback]}

def optimize_caption(state: CaptionState):
    messages = [
        SystemMessage(content="""You are a social media content optimizer who transforms good captions into viral-worthy ones. 
        You understand Instagram algorithms, user behavior, and what drives engagement."""),
        HumanMessage(content=f"""
Improve this Instagram caption based on the feedback:

Topic: "{state['topic']}"
Current Caption: "{state['caption']}"
Feedback: "{state['feedback']}"

Create a better version that:
- Addresses the specific feedback points
- Increases engagement potential
- Maintains authenticity and humor
- Feels more relatable and shareable
- Uses appropriate emojis and tone for Instagram
- Connects better with Indian social media users

Make it punchy, relatable, and scroll-stopping!
""")
    ]

    response = optimizer_model.invoke(messages).content
    iteration = state['iteration'] + 1

    return {'caption': response, 'iteration': iteration, 'caption_history': [response]}

def route_evaluation(state: CaptionState):
    if state['evaluation'] == 'approved' or state['iteration'] >= state['max_iteration']:
        return 'approved'
    else:
        return 'needs_improvement'
    
# Build the workflow graph
graph = StateGraph(CaptionState)

graph.add_node('generate', generate_caption)
graph.add_node('evaluate', evaluate_caption)
graph.add_node('optimize', optimize_caption)

graph.add_edge(START, 'generate')
graph.add_edge('generate', 'evaluate')
graph.add_conditional_edges('evaluate', route_evaluation, {'approved': END, 'needs_improvement': 'optimize'})
graph.add_edge('optimize', 'evaluate')

workflow = graph.compile()

# Example usage
def generate_instagram_caption(topic, max_iterations=2):
    """Generate an Instagram caption for a given topic"""
    initial_state = {
        "topic": topic,
        "iteration": 0,
        "max_iteration": max_iterations
    }
    
    result = workflow.invoke(initial_state)
    
    print(f"🎯 Topic: {topic}")
    print(f"📝 Final Caption: {result['caption']}")
    print(f"🔄 Iterations: {result['iteration']}")
    print(f"✅ Status: {result['evaluation']}")
    
    if result.get('feedback'):
        print(f"💬 Final Feedback: {result['feedback']}")
    
    # Show the evolution of captions
    if len(result['caption_history']) > 1:
        print("\n📈 Caption Evolution:")
        for i, caption in enumerate(result['caption_history'], 1):
            print(f"{i}. {caption}")
    
    return result

# Test with different topics
if __name__ == "__main__":
    # Test various topics
    topics = [
        "AWS Ec2 Instance",
        "Amazon S3 Bucket",
    ]
    
    for topic in topics[:2]:  # Test first 2 topics
        print("="*60)
        generate_instagram_caption(topic, max_iterations=3)
        print("\n")