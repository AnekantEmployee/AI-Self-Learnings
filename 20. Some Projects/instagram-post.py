import os
from typing import TypedDict, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from huggingface_hub import InferenceClient
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

class CaptionGenerate(BaseModel):
    caption: str = Field(..., description="Hinglish Instagram caption (30-40 words) with tech references that Indian developers will find hilarious")
    image_prompt: str = Field(..., description="Detailed image prompt for informative yet funny tech visual content")
    
class FeedbackEvaluation(BaseModel):
    humor_score: int = Field(..., description="Humor level from 1-5 (5 being hilarious)")
    relatability_score: int = Field(..., description="How relatable for Indian techies (1-5)")
    informativeness_score: int = Field(..., description="How informative the content is (1-5)")
    overall_feedback: str = Field(..., description="Detailed feedback on what to improve")
    should_iterate: bool = Field(..., description="Whether to iterate again (true if any score < 4)")

model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
caption_model = model.with_structured_output(CaptionGenerate)
feedback_model = model.with_structured_output(FeedbackEvaluation)

# Enhanced system message focusing on tech concepts + humor
system_message = SystemMessage(content="""Tu ek experienced Indian tech lead hai jo complex coding concepts ko funny + informative way mein explain karta hai. Tera style:

HUMOR APPROACH:
- Indian tech company culture ke real scenarios use kar (standups, code reviews, deployments)
- Relatable coding fails aur debugging nightmares
- Hinglish naturally mix kar (not forced)
- Self-deprecating humor about programming struggles
- Indian context ke references (chai breaks, traffic delays affecting standups, etc.)

EDUCATIONAL APPROACH:
- Technical concepts ko simple analogies se explain kar
- Common mistakes aur their solutions mention kar
- Practical examples de jo developers daily face karte hain
- Code snippets ya concepts visually represent kar

LANGUAGE STYLE:
- "Bhai", "Yaar", "Arrey", "Bas kar" naturally use kar
- Hindi + English smooth blend
- Modern slang + traditional phrases
- Indian IT office references

Goal: Make complex tech concepts easy to understand while keeping it funny and relatable for Indian developers.""")

# Template for caption generation
caption_template = PromptTemplate(
    template="""Topic: {topic}

Previous attempt (if any): {previous_caption}
Previous feedback (if any): {previous_feedback}

Create a 30-40 word Hinglish Instagram caption that:

HUMOR REQUIREMENTS:
- Uses real coding scenarios Indian developers face daily
- Includes relatable debugging/deployment disasters
- Has natural Hinglish flow (not forced translations)
- References Indian IT culture (office politics, manager expectations, etc.)
- Uses expressions like "Bhai kya kar diya", "Yaar ye kya hai", "Arrey bas kar"

EDUCATIONAL REQUIREMENTS:
- Explains the technical concept in simple terms
- Mentions practical use cases or common mistakes
- Gives actionable insight about the topic
- Makes the concept memorable through humor

IMAGE PROMPT REQUIREMENTS (separate from caption):
Create a detailed VISUAL-ONLY prompt for a funny yet informative illustration:

VISUAL STORYTELLING (NO TEXT, SPEECH BUBBLES, OR WORDS):
- Show the concept through visual metaphors and scenarios
- Use facial expressions, body language, and visual comedy
- Create scenes that tell the story without words
- Include visual symbols and icons that represent the concept
- Use gestures, pointing, and expressions instead of text

FUNNY VISUAL ELEMENTS:
- Confused/frustrated developer expressions (facepalm, head scratching, shocked face)
- Chaotic office scenes (multiple monitors, coffee cups, messy desk)
- Visual metaphors (like hoisting = things floating up, async = racing cars)
- Exaggerated reactions and comic situations
- Indian office context (typical IT office setup, Indian developer appearance)

INFORMATIVE VISUAL ELEMENTS:
- Use visual diagrams and flowcharts (arrows, boxes, connections)
- Show before/after scenarios visually
- Use color coding and visual hierarchy
- Include recognizable tech symbols (browsers, databases, servers)
- Show step-by-step processes through visual sequences

EXAMPLES:
Topic: "JavaScript Hoisting"
Caption: "Hoisting ke chakkar mein aaj bhi confused hun! 😅 Variable declaration upar chala jata hai bhai, jaise office mein promotion - pata nahi kab kya upar aa jaye! #JSStruggles #DevLife"

Image: "Indian developer looking confused at floating variable declarations above his head like balloons, computer screen showing code execution flow with arrows pointing upward, messy office desk with multiple coffee cups, developer's shocked facial expression"

Topic: "Async/Await"
Image: "Split scene showing synchronous vs asynchronous: Left side - Indian developer waiting impatiently tapping fingers while single task loads, Right side - same developer juggling multiple floating task bubbles with relaxed expression, visual arrows showing parallel execution"

Focus on VISUAL STORYTELLING that's immediately funny and educational!""",
    input_variables=['topic', 'previous_caption', 'previous_feedback']
)

# Template for feedback evaluation
feedback_template = PromptTemplate(
    template="""Evaluate this Hinglish tech meme content:

Topic: {topic}
Caption: {caption}
Image Prompt: {image_prompt}

Rate on scale 1-5 and provide detailed feedback:

HUMOR (1-5): Is it genuinely funny for Indian developers?
- Does it use relatable coding scenarios?
- Is the Hinglish natural and not forced?
- Will Indian techies actually laugh or just smile?
- Are the references authentic to Indian IT culture?

RELATABILITY (1-5): How well will Indian developers connect?
- Uses familiar office/coding situations?
- Includes real programming pain points?
- References Indian context properly?
- Speaks to daily developer struggles?

INFORMATIVENESS (1-5): How educational is it?
- Explains the technical concept clearly?
- Provides practical insights?
- Makes learning memorable?
- Gives actionable information?

OVERALL FEEDBACK:
- What specific improvements needed?
- Which elements work well?
- How to make it more viral?
- Technical accuracy issues?

Should iterate if ANY score is below 4/5.""",
    input_variables=['topic', 'caption', 'image_prompt']
)

class InstagramPost(TypedDict):
    topic: str
    caption: str
    image_prompt: str
    iteration_count: int
    humor_score: int
    relatability_score: int
    informativeness_score: int
    feedback: str
    should_continue: bool
    
def generate_caption(state: InstagramPost) -> InstagramPost:
    try:
        previous_caption = state.get('caption', '')
        previous_feedback = state.get('feedback', '')
        
        messages = [
            system_message,
            HumanMessage(content=caption_template.format(
                topic=state['topic'],
                previous_caption=previous_caption,
                previous_feedback=previous_feedback
            ))
        ]
        
        result = caption_model.invoke(messages)
        state['caption'] = result.caption
        state['image_prompt'] = result.image_prompt
        
        print(f"🔄 Iteration {state['iteration_count']}")
        print(f"📝 Generated caption: {result.caption}")
        print(f"🎨 Generated image prompt: {result.image_prompt}")
        
    except Exception as e:
        print(f"❌ Error generating caption: {str(e)}")
        state['caption'] = f"Tech concept: {state['topic']} - Samjhana mushkil, implement karna aur bhi mushkil! 😅 #TechStruggles #DevLife"
        state['image_prompt'] = f"Diagram explaining {state['topic']} with confused developer, code snippets, Indian office background"
    
    return state

def evaluate_content(state: InstagramPost) -> InstagramPost:
    try:
        messages = [
            SystemMessage(content="Tu ek expert content evaluator hai jo Indian tech memes rate karta hai. Be honest and constructive."),
            HumanMessage(content=feedback_template.format(
                topic=state['topic'],
                caption=state['caption'],
                image_prompt=state['image_prompt']
            ))
        ]
        
        result = feedback_model.invoke(messages)
        state['humor_score'] = result.humor_score
        state['relatability_score'] = result.relatability_score
        state['informativeness_score'] = result.informativeness_score
        state['feedback'] = result.overall_feedback
        state['should_continue'] = result.should_iterate and state['iteration_count'] < 3
        
        print(f"📊 Scores - Humor: {result.humor_score}/5, Relatability: {result.relatability_score}/5, Info: {result.informativeness_score}/5")
        print(f"💭 Feedback: {result.overall_feedback}")
        print(f"🔄 Continue iterating: {state['should_continue']}")
        
    except Exception as e:
        print(f"❌ Error evaluating content: {str(e)}")
        # Default to continue if evaluation fails
        state['should_continue'] = state['iteration_count'] < 2
        state['feedback'] = "Evaluation failed, continuing iteration"
    
    return state

def check_iteration_condition(state: InstagramPost) -> Literal["continue_iteration", "generate_image"]:
    if state['should_continue']:
        return "continue_iteration"
    return "generate_image"

def increment_counter(state: InstagramPost) -> InstagramPost:
    state['iteration_count'] += 1
    print(f"🔄 Starting iteration {state['iteration_count']}")
    return state

def generate_image(state: InstagramPost) -> InstagramPost:
    topic = state['topic']
    prompt = state['image_prompt']
    filename = topic.replace(" ", "-").lower() + f"-final.png"
    
    os.makedirs("ig-post", exist_ok=True)
    
    try:
        # Clean the prompt to remove any text references
        cleaned_prompt = prompt.replace("Add speech bubbles:", "Show through expressions:")
        cleaned_prompt = cleaned_prompt.replace("speech bubbles", "facial expressions")
        cleaned_prompt = cleaned_prompt.replace("'Arrey yaar, adjust karo!'", "frustrated expressions")
        cleaned_prompt = cleaned_prompt.replace("'Sab theek hai na?'", "nervous expressions")
        
        # Enhanced prompt for visual storytelling without text
        enhanced_prompt = f"""Visual comedy illustration: {cleaned_prompt}. 
        STYLE: Cartoon/comic style, bright vibrant colors, exaggerated expressions, clean vector-like art
        ABSOLUTELY NO TEXT, WORDS, OR LETTERS: Pure visual storytelling through expressions, symbols, and scenarios only
        ELEMENTS: Indian developer character with expressive face, modern office environment, visual metaphors, 
        funny facial expressions (confused, shocked, happy, nervous), technical symbols and icons, 
        flowchart-style arrows and connections, color-coded elements, exaggerated comedic situations, 
        before/after visual comparisons, gesture-based communication"""
        
        print(f"🎨 Generating image with cleaned prompt...")
        
        client = InferenceClient(model="black-forest-labs/FLUX.1-schnell", token="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")
        image = client.text_to_image(enhanced_prompt)
        
        full_path = os.path.join("ig-post", filename)
        image.save(full_path)
        print(f"✅ Final image saved: {full_path}")
        
    except Exception as e:
        print(f"❌ Error generating image: {str(e)}")
        print(f"❌ Full error details: {type(e).__name__}: {str(e)}")
        
        # Try a simpler fallback prompt
        try:
            print("🔄 Trying fallback image generation...")
            fallback_prompt = f"Cartoon illustration of Indian software developer working on {topic}, funny expression, colorful, no text"
            image = client.text_to_image(fallback_prompt)
            full_path = os.path.join("ig-post", filename)
            image.save(full_path)
            print(f"✅ Fallback image saved: {full_path}")
        except Exception as fallback_error:
            print(f"❌ Fallback also failed: {str(fallback_error)}")
    
    return state

# Create the graph with iteration loop
graph = StateGraph(InstagramPost)

# Add nodes
graph.add_node('generate_caption', generate_caption)
graph.add_node('evaluate_content', evaluate_content)
graph.add_node('increment_counter', increment_counter)
graph.add_node('generate_image', generate_image)

# Add edges
graph.add_edge(START, 'generate_caption')
graph.add_edge('generate_caption', 'evaluate_content')
graph.add_conditional_edges(
    'evaluate_content',
    check_iteration_condition,
    {
        'continue_iteration': 'increment_counter',
        'generate_image': 'generate_image'
    }
)
graph.add_edge('increment_counter', 'generate_caption')
graph.add_edge('generate_image', END)

workflow = graph.compile()

# Run the workflow
if __name__ == "__main__":
    # Example topics - mix of technical concepts with Indian context
    test_topics = [
        'JavaScript Hoisting',
        'React useEffect Dependency Array', 
        'CSS Flexbox vs Grid',
        'Async Await Error Handling',
        'Database Indexing Performance',
        'Git Merge vs Rebase',
        'Python List Comprehensions',
        'API Rate Limiting',
        'Docker Container Optimization',
        'Redis Caching Strategies',
        'Callback Hell',
        'Memory Leaks in JavaScript',
        'CSS Specificity Wars',
        'Microservices vs Monolith',
        'Promise.all vs Promise.allSettled'
    ]
    
    topic = 'python loop'  # Change this to test different topics
    
    initial_state = {
        'topic': topic,
        'caption': '',
        'image_prompt': '',
        'iteration_count': 1,
        'humor_score': 0,
        'relatability_score': 0,
        'informativeness_score': 0,
        'feedback': '',
        'should_continue': True
    }
    
    print(f"🚀 Starting meme generation for: {topic}")
    print("=" * 60)
    
    result = workflow.invoke(initial_state)
    
    print("\n" + "=" * 60)
    print("🎉 FINAL HINGLISH TECH MEME:")
    print(f"🎯 Topic: {result['topic']}")
    print(f"😂 Caption: {result['caption']}")
    print(f"🎨 Image Prompt: {result['image_prompt']}")
    print(f"📊 Final Scores: H:{result.get('humor_score', 0)}/5, R:{result.get('relatability_score', 0)}/5, I:{result.get('informativeness_score', 0)}/5")
    print(f"🔄 Total Iterations: {result['iteration_count']}")
    print("\n🔥 Ye pakka viral hoga! Share kar bhai! 🚀📈")