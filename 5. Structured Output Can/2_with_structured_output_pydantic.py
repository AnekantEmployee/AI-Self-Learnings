from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class Review(BaseModel):
    name: Optional[str] = Field(description='Name of the reviewer')
    key: list[str] = Field(description='Key points of the review')
    summary: str = Field(description='Summary of the review')
    sentiment: Literal['pos', 'neg', 'neu'] = Field(description='Overall sentiment of the review out of positive, negative & neutral')
    pros: Optional[list[str]] = Field(description="List of the positive points")
    cons: Optional[list[str]] = Field(description='List of the negative reviews')
    
    # name: Annotated[Optional[str], "Name of the reviewer"]
    # key: Annotated[list[str], "Key points of the review"]
    # summary: Annotated[str, "A Brief summary of the review"]
    # sentiment: Annotated[Literal['pos', 'neg', 'neu'], "Overall sentiment of the review out of positive, negative & neutral"]
    # pros: Annotated[Optional[list[str]], "List of positive points of the review"]
    # cons: Annotated[Optional[list[str]], "List of negative points of the review"]
    
    
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
structured_model = model.with_structured_output(Review)

result = structured_model.invoke(
    """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

    The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

    However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

    Pros:
    Insanely powerful processor (great for gaming and productivity)
    Stunning 200MP camera with incredible zoom capabilities
    Long battery life with fast charging
    S-Pen support is unique and useful
                                    
    Review by Nitish Singh"""
)

print(result)
for item, val in dict(result).items():
    print(item, val) 