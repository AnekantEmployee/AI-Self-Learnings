from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv
import torch

load_dotenv()

# Load model and tokenizer locally
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Check if CUDA is available
device = 0 if torch.cuda.is_available() else -1

# Create the pipeline with local model
pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device=device,
    max_new_tokens=512,
    temperature=0.5,
    do_sample=True,
    pad_token_id=50256,  # Set pad token to avoid warnings
)

# Create HuggingFacePipeline with the local pipeline
llm = HuggingFacePipeline(pipeline=pipe)

# Create ChatHuggingFace model
model = ChatHuggingFace(llm=llm)

# Test the model
try:
    result = model.invoke("Tell me about Indore")
    print(result.content)
except Exception as e:
    print(f"Error: {e}")
