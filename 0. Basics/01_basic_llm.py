
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

print("=== CPU-COMPATIBLE LANGUAGE MODELS ===")
print("DeepSeek-R1 requires GPU. Here are CPU alternatives that work well:")
print()

# Method 1: GPT-Neo (Better than GPT-2, works on CPU)
print("=== METHOD 1: GPT-NEO 1.3B (RECOMMENDED) ===")
try:
    generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')
    
    prompt = "Tell me about Rishabh Pant"
    result = generator(
        prompt, 
        max_length=120,
        temperature=0.7,
        do_sample=True,
        repetition_penalty=1.1,
        no_repeat_ngram_size=2
    )
    
    print("GPT-Neo Result:")
    print(result[0]['generated_text'])
    print()
    
except Exception as e:
    print(f"GPT-Neo failed: {e}")
    print()
