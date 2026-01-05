from langchain_core.tools import tool

# Step 1
def multiply(a, b):
    """Multiply two numbers""" # Very Important for understanding
    return a * b

# Step 2
def multiply(a:int, b:int) -> int: # Use Type Hinting
    """Multiply two numbers"""
    return a * b

# Step 3
@tool # Add a decorator
def multiply(a:int, b:int) -> int:
    """Multiply two numbers"""
    return a * b