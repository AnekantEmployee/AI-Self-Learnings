from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="First number of multiply")
    b: int = Field(required=True, description="Second number of multiply")

def multiply(a:int, b:int) -> int:
    """Multiply two numbers"""
    return a * b

multiply_tool = StructuredTool.from_function(
    func=multiply,
    name="Multiply",
    description='Multiply two numbers',
    args_schema=MultiplyInput
)

result = multiply_tool.invoke({'a': 3, 'b': 5})


print(result)
print(multiply_tool.name) # Name
print(multiply_tool.description) # Description of the functon
print(multiply_tool.args) # Arguments
print(multiply_tool.args_schema.model_json_schema()) # Schema