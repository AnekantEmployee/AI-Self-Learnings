from langgraph.graph import StateGraph, START, END
# from IPython.display import Image
from typing import TypedDict

# Created a State
class BMIState(TypedDict):
    weight: float
    height: float
    bmi: float
    category: str

# Created a Graph    
graph = StateGraph(BMIState)

# Required Functions
def calculate_bmi(state: BMIState) -> BMIState:
    weight = state['weight']
    height = state['height']
    
    bmi = weight / (height ** 2)
    
    state['bmi'] = round(bmi, 2)
    return state
 
def label_bmi(state: BMIState) -> BMIState:
    bmi = state['bmi']
    
    if bmi < 18.5:
        state["category"] = "Underweight"
    elif 18.5 <= bmi < 25:
        state["category"] = "Normal"
    elif 25 <= bmi < 30:
        state["category"] = "Overweight"
    else:
        state["category"] = "Obese"
    
    return state 
 
# Add Node
graph.add_node('calculate_bmi', calculate_bmi)
graph.add_node('label_bmi', label_bmi)

# Add Edges
graph.add_edge(START, 'calculate_bmi')
graph.add_edge('calculate_bmi', 'label_bmi')
graph.add_edge('label_bmi', END)

# Compliation
workflow = graph.compile()

# Execution
result = workflow.invoke({'weight': 53, 'height': 1.8288})
print(result)

# Image(workflow.get_graph().draw_mermaid_png())