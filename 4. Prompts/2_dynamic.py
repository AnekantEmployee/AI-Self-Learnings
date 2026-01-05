import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

template = PromptTemplate(
    template="""
Please create a receipe of {item_name} item in simple don't include my details also write it in {required_length} word range for the easy understanding
Also,
1. Use ingredients with their corresponding values to define each item
If the data is not available give "Insufficient Info." instead of guessing
""",
input_variables=['item_name', 'required_length']
)

# You can save these prompts into json format
## template.save('prompt.json')
## using load_prompt you can use it

st.header('Simple Receipe Maker')
item_name = st.text_input("Enter your Food Item Name", value="Sandwitch")
required_length = st.selectbox("Enter your Required Length of Receipe", options=['Short', 'Long'])

prompt_val = template.invoke({
    'item_name': item_name,
    'required_length': required_length
})
button = st.button("Get Receipe")

if button:
    llm = HuggingFaceEndpoint(repo_id="moonshotai/Kimi-K2-Instruct", task='text-generation')
    model = ChatHuggingFace(llm=llm)
    result = model.invoke(prompt_val)
    st.write(result.content)
