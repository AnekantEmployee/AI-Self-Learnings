import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

st.header('Simple Receipe Maker')
prompt_val = st.text_input("Enter your prompt", value="Create a recipe for making tea")
button = st.button("Get Receipe")

if button and prompt_val:
    llm = HuggingFaceEndpoint(repo_id="moonshotai/Kimi-K2-Instruct", task='text-generation')
    model = ChatHuggingFace(llm=llm)
    result = model.invoke(prompt_val)
    st.write(result.content)
