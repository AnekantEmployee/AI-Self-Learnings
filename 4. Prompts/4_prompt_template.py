from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate([
    ('system', "Consider yourself as a {expert} expert"),
    ('human', 'Tell me about {topic}')
])

prompt_text = prompt.invoke({
    'expert': 'Cricket',
    'topic': 'Wide Ball'
})
print(prompt_text)