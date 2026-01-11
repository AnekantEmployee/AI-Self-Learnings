import streamlit as st
from bot_backend import get_chatbot
from langchain_core.messages import HumanMessage

# Initialize chatbot once
if "chatbot" not in st.session_state:
    st.session_state.chatbot = get_chatbot()

# Streamlit UI
st.title("🤖 AI Chatbot")
st.write("Chat with your AI assistant!")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit_session"

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            try:
                results = st.session_state.chatbot.invoke({"messages": [HumanMessage(prompt)]}, config=config)
                response = results["messages"][-1].content
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Sidebar with options
with st.sidebar:
    st.header("Options")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        # Create new thread ID and reinitialize chatbot
        st.session_state.thread_id = f"streamlit_session_{st.session_state.get('session_counter', 0)}"
        st.session_state.session_counter = st.session_state.get('session_counter', 0) + 1
        st.session_state.chatbot = get_chatbot()
        st.rerun()
    
    st.write(f"**Thread ID:** {st.session_state.thread_id}")
    st.write(f"**Messages:** {len(st.session_state.messages)}")
