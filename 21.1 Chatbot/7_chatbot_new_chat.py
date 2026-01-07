import uuid
import streamlit as st
from bot_backend import get_chatbot
from langchain_core.messages import HumanMessage

chatbot = get_chatbot()

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "threads_list" not in st.session_state:
    st.session_state["threads_list"] = []


if "current_thread" not in st.session_state:
    st.session_state["current_thread"] = ""


def get_unique_id():
    my_uuid = uuid.uuid4()
    return my_uuid


def switch_chat(id):
    st.session_state["current_thread"] = id
    is_new_thread = id not in st.session_state["threads_list"]

    if is_new_thread:
        st.session_state["threads_list"].append(id)
        st.session_state["message_history"] = []  # Clear for new thread
    else:
        # Load existing thread state
        state = chatbot.get_state(config={"configurable": {"thread_id": id}})
        st.session_state["message_history"] = state.values.get("messages", [])


if len(st.session_state["threads_list"]) == 0:
    id = get_unique_id()
    switch_chat(id)

CONFIG = {"configurable": {"thread_id": st.session_state["current_thread"]}}

# loading the conversation history
for message in st.session_state["message_history"]:
    if hasattr(message, "type"):  # LangChain message object
        role = "user" if message.type == "human" else "assistant"
        content = message.content
    else:  # Dictionary format
        role = message["role"]
        content = message["content"]

    with st.chat_message(role):
        st.text(content)


user_input = st.chat_input("Type here")


if user_input:

    # first add the message to message_history
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    # first add the message to message_history
    with st.chat_message("assistant"):

        ai_message = st.write_stream(
            message_chunk[0].content
            for message_chunk in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            )
        )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

with st.sidebar:
    st.title(f"Current Chat: {st.session_state['current_thread']}")

    st.title("Chatbot")

    new_chat = st.button("New Chat")

    if new_chat:
        id = get_unique_id()
        switch_chat(id)

    st.header("My Conversations")

    if len(st.session_state["threads_list"]) != 0:
        for id in st.session_state["threads_list"]:
            id_btn = st.button(str(id), key=f"streamlit_btn_{id}")
            if id_btn:
                switch_chat(id)
