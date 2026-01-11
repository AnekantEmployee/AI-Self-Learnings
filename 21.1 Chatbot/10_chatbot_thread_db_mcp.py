import uuid
import sqlite3
import streamlit as st
from bot_backend_with_db_tools_mcp import get_chatbot, get_threads
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

chatbot = get_chatbot()

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "threads_list" not in st.session_state:
    st.session_state["threads_list"] = get_threads()

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


def delete_chat(id):
    if id in st.session_state["threads_list"]:
        st.session_state["threads_list"].remove(id)
        # Delete from database
        conn = sqlite3.connect("chatbot_tools.db", check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (str(id),))
        conn.commit()
        conn.close()
        # If deleting current thread, switch to another or create new
        if st.session_state["current_thread"] == id:
            if st.session_state["threads_list"]:
                switch_chat(st.session_state["threads_list"][-1])
            else:
                new_id = get_unique_id()
                switch_chat(new_id)


if len(st.session_state["threads_list"]) == 0:
    id = get_unique_id()
    switch_chat(id)

# CONFIG = {"configurable": {"thread_id": st.session_state["current_thread"]}}
CONFIG = {
    "configurable": {"thread_id": st.session_state["current_thread"]},
    "metadata": {
        "thread_id": st.session_state["current_thread"]
    },  # To Segregate runs by thread in langsmith
    "run_name": "chat_turn",  # To give a name to each run in langsmith
}

if len(st.session_state["message_history"]) == 0:
    st.markdown(
        """
        <div style="text-align: center; padding: 50px;">
            <h1 style="color: #4CAF50; margin-bottom: 20px;">🤖 AI Assistant</h1>
            <p style="font-size: 18px; color: #666; margin-bottom: 30px;">
                Welcome! I'm here to help you with calculations, web searches, and stock information.
            </p>
            <p style="font-size: 16px; color: #888;">
                💬 Start a conversation by typing your question below
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# loading the conversation history
for message in st.session_state["message_history"]:
    if hasattr(message, "type"):  # LangChain message object
        # Skip ToolMessage types and AI messages with tool_calls
        if message.type == "tool":
            continue
        if (
            message.type == "ai"
            and hasattr(message, "tool_calls")
            and message.tool_calls
        ):
            continue
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

        status_holder = {"box": None}

        def only_ai_message():
            for message_chunk in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk[0], ToolMessage):
                    tool_name = getattr(message_chunk[0], "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk[0], AIMessage):
                    yield message_chunk[0].content

        ai_message = st.write_stream(only_ai_message())

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
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
        st.rerun()

    st.header("My Conversations")

    if len(st.session_state["threads_list"]) != 0:
        for id in list(reversed(st.session_state["threads_list"])):
            col1, col2 = st.columns([3, 1])
            with col1:
                id_btn = st.button(str(id), key=f"streamlit_btn_{id}")
                if id_btn:
                    switch_chat(id)
                    st.rerun()
            with col2:
                del_btn = st.button("🗑️", key=f"del_btn_{id}")
                if del_btn:
                    delete_chat(id)
                    st.rerun()
