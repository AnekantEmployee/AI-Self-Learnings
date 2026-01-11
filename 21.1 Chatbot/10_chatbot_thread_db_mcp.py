import queue
import uuid
import os
import tempfile

import streamlit as st
from helpers.rag_tool import initialize_rag
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from bot_backend_with_db_tools_mcp import chatbot, get_all_threads
from helpers.async_utils import submit_async_task


# =========================== Utilities ===========================
def generate_thread_id():
    return uuid.uuid4()


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def delete_thread(thread_id):
    if thread_id in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].remove(thread_id)
        if st.session_state["thread_id"] == thread_id:
            if st.session_state["chat_threads"]:
                st.session_state["thread_id"] = st.session_state["chat_threads"][-1]
                messages = load_conversation(st.session_state["thread_id"])
                temp_messages = []
                for msg in messages:
                    if isinstance(msg, (HumanMessage, AIMessage)):
                        role = "user" if isinstance(msg, HumanMessage) else "assistant"
                        temp_messages.append({"role": role, "content": msg.content})
                st.session_state["message_history"] = temp_messages
            else:
                reset_chat()


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    # Check if messages key exists in state values, return empty list if not
    messages = state.values.get("messages", [])
    
    # Filter out system messages and tool messages
    filtered_messages = []
    for msg in messages:
        if isinstance(msg, (HumanMessage, AIMessage)):
            filtered_messages.append(msg)
    
    return filtered_messages


# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()

add_thread(st.session_state["thread_id"])

# ============================ Sidebar ============================
st.sidebar.title("LangGraph MCP Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

# PDF Upload Section
st.sidebar.header("📄 Document Upload")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    if initialize_rag(tmp_path):
        st.sidebar.success(f"✅ {uploaded_file.name} loaded successfully!")
    else:
        st.sidebar.error("❌ Failed to load PDF")

    os.unlink(tmp_path)

st.sidebar.header("My Conversations")
for thread_id in st.session_state["chat_threads"][::-1]:
    col1, col2 = st.sidebar.columns([3, 1])

    with col1:
        is_current = thread_id == st.session_state["thread_id"]
        button_label = f"{'🟢 ' if is_current else ''}{str(thread_id)}"
        if st.button(button_label, key=f"chat_{thread_id}"):
            st.session_state["thread_id"] = thread_id
            messages = load_conversation(thread_id)

            temp_messages = []
            for msg in messages:
                if isinstance(msg, (HumanMessage, AIMessage)):
                    role = "user" if isinstance(msg, HumanMessage) else "assistant"
                    temp_messages.append({"role": role, "content": msg.content})
            st.session_state["message_history"] = temp_messages
            st.rerun()

    with col2:
        if st.button("🗑️", key=f"del_{thread_id}", help="Delete chat"):
            delete_thread(thread_id)
            st.rerun()

# ============================ Main UI ============================

# Render history
for message in st.session_state["message_history"]:
    if message.get('content') and message['content'].strip():
        with st.chat_message(message["role"]):
            st.text(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    # Show user's message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    # Assistant streaming block
    with st.chat_message("assistant"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        def ai_only_stream():
            event_queue: queue.Queue = queue.Queue()

            async def run_stream():
                try:
                    async for message_chunk, metadata in chatbot.astream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=CONFIG,
                        stream_mode="messages",
                    ):
                        event_queue.put((message_chunk, metadata))
                except Exception as exc:
                    print(f"Stream error: {exc}")  # Debug logging
                    event_queue.put(("error", exc))
                finally:
                    event_queue.put(None)

            submit_async_task(run_stream())

            while True:
                item = event_queue.get()
                if item is None:
                    break
                message_chunk, metadata = item
                if message_chunk == "error":
                    print(f"Error in stream: {metadata}")  # Debug logging
                    raise metadata

                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    tool_content = getattr(message_chunk, "content", "")
                    print(f"Tool {tool_name} result: {tool_content}")  # Debug logging

                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=False
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=False,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Save assistant message
    if ai_message and ai_message.strip():
        st.session_state["message_history"].append(
            {"role": "assistant", "content": ai_message}
        )
