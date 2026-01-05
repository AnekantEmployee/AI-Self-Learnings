# app.py
import os
import streamlit as st
from google import genai
from PyPDF2 import PdfReader
from pathlib import Path
from dotenv import load_dotenv

# ------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------
load_dotenv()                         # .env must contain GEMINI_API_KEY
GEMINI_MODEL = "gemini-2.5-flash"
PDF_PATH     = Path("data/Resume.pdf")

# ------------------------------------------------------------------
# 2. One-time PDF ingestion
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_pdf_text(pdf_path: Path) -> str:
    """Extract plain text from the PDF. Cached for the whole session."""
    if not pdf_path.exists():
        st.error(f"PDF not found: {pdf_path}")
        st.stop()
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

# ------------------------------------------------------------------
# 3. Gemini client
# ------------------------------------------------------------------
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def ask_gemini(question: str, pdf_text: str) -> str:
    """Return the LLM answer as a single string."""
    prompt = f"""
    You are a helpful assistant.  
    Answer the following question using ONLY the document below.  
    If the document does not contain the information, say so.

    Document:
    {pdf_text}

    Question: {question}
    """
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={"temperature": 0.0}
    )
    return response.text

# ------------------------------------------------------------------
# 4. Streamlit UI
# ------------------------------------------------------------------
st.set_page_config(page_title="📄 PDF Chatbot", layout="centered")

# --- Header
st.title("📄 Chat with your PDF")
st.caption("Ask anything about the document loaded below.")

# --- Sidebar
with st.sidebar:
    st.subheader("Loaded PDF")
    st.write(PDF_PATH.name)

# --- Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Load PDF once
pdf_text = load_pdf_text(PDF_PATH)

# --- Chat display
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- User input
if prompt := st.chat_input("Ask something about the document…"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant answer
    with st.chat_message("assistant"):
        with st.spinner("Processing…"):
            answer = ask_gemini(prompt, pdf_text)
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})