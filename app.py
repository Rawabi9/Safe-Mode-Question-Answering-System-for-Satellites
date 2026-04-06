import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import base64

st.set_page_config(page_title="Safe Mode Assistant", layout="wide")

def set_style():
    with open("kacst_logo.png", "rb") as f:
        img = f.read()
    b64 = base64.b64encode(img).decode()

    st.markdown(f"""
    <style>
    .stApp {{
        background:
        linear-gradient(rgba(255,255,255,0.92), rgba(255,255,255,0.7)),
        url("data:image/png;base64,{b64}");
        background-size: cover;
        background-position: center;
    }}

    .block-container {{
        max-width: 800px;
        margin: auto;
        text-align: center;
    }}

    .stApp {{
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }}

    h1, h2, h3, h4, h5, h6, p, label, span {{
        color: black !important;
    }}

    .stTextInput input {{
        color: black !important;
        background-color: white !important;
    }}

    .stButton > button {{
        background-color: white !important;
        color: black !important;
        border: 1px solid #ccc !important;
        border-radius: 8px !important;
    }}

    .stButton > button:hover {{
        background-color: #f3f4f6 !important;
    }}

    .stButton > button:active {{
        background-color: #9ca3af !important;
        color: white !important;
    }}

    div.stButton {{
        display: flex;
        justify-content: center;
    }}
    </style>
    """, unsafe_allow_html=True)

set_style()

@st.cache_resource
def load_system():
    reader = PdfReader("nasaSafeMode.pdf")

    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t

    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return chunks, model, index

chunks, model, index = load_system()

if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    st.markdown("## Ask question about satellite safe mode")

    query = st.text_input("", placeholder="Type your question")

    if st.button("Ask"):
        if len(query.strip()) < 5:
            st.session_state.error = "Please enter a valid question."
            st.session_state.page = "error"
            st.rerun()

        elif "safe mode" not in query.lower():
            st.session_state.error = "Only safe mode questions are allowed."
            st.session_state.page = "error"
            st.rerun()

        else:
            emb = model.encode([query]).astype("float32")
            d, i = index.search(emb, 3)

            context = "\n".join([chunks[x] for x in i[0]])

            prompt = f"""
Answer ONLY from this text:

{context}

Question: {query}
Answer:
"""

            res = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen2.5:0.5b",
                    "prompt": prompt,
                    "stream": False
                }
            )

            answer = res.json().get("response", "No answer")

            st.session_state.answer = answer
            st.session_state.page = "answer"
            st.rerun()

elif st.session_state.page == "answer":
    st.markdown("""
    <div style='text-align: center; margin-top: 120px;'>
        <h1>Answer</h1>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='text-align: center;'>
        <p style='font-size:20px; color:black;'>
            {st.session_state.answer}
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Back"):
        st.session_state.page = "home"
        st.rerun()

elif st.session_state.page == "error":
    st.markdown("""
    <div style='text-align: center; margin-top: 120px;'>
        <h1 style='color: red;'>Error</h1></div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='text-align: center;'>
        <p style='font-size:20px; color:black;'>
            {st.session_state.error}
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Back"):
        st.session_state.page = "home"
        st.rerun()