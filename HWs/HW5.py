import sys, os, streamlit as st
from typing import List, Dict

__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb
from chromadb.utils import embedding_functions

# Page setup
st.title("HW 5 — Intelligent iSchool Orgs Chatbot")

# Constants
CHROMA_DIR = ".ChromaDB_hw4"       # reusing HW4 database
COLLECTION_NAME = "Lab4Collection"
EMBED_MODEL = "text-embedding-3-small"
TOP_K = 4
MEM_KEEP = 5

# API Key
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Session State for memory
if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, str]] = []

# Load ChromaDB collection
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY, model_name=EMBED_MODEL
)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME, embedding_function=openai_ef
)

# Vector search function
def retrieve_relevant_info(query: str) -> str:
    """Return relevant org info using vector search (TOP_K chunks)."""
    res = collection.query(query_texts=[query], n_results=TOP_K)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    out = []
    for d, md in zip(docs, metas):
        out.append(f"[{md.get('filename','?')} | chunk {md.get('chunk_id',0)}]\n{d}")
    return "\n\n".join(out)

# LLM call
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

def answer_with_context(query: str) -> str:
    """Run vector search, then call LLM with results + memory."""
    context = retrieve_relevant_info(query)
    system = {"role": "system", "content": "You answer questions about Syracuse iSchool student organizations. Use ONLY the provided sources; if unsure, say you don't know."}
    msgs = [system] + st.session_state.history[-(MEM_KEEP*2):]
    if context:
        msgs.append({"role": "user", "content": f"SOURCES:\n{context}"})
    msgs.append({"role": "user", "content": query})

    resp = client.chat.completions.create(
        model="gpt-4o-mini", messages=msgs, temperature=0.2
    )
    return resp.choices[0].message.content

# Chat UI
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_q = st.chat_input("Ask about iSchool student organizations…")
if user_q:
    st.session_state.history.append({"role": "user", "content": user_q})
    with st.chat_message("assistant"):
        ans = answer_with_context(user_q)
        st.markdown(ans)
    st.session_state.history.append({"role": "assistant", "content": ans})
    if len(st.session_state.history) > MEM_KEEP*2:
        st.session_state.history = st.session_state.history[-(MEM_KEEP*2):]
