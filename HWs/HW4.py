import sys
import streamlit as st 

import sys
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")


import os, glob
from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from bs4 import BeautifulSoup

# Page
st.title("HW 4 — iSchool Student Orgs Chatbot (RAG)")

# Constants 
HTML_DIR = "data/su_orgs_html"        
CHROMA_DIR = ".ChromaDB_hw4"        
COLLECTION_NAME = "Lab4Collection"    
EMBED_MODEL = "text-embedding-3-small"  # OpenAI embeddings (1536-dim)
TOP_K = 4                              
MEM_KEEP = 5                           

# Sidebar: choose LLM 
with st.sidebar:
    vendor = st.selectbox("Model vendor", ["OpenAI", "Mistral", "Gemini"])
    use_advanced = st.checkbox("Use advanced model", value=False)
MODEL = {
    "OpenAI":  {True: "gpt-4o",              False: "gpt-4o-mini"},
    "Mistral": {True: "mistral-large-latest",False: "mistral-small-latest"},
    "Gemini":  {True: "gemini-2.5-flash",    False: "gemini-2.5-flash-lite"},
}[vendor][use_advanced]
st.caption(f"Using **{vendor}** — `{MODEL}`")

#Secrets (keys)
OPENAI_API_KEY  = st.secrets.get("OPENAI_API_KEY")
MISTRAL_API_KEY = st.secrets.get("MISTRAL_API_KEY")
GEMINI_API_KEY  = st.secrets.get("GEMINI_API_KEY")

# Session state
if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, str]] = []

# Chroma persistent collection 
os.makedirs(CHROMA_DIR, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY, model_name=EMBED_MODEL
)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME, embedding_function=openai_ef
)

def read_html_text(path: str) -> str:
    """Read visible text from an HTML file (scripts/styles removed)."""
    with open(path, "rb") as f:
        soup = BeautifulSoup(f.read(), "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    body = soup.find("body") or soup
    text = body.get_text(separator="\n", strip=True)
    return "\n".join([ln for ln in (t.strip() for t in text.splitlines()) if ln])

def two_chunks(text: str) -> List[str]:
    """
    CHUNKING (exactly two chunks per document).
    Method: simple 50/50 split by character count.
    Why: pages are short/medium; a midpoint split preserves coherence and
    satisfies the spec (“create two mini-docs from each supplied document”)
    without over-fragmenting.
    """
    if not text: return []
    mid = max(1, len(text)//2)
    return [text[:mid], text[mid:]]

def build_vector_db_once():
    """Create the vector DB only if empty; otherwise reuse existing persisted data."""
    try:
        if collection.count() > 0:
            return
    except Exception:
        pass
    html_paths = sorted(glob.glob(os.path.join(HTML_DIR, "*.html")))
    if not html_paths:
        st.error(f"No HTML files found in {HTML_DIR}.")
        st.stop()
    docs, ids, metas = [], [], []
    n = 0
    for p in html_paths:
        text = read_html_text(p)
        for j, ch in enumerate(two_chunks(text)):
            if not ch.strip(): continue
            n += 1
            docs.append(ch)
            ids.append(f"doc_{n}")
            metas.append({"filename": os.path.basename(p), "chunk_id": j})
    if not docs:
        st.error("No text extracted from any HTML files.")
        st.stop()
    collection.add(documents=docs, ids=ids, metadatas=metas)

def memory_messages() -> List[Dict[str, str]]:
    """Return last 5 Q&A pairs (10 messages)."""
    return st.session_state.history[-(MEM_KEEP*2):]

def retrieve_context(question: str) -> str:
    """Retrieve top chunks and format as a sources block."""
    res = collection.query(query_texts=[question], n_results=TOP_K)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    out = []
    for d, md in zip(docs, metas):
        out.append(f"[{md.get('filename','unknown')} | chunk {md.get('chunk_id',0)}]\n{d}")
    return "\n\n".join(out)

def stream_openai(messages: List[Dict[str, str]], model: str):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    stream = client.chat.completions.create(model=model, messages=messages, stream=True, temperature=0.2)
    for ch in stream:
        piece = ch.choices[0].delta.content or ""
        if piece: yield piece

def stream_mistral(messages: List[Dict[str, str]], model: str):
    from mistralai import Mistral
    client = Mistral(api_key=MISTRAL_API_KEY)
    resp = client.chat.complete(model=model, messages=messages, temperature=0.2, max_tokens=700)
    txt = resp.choices[0].message.content or ""
    for i in range(0, len(txt), 60): yield txt[i:i+60]

def stream_gemini(prompt_text: str, model: str):
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    gmodel = genai.GenerativeModel(model)
    resp = gmodel.generate_content(prompt_text, stream=True)
    for ev in resp:
        if getattr(ev, "text", ""): yield ev.text

def answer_with_rag(user_q: str) -> str:
    """RAG + memory, streamed via the selected model."""
    src_block = retrieve_context(user_q)
    system = {"role": "system", "content":
              "You answer questions about Syracuse iSchool student organizations. "
              "Use ONLY the provided sources; if unsure, say you don't know."}
    msgs = [system] + memory_messages()
    if src_block:
        msgs.append({"role":"user","content":f"SOURCES:\n\n{src_block}"})
    msgs.append({"role":"user","content":user_q})
    gemini_prompt = f"{system['content']}\n\nSOURCES:\n{src_block}\n\nQUESTION:\n{user_q}"

    chunks: List[str] = []
    def gen():
        try:
            if vendor == "OpenAI":
                src = stream_openai(msgs, MODEL)
            elif vendor == "Mistral":
                src = stream_mistral(msgs, MODEL)
            else:
                src = stream_gemini(gemini_prompt, MODEL)
            for p in src:
                chunks.append(p); yield p
        except Exception as e:
            yield f"\n\n[Error: {e}]"
    st.write_stream(gen())
    return "".join(chunks).strip()

# Build vector DB 
build_vector_db_once()

for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_q = st.chat_input("Ask about iSchool student organizations…")
if user_q:
    st.session_state.history.append({"role":"user","content":user_q})
    with st.chat_message("assistant"):
        ans = answer_with_rag(user_q)
    st.session_state.history.append({"role":"assistant","content":ans})
    if len(st.session_state.history) > MEM_KEEP*2:
        st.session_state.history = st.session_state.history[-(MEM_KEEP*2):]
