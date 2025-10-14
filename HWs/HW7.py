import sys, os, streamlit as st
from typing import List, Dict
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from mistralai import Mistral

# ----------------- Page setup -----------------
st.title("HW 7 — News Info Bot 📰")

# Paths & Constants
CSV_PATH = "data/news/news.csv"
CHROMA_DIR = ".ChromaDB_hw7"
COLLECTION_NAME = "HW7Collection"
EMBED_MODEL = "text-embedding-3-small"
TOP_K = 6
MEM_KEEP = 5

# ----------------- Sidebar for model selection -----------------
with st.sidebar:
    st.subheader("Model Settings")
    vendor = st.selectbox("Vendor", ["OpenAI", "Mistral"])
    advanced = st.checkbox("Use advanced model", value=True)

MODEL_MAP = {
    "OpenAI":  {True: "gpt-4o", False: "gpt-4o-mini"},
    "Mistral": {True: "mistral-large-latest", False: "mistral-small-latest"},
}
MODEL_NAME = MODEL_MAP[vendor][advanced]
st.caption(f"Using **{vendor}** — `{MODEL_NAME}`")

# ----------------- API Keys -----------------
OPENAI_API_KEY  = st.secrets.get("OPENAI_API_KEY")
MISTRAL_API_KEY = st.secrets.get("MISTRAL_API_KEY")

# ----------------- Load CSV -----------------
def load_news(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        st.error(f"CSV not found at {csv_path}")
        st.stop()
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    required = ["company_name", "document", "date"]
    if not all(col in cols for col in required):
        st.error("CSV must have 'company_name', 'Document', and 'Date' columns.")
        st.stop()
    df["_text"] = (
        df[cols["company_name"]].astype(str)
        + " — "
        + df[cols["document"]].astype(str)
        + " (" + df[cols["date"]].astype(str) + ")"
    )
    df["_url"] = df[cols["url"]] if "url" in cols else ""
    return df

news_df = load_news(CSV_PATH)

# ----------------- ChromaDB Setup -----------------
os.makedirs(CHROMA_DIR, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name=EMBED_MODEL)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=openai_ef)

if collection.count() == 0:
    docs, ids, metas = [], [], []
    for i, r in news_df.iterrows():
        docs.append(r["_text"])
        ids.append(f"news_{i}")
        metas.append({"company": r.get("company_name", ""), "date": r.get("Date", ""), "url": r.get("_url", "")})
    collection.add(documents=docs, ids=ids, metadatas=metas)

# ----------------- Retrieval -----------------
def retrieve_relevant_news(query: str) -> str:
    """Return top matching news as context."""
    res = collection.query(query_texts=[query], n_results=TOP_K)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    out = []
    for i, (d, md) in enumerate(zip(docs, metas), 1):
        title = md.get("company", "Unknown Company")
        date = md.get("date", "")
        url = md.get("url", "")
        out.append(f"[{i}] {title} ({date})\n{d}\nURL: {url}")
    return "\n\n".join(out)

# ----------------- LLM setup -----------------
def llm_client():
    if vendor == "OpenAI":
        return OpenAI(api_key=OPENAI_API_KEY)
    else:
        return Mistral(api_key=MISTRAL_API_KEY)

def answer_with_context(query: str) -> str:
    """Do retrieval, use context + memory, and return LLM response."""
    context = retrieve_relevant_news(query)

    system = {"role": "system", "content": (
        "You are a professional legal news assistant for a global law firm. "
        "Use the provided news sources to answer queries or follow-ups precisely. "
        "If the user asks 'explain 3rd news' or similar, refer to the previous news ranking."
    )}

    # memory: last 5 Q&A pairs
    msgs = [system] + st.session_state.history[-(MEM_KEEP*2):]

    # add RAG context as user message
    if context:
        msgs.append({"role": "user", "content": f"SOURCES:\n{context}"})
    msgs.append({"role": "user", "content": query})

    client = llm_client()
    if vendor == "OpenAI":
        resp = client.chat.completions.create(model=MODEL_NAME, messages=msgs, temperature=0.3)
        return resp.choices[0].message.content.strip()
    else:
        resp = client.chat.complete(model=MODEL_NAME, messages=msgs, temperature=0.3)
        return resp.choices[0].message.content.strip()

# ----------------- Chat UI -----------------
if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, str]] = []

st.markdown("💬 Ask questions like *'find the most interesting news'* or *'find news about AI regulation'*")

for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_q = st.chat_input("Ask about the news...")

if user_q:
    st.session_state.history.append({"role": "user", "content": user_q})
    with st.chat_message("assistant"):
        ans = answer_with_context(user_q)
        st.markdown(ans)
    st.session_state.history.append({"role": "assistant", "content": ans})

    # keep only last MEM_KEEP pairs
    if len(st.session_state.history) > MEM_KEEP*2:
        st.session_state.history = st.session_state.history[-(MEM_KEEP*2):]
