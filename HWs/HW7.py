import os
import re
import pandas as pd
import streamlit as st
from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from mistralai import Mistral

# Page setup 
st.title("HW 7 — News Info Bot 📰")

CSV_PATH = "data/news/news.csv"
CHROMA_DIR = "./ChromaDB_hw7"
COLLECTION_NAME = "HW7News"
EMBED_MODEL = "text-embedding-3-small"
TOP_K = 8
SHOW_K = 5
MEM_KEEP = 5 

# Sidebar: model selector
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

# Keys
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
MISTRAL_API_KEY = st.secrets.get("MISTRAL_API_KEY")

# Load CSV
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

#  Chroma DB
os.makedirs(CHROMA_DIR, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY, model_name=EMBED_MODEL
)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME, embedding_function=openai_ef
)

if collection.count() == 0:
    docs, ids, metas = [], [], []
    for i, r in news_df.iterrows():
        docs.append(r["_text"])
        ids.append(f"news_{i}")
        metas.append({
            "company": r.get("company_name", ""),
            "date": r.get("Date", ""),
            "url": r.get("_url", "")
        })
    collection.add(documents=docs, ids=ids, metadatas=metas)

# Retrieval
def retrieve_news(query: str, top_k: int = TOP_K):
    return collection.query(query_texts=[query], n_results=top_k)

def summarize_items(metas, docs):
    out = []
    for i, (m, d) in enumerate(zip(metas, docs), 1):
        out.append(f"{i}. {m.get('company','')} ({m.get('date','')})\n{d[:250]}")
    return "\n".join(out)

# LLM 
def call_llm(messages: List[Dict[str, str]]):
    if vendor == "OpenAI":
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(model=MODEL_NAME, messages=messages)
        return resp.choices[0].message.content.strip()
    else:
        client = Mistral(api_key=MISTRAL_API_KEY)
        resp = client.chat.complete(model=MODEL_NAME, messages=messages)
        return (resp.choices[0].message.content or "").strip()

def parse_rank(text: str, n: int):
    nums = [int(x) for x in re.findall(r"\b(\d+)\b", text)]
    seen, out = set(), []
    for x in nums:
        if 1 <= x <= n and x not in seen:
            out.append(x)
            seen.add(x)
    return out[:SHOW_K] if out else list(range(1, min(SHOW_K, n)+1))

# Memory setup 
# ----------------- Memory setup -----------------
if "history" not in st.session_state:
    st.session_state.history = []  # type: ignore
if "last_results" not in st.session_state:
    st.session_state.last_results = []  # keep last retrieved docs

# Chat Interface
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask something like 'find the most interesting news' or 'find news about AI regulation'")

if query:
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Prepare context for follow-ups
    if re.search(r"\b(\d+)(st|nd|rd|th)\b", query.lower()):
        # If user references a number ("4th news", "2nd one", etc.)
        context = st.session_state.last_results
    else:
        # Otherwise retrieve fresh context
        res = retrieve_news(query)
        docs, metas = res["documents"][0], res["metadatas"][0]
        context = [f"{i+1}. {m.get('company','')} ({m.get('date','')})\n{d[:250]}" for i, (m, d) in enumerate(zip(metas, docs))]
        st.session_state.last_results = context

        # Display ranked list before generating answer
        st.subheader("Top Results")
        for i, (m, d) in enumerate(zip(metas, docs), 1):
            st.markdown(f"**{i}. {m.get('company','(no title)')}** ({m.get('date','')})")
            st.write(d[:500] + ("..." if len(d) > 500 else ""))
            if m.get("url"):
                st.markdown(f"[Read more]({m['url']})")
            st.divider()

    # Create memory messages
    system = {"role": "system", "content": "You are a news bot for a global law firm. Use memory context and retrieved items to answer clearly."}
    msgs = [system] + st.session_state.history[-(MEM_KEEP*2):]
    msgs.append({"role": "user", "content": f"CONTEXT:\n{chr(10).join(context)}"})
    msgs.append({"role": "user", "content": query})

    # Generate response
    ans = call_llm(msgs)
    with st.chat_message("assistant"):
        st.markdown(ans)

    st.session_state.history.append({"role": "assistant", "content": ans})

    if len(st.session_state.history) > MEM_KEEP * 2:
        st.session_state.history = st.session_state.history[-(MEM_KEEP*2):]
