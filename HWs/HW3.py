import streamlit as st
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Generator

# ========== Secrets ==========
OPENAI_API_KEY  = st.secrets.get("OPENAI_API_KEY")
MISTRAL_API_KEY = st.secrets.get("MISTRAL_API_KEY")
GEMINI_API_KEY  = st.secrets.get("GEMINI_API_KEY")

st.title("💬 HW 3: Streaming Chatbot over URLs")

# Sidebar: URLs, Vendor, Model, Memory 
with st.sidebar:
    st.header("Inputs & Settings")

    url1 = st.text_input("URL 1", placeholder="https://www.example.com/page-1")
    url2 = st.text_input("URL 2 (optional)", placeholder="https://www.example.com/page-2")

    vendor = st.selectbox("LLM vendor", ["OpenAI", "Mistral", "Gemini"], index=0)
    use_advanced = st.checkbox("Use Advanced (flagship) model", value=False)

    MODEL_MAP = {
        "OpenAI": {
            True:  "gpt-4o",        # flagship
            False: "gpt-4o-mini",   # cheap
        },
        "Mistral": {
            True:  "mistral-large-latest",
            False: "mistral-small-latest",
        },
        "Gemini": {
            True:  "gemini-2.5-flash",
            False: "gemini-2.5-flash-lite",
        },
    }
    model_id = MODEL_MAP[vendor][use_advanced]

    memory_mode = st.selectbox(
        "Conversation memory",
        ["Buffer: last 6 Q&A", "Conversation summary", "Token buffer: 2000 tokens"],
        index=0
    )

st.caption(f"Using **{vendor}** model: `{model_id}`  |  Memory: **{memory_mode}**")

# Session State
if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, str]] = []

if "conv_summary" not in st.session_state:
    st.session_state.conv_summary: str = ""

if "sources" not in st.session_state:
    st.session_state.sources = {"url1": "", "url2": "", "text1": None, "text2": None}

# URL Reader (lxml only)
def read_url_content(u: str) -> str | None:
    if not u:
        return None
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(u, headers=headers, timeout=20)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.content, "lxml")  

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        main = soup.select_one("#mw-content-text") or soup.select_one("main") or soup.body
        if not main:
            return None

        text = main.get_text(separator="\n", strip=True)
        lines = [ln for ln in (t.strip() for t in text.splitlines()) if ln]
        return "\n".join(lines) if lines else None

    except requests.RequestException as e:
        st.warning(f"Error reading {u}: {e}")
        return None

def load_sources():
    if st.session_state.sources["url1"] != url1:
        st.session_state.sources["url1"] = url1
        st.session_state.sources["text1"] = read_url_content(url1) if url1 else None
    if st.session_state.sources["url2"] != url2:
        st.session_state.sources["url2"] = url2
        st.session_state.sources["text2"] = read_url_content(url2) if url2 else None

def truncate(text: str, max_chars: int) -> str:
    return text if text is None or len(text) <= max_chars else text[:max_chars] + "\n...[truncated]..."

# Memory Builders
def build_memory_messages() -> List[Dict[str, str]]:
    hist = st.session_state.history

    if memory_mode == "Buffer: last 6 Q&A":
        return hist[-12:]  # 6 user+assistant pairs

    if memory_mode == "Token buffer: 2000 tokens":
        budget = 8000  # ~4 chars per token
        acc: List[Dict[str, str]] = []
        total = 0
        for msg in reversed(hist):
            c = len(msg["content"])
            if total + c > budget:
                break
            acc.append(msg)
            total += c
        return list(reversed(acc))

    if memory_mode == "Conversation summary" and st.session_state.conv_summary:
        return [{"role": "system", "content": f"Conversation summary so far:\n{st.session_state.conv_summary}"}]

    return []

def update_conversation_summary(user_msg: str, assistant_msg: str):
    if memory_mode != "Conversation summary":
        return
    cheap_model = {
        "OpenAI": "gpt-4o-mini",
        "Mistral": "mistral-small-latest",
        "Gemini": "gemini-2.5-flash-lite",
    }[vendor]

    instruction = (
        "Update the running conversation summary. Keep it concise, factual, and include only key points. "
        "Do not repeat the full conversation. Preserve prior summary content that remains relevant."
    )
    prior = st.session_state.conv_summary or "(empty)"
    text = f"PRIOR SUMMARY:\n{prior}\n\nNEW EXCHANGE:\nUser: {user_msg}\nAssistant: {assistant_msg}\n\nUPDATED SUMMARY:"

    try:
        if vendor == "OpenAI":
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model=cheap_model,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": text},
                ],
                temperature=0.2,
            )
            st.session_state.conv_summary = resp.choices[0].message.content.strip()

        elif vendor == "Mistral":
            from mistralai import Mistral
            client = Mistral(api_key=MISTRAL_API_KEY)
            resp = client.chat.complete(
                model=cheap_model,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": text},
                ],
                temperature=0.2,
                max_tokens=400,
            )
            st.session_state.conv_summary = resp.choices[0].message.content.strip()

        elif vendor == "Gemini":
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            gmodel = genai.GenerativeModel(cheap_model)
            prompt = f"{instruction}\n\n{text}"
            resp = gmodel.generate_content(prompt)
            st.session_state.conv_summary = (resp.text or "").strip()
    except Exception as e:
        st.warning(f"Summary update failed: {e}")

# Build Prompt/Context
def build_context_block() -> str:
    s1 = truncate(st.session_state.sources.get("text1"), 6000) if st.session_state.sources.get("text1") else None
    s2 = truncate(st.session_state.sources.get("text2"), 6000) if st.session_state.sources.get("text2") else None
    parts = []
    if s1:
        parts.append("SOURCE 1:\n" + s1)
    if s2:
        parts.append("SOURCE 2:\n" + s2)
    return "\n\n".join(parts) if parts else ""

def build_messages(user_prompt: str) -> tuple[list[Dict[str, str]], str]:
    system = {
        "role": "system",
        "content": (
            "You are a helpful assistant. Answer based only on the provided sources and the conversation context. "
            "If the answer is not in the sources, say you don't know. Be concise and accurate."
        ),
    }
    memory_msgs = build_memory_messages()
    context = build_context_block()

    # Chat-style messages (OpenAI/Mistral)
    chat_messages = [system]
    chat_messages.extend(memory_msgs)
    if context:
        chat_messages.append({"role": "user", "content": f"Here are sources:\n\n{context}"})
    chat_messages.append({"role": "user", "content": user_prompt})

    
    source_block = f"SOURCES:\n{context}\n\n" if context else ""
    text_prompt = (
        f"{system['content']}\n\n"
        f"{source_block}"
        f"USER QUESTION:\n{user_prompt}"
    )
    return chat_messages, text_prompt

# Streaming per vendor 
def stream_openai(messages: List[Dict[str, str]], model: str) -> Generator[str, None, None]:
    from openai import OpenAI
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing.")
    client = OpenAI(api_key=OPENAI_API_KEY)
    stream = client.chat.completions.create(model=model, messages=messages, stream=True, temperature=0.2)
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield delta

def stream_mistral(messages: List[Dict[str, str]], model: str) -> Generator[str, None, None]:
    # Pseudo-stream: call once, then yield progressively in chunks
    from mistralai import Mistral
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY missing.")
    client = Mistral(api_key=MISTRAL_API_KEY)
    resp = client.chat.complete(model=model, messages=messages, temperature=0.2, max_tokens=800)
    text = resp.choices[0].message.content or ""
    for i in range(0, len(text), 60):
        yield text[i:i+60]

def stream_gemini(prompt_text: str, model: str) -> Generator[str, None, None]:
    import google.generativeai as genai
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY missing.")
    genai.configure(api_key=GEMINI_API_KEY)
    gmodel = genai.GenerativeModel(model)
    resp = gmodel.generate_content(prompt_text, stream=True)
    for ev in resp:
        if hasattr(ev, "text") and ev.text:
            yield ev.text

def stream_and_collect(user_prompt: str) -> str:
    messages, text_prompt = build_messages(user_prompt)
    chunks: List[str] = []

    def gen():
        try:
            if vendor == "OpenAI":
                src = stream_openai(messages, model_id)
            elif vendor == "Mistral":
                src = stream_mistral(messages, model_id)
            else:  # Gemini
                src = stream_gemini(text_prompt, model_id)
            for piece in src:
                chunks.append(piece)
                yield piece
        except Exception as e:
            yield f"\n\n[Error: {e}]"

    st.write_stream(gen())
    return "".join(chunks).strip()

# Chat UI
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

load_sources()

user_input = st.chat_input("Ask about the URLs (e.g., rules, terms, differences, examples)…")

if user_input:
    if not (st.session_state.sources.get("text1") or st.session_state.sources.get("text2")):
        st.warning("Please provide at least one valid URL in the sidebar.")
    else:
        st.session_state.history.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            assistant_text = stream_and_collect(user_input)

        st.session_state.history.append({"role": "assistant", "content": assistant_text})

        update_conversation_summary(user_input, assistant_text)
