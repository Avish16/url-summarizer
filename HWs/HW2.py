import streamlit as st
import requests
from bs4 import BeautifulSoup

# ---- API keys from Streamlit secrets ----
OPENAI_API_KEY   = st.secrets.get("OPENAI_API_KEY")
MISTRAL_API_KEY  = st.secrets.get("MISTRAL_API_KEY")
GEMINI_API_KEY   = st.secrets.get("GEMINI_API_KEY")

st.title("🧾 HW 2: URL Summarizer (Multi-LLM: OpenAI / Mistral / Gemini)")

# 4) URL input at the top (not sidebar)
url = st.text_input("Enter a web page URL", placeholder="https://en.wikipedia.org/wiki/India_national_cricket_team")

# 5) Sidebar menus: summary type, output language, model selection
with st.sidebar:
    st.header("Summary options")
    summary_style = st.radio(
        "Summary type",
        ["100 words", "2 paragraphs", "5 bullet points"],
        index=0
    )

    language = st.selectbox(
        "Output language",
        ["English", "French", "Spanish", "Hindi"],  # ≥ 3 options
        index=0
    )

    st.divider()
    st.header("Model selection")
    provider = st.selectbox("LLM provider", ["OpenAI", "Mistral", "Gemini"], index=0)
    use_advanced = st.checkbox("Use Advanced Model", value=False)

# 10) Model map
MODEL_MAP = {
    "OpenAI": {
        True:  "gpt-4o",
        False: "gpt-4o-mini",
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
model_id = MODEL_MAP[provider][use_advanced]
st.caption(f"Using **{provider}** model: `{model_id}` | Output: **{language}** | Style: **{summary_style}**")

# 7) Hardened URL reader (headers + lxml + content focus)
def read_url_content(url: str) -> str | None:
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.content, "html.parser")
        
        # Strip non-content tags
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        # Prefer main article area (Wikipedia etc.), else <main>, else <body>
        main = soup.select_one("#mw-content-text") or soup.select_one("main") or soup.body
        if not main:
            return None

        text = main.get_text(separator="\n", strip=True)
        lines = [ln for ln in (t.strip() for t in text.splitlines()) if ln]
        return "\n".join(lines) if lines else None

    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None

# ---- Prompt builder (8,9)
def build_instruction(style: str, lang: str) -> str:
    base = f"Write the summary in {lang} only. No preamble or labels. Be faithful to the source."
    if style == "100 words":
        return f"Summarize the document in about 100 words. {base}"
    if style == "2 paragraphs":
        return f"Summarize the document in exactly two connected paragraphs (Paragraph 2 builds on Paragraph 1). {base}"
    return f"Summarize the document as exactly 5 concise bullet points capturing distinct key ideas. {base}"

# ---- Provider runners (with light key validation)
def summarize_openai(text: str, model: str, instruction: str) -> str:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY missing in secrets.")
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    client.models.list()  # validate
    msgs = [
        {"role": "system", "content": "You are a careful summarizer. Preserve meaning and avoid fabrications."},
        {"role": "user", "content": f"{instruction}\n\n---\nDOCUMENT:\n{text}\n---"},
    ]
    resp = client.chat.completions.create(model=model, messages=msgs, temperature=0.2, max_tokens=800)
    return resp.choices[0].message.content

def summarize_mistral(text: str, model: str, instruction: str) -> str:
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY missing in secrets.")
    from mistralai import Mistral
    mclient = Mistral(api_key=MISTRAL_API_KEY)
    _ = mclient.models.list()  # validate
    msgs = [
        {"role": "system", "content": "You are a careful summarizer. Preserve meaning and avoid fabrications."},
        {"role": "user", "content": f"{instruction}\n\n---\nDOCUMENT:\n{text}\n---"},
    ]
    resp = mclient.chat.complete(model=model, messages=msgs, temperature=0.2, max_tokens=800)
    return resp.choices[0].message.content

def summarize_gemini(text: str, model: str, instruction: str) -> str:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY missing in secrets.")
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    gmodel = genai.GenerativeModel(model)
    _ = gmodel.generate_content("OK")  # validate
    prompt = f"{instruction}\n\n---\nDOCUMENT:\n{text}\n---"
    resp = gmodel.generate_content(prompt)
    return resp.text

def run_summary(text: str, provider: str, model: str, style: str, lang: str) -> str:
    instruction = build_instruction(style, lang)
    if provider == "OpenAI":
        return summarize_openai(text, model, instruction)
    if provider == "Mistral":
        return summarize_mistral(text, model, instruction)
    if provider == "Gemini":
        return summarize_gemini(text, model, instruction)
    raise ValueError("Unsupported provider selected.")

# 6) Execute: read URL, summarize, display
if url:
    doc_text = read_url_content(url)
    if doc_text:
        with st.spinner("Summarizing…"):
            try:
                out = run_summary(doc_text, provider, model_id, summary_style, language)
                if summary_style == "5 bullet points" and not out.strip().startswith(("-", "•")):
                    out = "\n".join([f"- {ln.strip()}" for ln in out.splitlines() if ln.strip()])
                st.markdown(out)
            except Exception as e:
                st.error(f"Summarization failed: {e}")
    else:
        st.warning("Could not extract text from the URL.")
else:
    st.info("Enter a URL above to generate a summary.", icon="🌐")

