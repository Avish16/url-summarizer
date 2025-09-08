import streamlit as st
import requests
from bs4 import BeautifulSoup

# API keys from Streamlit secrets
OPENAI_API_KEY   = st.secrets.get("OPENAI_API_KEY")
MISTRAL_API_KEY  = st.secrets.get("MISTRAL_API_KEY")
GEMINI_API_KEY   = st.secrets.get("GEMINI_API_KEY")

st.title("🧾 HW 2: URL Summarizer (Multi-LLM: OpenAI / Mistral / Gemini)")

# 4) URL input at the top (not sidebar)
url = st.text_input("Enter a web page URL", placeholder="https://example.com/article")

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

# 10) Model map (OpenAI / Mistral / Gemini)
MODEL_MAP = {
    "OpenAI": {
        True:  "gpt-4o",       # advanced
        False: "gpt-4o-mini",  # cheaper
    },
    "Mistral": {
        True:  "mistral-large-latest",  # advanced
        False: "mistral-small-latest",  # cheaper
    },
    "Gemini": {
        True:  "gemini-1.5-pro",   # advanced
        False: "gemini-1.5-flash", # cheaper
    },
}
model_id = MODEL_MAP[provider][use_advanced]
st.caption(f"Using **{provider}** model: `{model_id}` | Output: **{language}** | Style: **{summary_style}**")

# 7) EXACT function provided (kept intact)
import requests
from bs4 import BeautifulSoup
def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None

# ---- Prompt builder (8,9) ----
def build_instruction(style: str, lang: str) -> str:
    base = f"Write the summary in {lang} only. No preamble or labels. Be faithful to the source."
    if style == "100 words":
        return f"Summarize the document in about 100 words. {base}"
    if style == "2 paragraphs":
        return f"Summarize the document in exactly two connected paragraphs (Paragraph 2 builds on Paragraph 1). {base}"
    return f"Summarize the document as exactly 5 concise bullet points capturing distinct key ideas. {base}"

# ---- Provider runners (with key validation) ----
def summarize_openai(text: str, model: str, instruction: str) -> str:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY missing in secrets.")
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    # light validation
    client.models.list()
    msgs = [
        {"role": "system", "content": "You are a careful summarizer. Preserve meaning and avoid fabrications."},
        {"role": "user", "content": f"{instruction}\n\n---\nDOCUMENT:\n{text}\n---"},
    ]
    resp = client.chat.completions.create(model=model, messages=msgs, temperature=0.2, max_tokens=800)
    return resp.choices[0].message.content

def summarize_mistral(text: str, model: str, instruction: str) -> str:
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY missing in secrets.")
    # SDK
    from mistralai import Mistral
    mclient = Mistral(api_key=MISTRAL_API_KEY)
    # light validation via a tiny call
    _ = mclient.models.list()
    # actual summarization
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
    # tiny validation
    _ = gmodel.generate_content("OK")
    # actual summarization
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
                # ensure bullets render as bullets if model returns plain lines
                if summary_style == "5 bullet points" and not out.strip().startswith(("-", "•")):
                    out = "\n".join([f"- {ln.strip()}" for ln in out.splitlines() if ln.strip()])
                st.markdown(out)
            except Exception as e:
                st.error(f"Summarization failed: {e}")
    else:
        st.warning("Could not extract text from the URL.")
else:
    st.info("Enter a URL above to generate a summary.", icon="🌐")
