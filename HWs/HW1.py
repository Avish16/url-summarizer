import streamlit as st
from openai import OpenAI
import PyPDF2

#Function to read pdf
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Show title and description.
st.title("📄 Document Question Answering (HW-1)")
st.write(
    "Upload a `.txt` or `.pdf` file and ask a question about it. "
    "To use this app, you need to provide an OpenAI API key."
)

# API key input
openai_api_key = st.text_input("OpenAI API Key", type="password")

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝")
else:
    # Validate API Key
    try:
        client = OpenAI(api_key=openai_api_key)
        _ = client.models.list()  # quick validation
        st.success("✅ API key validated. You can upload a file and ask a question.")
    except Exception as e:
        st.error("Invalid API key or network issue. Please check your key.")
        st.caption(f"Details: {e}")
        st.stop()

    # File uploader 
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .pdf only)", type=("txt", "pdf")
    )

    document = None  # Reset if no file uploaded

    if uploaded_file:
        # Detect file type
        file_extension = uploaded_file.name.split(".")[-1].lower()

        if file_extension == "txt":
            document = uploaded_file.read().decode(errors="ignore")
        elif file_extension == "pdf":
            document = read_pdf(uploaded_file)
        else:
            st.error("Unsupported file type.")
            document = None

    # Question input 
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Example: Can you give me a short summary?",
        disabled=(document is None),
    )

    # Run QA
    if document and question:
        messages = [
            {
                "role": "user",
                "content": f"Here is a document:\n\n{document}\n\n---\n\n{question}",
            }
        ]

        # I will change model here
        model_choice = "gpt-5-nano"

        try:
            stream = client.chat.completions.create(
                model=model_choice,
                messages=messages,
                stream=True,
            )
            st.write_stream(stream)
        except Exception as e:
            st.error("Error while generating response.")
            st.caption(f"Details: {e}")