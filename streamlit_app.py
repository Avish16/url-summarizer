import streamlit as st

st.set_page_config(page_title="HW manager", page_icon="📚", layout="wide")

pages = [
    st.Page("HWs/HW2.py", title="HW 2 — URL Summarizer", icon="🧾"),
    st.Page("HWs/HW1.py", title="HW 1 — Document QA", icon="📝"),
]

nav = st.navigation(pages)   # HW2 listed first → default page
nav.run()
