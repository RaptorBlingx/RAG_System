import streamlit as st

def file_uploader():
    uploaded_file = st.file_uploader("Upload PDF files", type="pdf")
    if uploaded_file:
        with open(f"data/{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File uploaded successfully!")
