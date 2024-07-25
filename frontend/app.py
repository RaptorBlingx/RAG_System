import streamlit as st
from backend.query.query_data import query_rag
import time

def main():
    st.title("RAG System - Document Search and QA")
    st.write("Upload your PDF documents and ask questions based on their content.")

    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type="pdf")

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with open(f"data/{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success("File(s) uploaded successfully!")

    query = st.text_input("Enter your query:")
    if st.button("Submit"):
        with st.spinner('Processing...'):
            start_time = time.time()
            response = query_rag(query)
            end_time = time.time()
            response_time = end_time - start_time
            st.write(f"Response: {response}")
            st.write(f"Response time: {response_time:.2f} seconds")

if __name__ == "__main__":
    main()
