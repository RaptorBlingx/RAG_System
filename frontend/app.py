import sys
import os
import time

# Add the root directory of the project to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pickle
from backend.processing.query_processing import search_faiss, search_elasticsearch, load_faiss_index
from backend.processing.rerank_documents import rerank_documents
from backend.processing.response_generation import generate_response
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch

ES_INDEX = 'documents'

def upload_files():
    st.title("RAG System - Document Search and QA")
    st.write("Upload your PDF documents and ask questions based on their content.")

    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type="pdf")
    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_path = os.path.join("data", uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success("Files uploaded successfully! Now you can query the documents.")

def process_query(query):
    # Print the current working directory
    print("Current working directory:", os.getcwd())

    # Encode query
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    query_embedding = embedding_model.encode([query], convert_to_tensor=False)
    
    # Elasticsearch search
    es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
    es_results = search_elasticsearch(es, ES_INDEX, query)
    
    # FAISS search
    faiss_index = load_faiss_index('faiss_index.index')
    D, I = search_faiss(faiss_index, query_embedding, k=5)
    
    with open('chunks.pkl', 'rb') as f:
        chunks = pickle.load(f)
    
    # Ensure indices are within range
    faiss_results = [{'content': chunks[i].page_content, 'score': D[0][idx]} for idx, i in enumerate(I[0]) if i < len(chunks)]
    
    # Combine results
    combined_results = es_results + faiss_results
    
    # Rerank combined results
    final_results = rerank_documents(query, combined_results)

    return final_results

def display_results(results):
    for result in results:
        st.write(result['content'])

def main():
    upload_files()
    
    query = st.text_input("Enter your query:")
    if st.button("Submit"):
        start_time = time.time()
        with st.spinner('Processing...'):
            results = process_query(query)
            st.success('Query processed successfully!')
            
            response = generate_response(query, results)
            st.write("Generated Response:")
            st.write(response)
        
        end_time = time.time()
        response_time = end_time - start_time
        st.write(f"Response time: {response_time:.2f} seconds")

        if st.checkbox("Show raw results"):
            display_results(results)

if __name__ == "__main__":
    main()
