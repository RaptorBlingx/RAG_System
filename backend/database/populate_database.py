import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import shutil
import time
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from backend.faiss_index_handler import FAISSIndexHandler

CHROMA_PATH = "chroma"
DATA_PATH = "data"

index_handler = FAISSIndexHandler.get_instance()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        clear_database()

    documents = load_documents()
    if not documents:
        print("No documents loaded. Please ensure there are PDF files in the data directory.")
        return

    chunks = split_documents(documents)
    if not chunks:
        print("No document chunks created. Please check the document splitting process.")
        return

    add_documents_to_index(chunks)
    index_handler.save_index()

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = document_loader.load()
    print(f"Loaded documents: {documents}")
    return documents

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into chunks: {chunks}")
    return chunks

def add_documents_to_index(chunks: list[Document]):
    document_texts = [chunk.page_content for chunk in chunks]
    if not document_texts:
        print("No document texts to add to index.")
        return

    print(f"Document texts to be indexed: {document_texts}")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedding_model.encode(document_texts, convert_to_tensor=False)
    print(f"Generated embeddings: {embeddings}")
    index_handler.add_embeddings(embeddings)
    print(f"Added documents to index: {embeddings}")

def clear_database():
    if os.path.exists(CHROMA_PATH):
        print(f"Clearing database at path: {CHROMA_PATH}")
        for _ in range(5):
            try:
                shutil.rmtree(CHROMA_PATH)
                break
            except PermissionError as e:
                print(f"PermissionError: {e}. Retrying...")
                time.sleep(1)
        else:
            print(f"Failed to clear database at path: {CHROMA_PATH}")

if __name__ == "__main__":
    main()
