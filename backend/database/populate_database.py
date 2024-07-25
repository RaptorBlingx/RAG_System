import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import shutil
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from backend.embedding.get_embedding_function import get_embedding_function
from langchain_chroma import Chroma

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = document_loader.load()
    print(f"Loaded documents: {documents}")  # Logging
    return documents

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into chunks: {chunks}")  # Logging
    return chunks

def add_to_chroma(chunks: list[Document]):
    embedding_function = get_embedding_function()
    print(f"Embedding function instance: {embedding_function}")  # Logging
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    chunks_with_ids = calculate_chunk_ids(chunks)
    print(f"Chunks with IDs: {chunks_with_ids}")  # Logging

    existing_items = db.get(include=[])
    print(f"Existing items in DB: {existing_items}")  # Logging
    existing_ids = set(existing_items["ids"])
    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]
    print(f"New chunks to add: {new_chunks}")  # Logging

    if new_chunks:
        batch_size = 10  # Adjust the batch size as needed
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i:i+batch_size]
            print(f"Adding batch {i//batch_size + 1} to Chroma")  # Debugging line
            try:
                print(f"Batch embeddings: {embedding_function.embed_documents([chunk.page_content for chunk in batch])}")  # Log embeddings
                db.add_documents(batch, ids=[chunk.metadata["id"] for chunk in batch])  # Add batch of chunks
            except Exception as e:
                print(f"Error adding documents to Chroma: {e}")

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    print(f"Calculated chunk IDs: {chunks}")  # Logging
    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()
