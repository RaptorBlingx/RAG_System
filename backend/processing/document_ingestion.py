import os
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pickle

DATA_PATH = "data"

def load_documents(data_path):
    document_loader = PyPDFDirectoryLoader(data_path)
    documents = document_loader.load()
    print(f"Loaded {len(documents)} documents.")
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")
    return chunks

def generate_embeddings(chunks):
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = embedding_model.encode([chunk.page_content for chunk in chunks], convert_to_tensor=False)
    print(f"Generated embeddings for {len(chunks)} chunks.")
    return embeddings

def main():
    documents = load_documents(DATA_PATH)
    chunks = split_documents(documents)
    embeddings = generate_embeddings(chunks)
    
    # Save chunks and embeddings for later use
    with open('chunks.pkl', 'wb') as f:
        pickle.dump(chunks, f)
    
    with open('embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)

if __name__ == "__main__":
    main()
