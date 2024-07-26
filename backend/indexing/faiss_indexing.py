import faiss
import pickle

def index_embeddings(embeddings):
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, 'faiss_index.index')
    print(f"Indexed {len(embeddings)} embeddings with FAISS")

def main():
    with open('embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    
    index_embeddings(embeddings)

if __name__ == "__main__":
    main()
