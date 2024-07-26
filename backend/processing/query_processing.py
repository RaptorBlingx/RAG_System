from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import faiss
import pickle

ES_INDEX = 'documents'

def load_faiss_index(index_path):
    return faiss.read_index(index_path)

def search_elasticsearch(es, index_name, query):
    response = es.search(index=index_name, body={"query": {"match": {"content": query}}})
    hits = response['hits']['hits']
    results = [{'content': hit['_source']['content'], 'score': hit['_score']} for hit in hits]
    return results

def search_faiss(faiss_index, query_embedding, k=5):
    D, I = faiss_index.search(query_embedding, k)
    return D, I

def main():
    query = "What is Creamobile?"
    
    # Encode query
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    query_embedding = embedding_model.encode([query], convert_to_tensor=False)
    
    # Elasticsearch search
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
    es_results = search_elasticsearch(es, ES_INDEX, query)
    
    # FAISS search
    faiss_index = load_faiss_index('faiss_index.index')
    D, I = search_faiss(faiss_index, query_embedding, k=5)
    
    with open('chunks.pkl', 'rb') as f:
        chunks = pickle.load(f)
    
    faiss_results = [{'content': chunks[i].page_content, 'score': D[0][idx]} for idx, i in enumerate(I[0])]

    # Combine results
    combined_results = es_results + faiss_results

    print("Elasticsearch Results:", es_results)
    print("FAISS Results:", faiss_results)
    print("Combined Results:", combined_results)

if __name__ == "__main__":
    main()
