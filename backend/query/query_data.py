from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from backend.faiss_index_handler import FAISSIndexHandler

index_handler = FAISSIndexHandler.get_instance()
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

documents = []

def add_to_index(docs):
    global documents
    documents.extend(docs)
    print(f"Documents added to FAISS index: {docs}")
    embeddings = embedding_model.encode(docs, convert_to_tensor=False)
    print(f"Generated embeddings: {embeddings}")
    index_handler.add_embeddings(embeddings)
    return embeddings

def query_rag(query_text):
    query_embedding = embedding_model.encode([query_text], convert_to_tensor=False)
    print(f"Query embedding: {query_embedding}")
    D, I = index_handler.index.search(query_embedding, k=5)
    
    print(f"FAISS Search Results: D={D}, I={I}")

    # Introduce a distance threshold
    threshold = 1.0
    valid_indices = [i for i, dist in zip(I[0], D[0]) if dist < threshold and i >= 0 and i < len(documents)]
    if not valid_indices:
        return "No relevant documents found.", []

    top_k_docs = [documents[i].page_content for i in valid_indices]
    
    print(f"Top-k Documents: {top_k_docs}")

    if not top_k_docs:
        return "No relevant documents found.", []

    input_text = " ".join(top_k_docs) + " " + query_text
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response, top_k_docs
