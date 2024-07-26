from transformers import pipeline

def rerank_documents(query, documents):
    reranker = pipeline("text-classification", model="cross-encoder/ms-marco-MiniLM-L-6-v2", tokenizer="cross-encoder/ms-marco-MiniLM-L-6-v2")
    inputs = [{"text": query, "text_pair": doc['content']} for doc in documents]
    # Ensure the sequences are truncated to the model's max length
    scores = reranker(inputs, truncation=True)
    sorted_docs = sorted(zip(documents, scores), key=lambda x: x[1]['score'], reverse=True)
    return [doc for doc, score in sorted_docs]

def main():
    query = "What is Creamobile?"
    
    # Assume combined_results is obtained from previous step
    combined_results = [...]  # Replace with actual results
    
    final_results = rerank_documents(query, combined_results)
    print(final_results)

if __name__ == "__main__":
    main()
