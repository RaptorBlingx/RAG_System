from sentence_transformers import SentenceTransformer

class LocalEmbeddings:
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text):
        embeddings = self.model.encode(text, convert_to_tensor=False)
        return embeddings.tolist()  # Convert to list

    def embed_documents(self, texts):
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()  # Convert to list

def get_embedding_function():
    print("Creating LocalEmbeddings instance with Sentence Transformers")  # Logging
    return LocalEmbeddings()
