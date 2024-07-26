# get_embedding_function.py
from sentence_transformers import SentenceTransformer

class LocalEmbeddings:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text):
        return self.model.encode(text, convert_to_tensor=True)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=True)

def get_embedding_function():
    return LocalEmbeddings()
