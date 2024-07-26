import faiss
import os
from sentence_transformers import SentenceTransformer

class FAISSIndexHandler:
    _instance = None
    FAISS_INDEX_PATH = "faiss_index.index"

    @staticmethod
    def get_instance():
        if FAISSIndexHandler._instance is None:
            FAISSIndexHandler._instance = FAISSIndexHandler()
        return FAISSIndexHandler._instance

    def __init__(self):
        if self.__class__._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            self.load_index()

    def load_index(self):
        if os.path.exists(self.FAISS_INDEX_PATH):
            self.index = faiss.read_index(self.FAISS_INDEX_PATH)
            print(f"FAISS index loaded from {self.FAISS_INDEX_PATH}")
        else:
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
            print(f"Initialized new FAISS index")

    def save_index(self):
        faiss.write_index(self.index, self.FAISS_INDEX_PATH)
        print(f"FAISS index saved to {self.FAISS_INDEX_PATH}")

    def add_embeddings(self, embeddings):
        self.index.add(embeddings)
        print("Embeddings added to FAISS index.")
