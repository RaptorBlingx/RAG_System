from celery import Celery
from backend.embedding.get_embedding_function import get_embedding_function

app = Celery('tasks', broker='redis://localhost:6379/0')

@app.task
def generate_embeddings(documents):
    embeddings = []
    embedding_function = get_embedding_function()
    for doc in documents:
        embedding = embedding_function.embed_documents([doc])
        embeddings.append(embedding)
    return embeddings
