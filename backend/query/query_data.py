import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from backend.embedding.get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    print(f"Embedding function instance: {embedding_function}")  # Logging
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=5)
    print(f"Query results: {results}")  # Logging

    if not results:
        return "No relevant documents found.", []

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    model = Ollama(model="llama3:latest")
    try:
        response_text = model.invoke(prompt)
    except Exception as e:
        response_text = f"Error invoking Ollama model: {e}"
    sources = [doc.metadata.get("id", None) for doc, _score in results]

    print(f"Final response: {response_text}")  # Logging
    return f"Response: {response_text}\nSources: {sources}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    response = query_rag(args.query_text)
    print(response)