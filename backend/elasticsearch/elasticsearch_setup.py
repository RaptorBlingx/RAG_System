from elasticsearch import Elasticsearch, helpers
import pickle

ES_INDEX = 'documents'

def connect_elasticsearch():
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
    if es.ping():
        print('Elasticsearch connected')
    else:
        print('Elasticsearch could not connect')
    return es

def create_index(es, index_name):
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, ignore=400, body={
            "mappings": {
                "properties": {
                    "content": {"type": "text"},
                    "metadata": {"type": "object"}
                }
            }
        })
        print(f"Created index: {index_name}")
    else:
        print(f"Index {index_name} already exists")

def index_documents(es, index_name, chunks):
    documents = [{"_index": index_name, "_source": {"content": chunk.page_content, "metadata": chunk.metadata}} for chunk in chunks]
    helpers.bulk(es, documents)
    print(f"Indexed {len(documents)} documents")

def main():
    es = connect_elasticsearch()
    create_index(es, ES_INDEX)
    
    with open('chunks.pkl', 'rb') as f:
        chunks = pickle.load(f)
    
    index_documents(es, ES_INDEX, chunks)

if __name__ == "__main__":
    main()
