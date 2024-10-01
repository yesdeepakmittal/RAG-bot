from elasticsearch import Elasticsearch
import os

from dotenv import load_dotenv
load_dotenv()

client = Elasticsearch(
    cloud_id=os.getenv("ES_CLOUD_ID"),
    api_key=os.getenv("ES_API_KEY"),
)

index_name = os.getenv("ES_INDEX_NAME")
mapping = {
    "mappings": {
        "properties": {
            "text_chunk": {"type": "text"},
            "page_num": {"type": "integer"},
            "document_name": {"type": "text"},
            "embedding": {
                "type": "dense_vector",
                "dims": 1536
            }
        }
    }
}

client.indices.create(index=index_name, body=mapping)