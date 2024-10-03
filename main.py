import os

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

from src.generation.response_generation import generate_response
from src.ingestion.indexing import create_index_if_not_exists
from src.ingestion.ingestion import run_ingestion_pipeline

load_dotenv()

client = Elasticsearch(
    cloud_id=os.getenv("ES_CLOUD_ID"),
    api_key=os.getenv("ES_API_KEY"),
)

# Ensure the index exists
create_index_if_not_exists(client)

if __name__ == "__main__":
    # Run the ingestion pipeline
    ingest_flag = False
    run_ingestion_pipeline(client, ingest_flag)

    # Generate a response using GPT-3.5
    query_text = "Sabka Saath, Sabka Vikas"
    response = generate_response(client, query_text)
