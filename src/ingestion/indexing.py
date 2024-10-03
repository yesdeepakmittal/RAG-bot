import logging
import os

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


index_name = os.getenv("ES_INDEX_NAME")

mapping = {
    "mappings": {
        "properties": {
            "chunk_text": {"type": "text"},
            "page_num": {"type": "integer"},
            "document_name": {"type": "text"},
            "embedding": {"type": "dense_vector", "dims": 1536},
            "chunk_token_count": {"type": "integer"},
        }
    }
}


def create_index_if_not_exists(client, delete_existing=False):
    """
    Create an Elasticsearch index if it does not exist.

    Parameters
    ----------
    client : Elasticsearch
        The Elasticsearch client.
    delete_existing : bool, optional
        Whether to delete the existing index if it exists (default is False).
    """

    if client.indices.exists(index=index_name):
        if delete_existing:
            logger.info(f"Index {index_name} already exists. Deleting existing index.")

            client.indices.delete(index=index_name)
            logger.info(f"Index {index_name} deleted.")

            # Create the index with the specified mapping
            client.indices.create(index=index_name, body=mapping)
            logger.info(f"Index {index_name} created.")

        else:
            logger.info(f"Index {index_name} already exists.")

    else:
        logger.info(f"Index {index_name} does not exist. Creating index.")

        # Create the index with the specified mapping
        client.indices.create(index=index_name, body=mapping)
        logger.info(f"Index {index_name} created.")


def ingest_chunk_es(client, chunk_text, embedding, chunk_id, metadata):
    """
    Ingest a chunk of text into the Elasticsearch index.

    Parameters
    ----------
    client : Elasticsearch
        The Elasticsearch client instance.
    chunk_text : str
        The text chunk to be ingested.
    embedding : list of float
        The embedding vector for the text chunk.
    chunk_id : str
        The unique identifier for the text chunk.
    metadata : dict
        Additional metadata associated with the text chunk.

    Returns
    -------
    None
    """
    try:
        doc = {
            "chunk_text": chunk_text,
            "embedding": embedding,
            "page_num": metadata["page_num"],
            "document_name": metadata["document_name"],
            "chunk_token_count": metadata["chunk_token_count"],
        }
        client.index(index=index_name, id=chunk_id, body=doc)
        logger.info(f"Ingested chunk {chunk_id} into Elasticsearch index {index_name}")
    except Exception as e:
        logger.error(f"Failed to ingest chunk {chunk_id} into Elasticsearch: {str(e)}")
