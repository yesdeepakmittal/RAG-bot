import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_chunk_es(client, index_name, chunk, embedding, chunk_id, metadata):
    """
    Ingest a chunk of text into the Elasticsearch index.

    Parameters
    ----------
    client : Elasticsearch
        The Elasticsearch client instance.
    index_name : str
        The name of the Elasticsearch index.
    chunk : str
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
            "text_chunk": chunk,
            "embedding": embedding,
            "page_num": metadata["page_num"],
            "document_name": metadata["document_name"]
        }
        client.index(index=index_name, id=chunk_id, body=doc)
        logger.info(f"Ingested chunk {chunk_id} into Elasticsearch index {index_name}")
    except Exception as e:
        logger.error(f"Failed to ingest chunk {chunk_id} into Elasticsearch: {str(e)}")
