import chromadb
import logging

chroma_client = chromadb.Client()

# Use OpenAI embeddings with 1536-dimensional vectors
collection = chroma_client.create_collection(name="document_collection")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_chunk(chunk, embedding, chunk_id, metadata):
    """
    Ingest a chunk of text into the ChromaDB collection.

    Parameters
    ----------
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
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[chunk_id]
        )
        logger.info(f"Ingested chunk {chunk_id} into collection")
    except Exception as e:
        logger.error(f"Failed to ingest chunk {chunk_id}: {e}")