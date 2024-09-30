import chromadb
import logging

chroma_client = chromadb.Client()
collection = chroma_client.get_collection(name="document_collection")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def query_chunks(query_text: str, n_results: int = 5):
    """
    Query ChromaDB to retrieve the top N relevant chunks.

    Parameters
    ----------
    query_text : str
        The text to query against the document collection.
    n_results : int, optional
        The number of top relevant chunks to retrieve (default is 5).

    Returns
    -------
    dict or None
        A dictionary containing the retrieved documents and metadata if the query is successful,
        None otherwise.
    """
    try:
        results = collection.query(query_texts=[query_text], n_results=n_results)
        logger.info(f"Retrieved {len(results['documents'])} results for query: {query_text}")
        return results
    except Exception as e:
        logger.error(f"Failed to query ChromaDB: {str(e)}")
        return None
