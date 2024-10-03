import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()

index_name = os.getenv("ES_INDEX_NAME")


def query_chunks_es(client, query_embedding, n_results=50):
    """
    Query Elasticsearch to retrieve the top N relevant chunks based on the embedding.

    Parameters
    ----------
    client : Elasticsearch
        The Elasticsearch client instance.
    query_embedding : list
        The embedding of the query text.
    n_results : int, optional
        The number of top relevant chunks to retrieve (default is 50).

    Returns
    -------
    dict or None
        A dictionary containing the retrieved documents and metadata if the query is successful,
        None otherwise.
    """
    try:
        body = {
            "size": n_results,
            "_source": ["chunk_text", "page_num", "document_name", "chunk_token_count"],
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_embedding},
                    },
                }
            },
        }
        results = client.search(index=index_name, body=body)
        logger.info(f"Retrieved {len(results['hits']['hits'])} results from Elasticsearch")
        return results
    except Exception as e:
        logger.error(f"Failed to query Elasticsearch: {str(e)}")
        return None
