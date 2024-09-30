import os
import openai
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(text, model="text-embedding-ada-002"):
    """
    Get the embedding for a given text using the specified model.

    Parameters
    ----------
    text : str
        The input text for which the embedding is to be generated.
    model : str, optional
        The model to be used for generating the embedding (default is "text-embedding-ada-002").

    Returns
    -------
    list
        A list representing the embedding of the input text.

    Raises
    ------
    Exception
        If there is an error in retrieving the embedding.
    """
    logger.info("Requesting embedding for text: %s", text)
    try:
        response = openai.Embedding.create(input=text, model=model)
        embedding = response["data"][0]["embedding"]
        logger.info("Successfully retrieved embedding")
        return embedding
    except Exception as e:
        logger.error("Error retrieving embedding: %s", e)
        raise
