import logging
import os
import time

import openai
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# https://learn.microsoft.com/en-us/answers/questions/1188074/text-embedding-ada-002-token-context-length
MAX_TOKENS = 8191


def truncate_text(text, model):
    """
    Truncate text if token count exceeds the maximum allowed tokens.

    Parameters
    ----------
    text : str
        The input text to truncate.
    model : str
        The model to be used for generating the embedding.

    Returns
    -------
    str
        The truncated text if token count exceeds MAX_TOKENS, else the original text.
    """

    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)

    if len(tokens) > MAX_TOKENS:
        logger.warning(f"Text exceeds {MAX_TOKENS} tokens. Truncating.")

        truncated_tokens = tokens[:MAX_TOKENS]
        return encoding.decode(truncated_tokens)

    return text


@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5))
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
    """
    logger.info("Requesting embedding for text: %s", text)

    truncated_text = truncate_text(text, model)

    try:
        response = openai.Embedding.create(input=truncated_text, model=model)
        embedding = response["data"][0]["embedding"]
        logger.info("Successfully retrieved embedding")
        return embedding

    except openai.error.RateLimitError as e:
        logger.warning(f"Rate limit exceeded: {e}. Retrying...")
        raise

    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI error occurred: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
