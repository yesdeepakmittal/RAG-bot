import logging
import os

import openai

from src.embedding.embedder import get_embedding
from src.retrieval.retriever import query_chunks_es
from utils.utils import load_prompt_from_yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_response_with_gpt(documents, query_text):
    """
    Generate a response using GPT-3.5-turbo-instruct model.

    Parameters
    ----------
    documents : list of str
        The list of documents to use as context.
    query_text : str
        The query text to generate a response for.

    Returns
    -------
    str
        The generated response text.
    """
    context = "\n\n".join(documents)
    prompt_template = load_prompt_from_yaml("configs/prompt.yaml")
    prompt = prompt_template.format(context=context, query=query_text)
    logger.info(f"Generated prompt: {prompt}")

    logger.info("Generating response with GPT-3.5-turbo-instruct model.")
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=500,
        temperature=0.7,
        top_p=1.0,
        n=1,
    )
    generated_text = response["choices"][0]["text"].strip()
    logger.info(f"Generated response: {generated_text}")
    return generated_text


def generate_response(client, query_text):
    """
    Generate a response using the specified query text.

    Parameters
    ----------
    client : Elasticsearch
        The Elasticsearch client instance.
    query_text : str
        The query text to generate a response for.

    Returns
    -------
    str
        The generated response text.
    """

    query_embedding = get_embedding(query_text)

    results = query_chunks_es(client, query_embedding, n_results=5)

    documents = [hit["_source"]["text_chunk"] for hit in results["hits"]["hits"]]
    response = generate_response_with_gpt(documents, query_text)

    logger.info(f"Generated response: {response}")
    return response
