import logging
import re
from typing import List

import spacy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading SpaCy model...")
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("SpaCy model loaded successfully.")
except Exception as e:
    logger.exception("Failed to load SpaCy model.")
    raise e


def split_list(input_list: list, slice_size: int) -> list[list[str]]:
    """
    Splits the input_list into sublists of size slice_size.

    Parameters
    ----------
    input_list : list
        The list to be split into chunks.
    slice_size : int
        The size of each chunk.

    Returns
    -------
    list[list[str]]
        A list of sublists, each containing elements of the original list.
    """

    logger.debug(f"Splitting list into chunks of size {slice_size}.")
    return [input_list[i : i + slice_size] for i in range(0, len(input_list), slice_size)]


def get_pdf_chunks(parsed_result: List, num_sentence_chunk_size=5) -> List[dict]:
    """
    Chunk parsed PDF content into sentences, then group them into chunks.

    Parameters
    ----------
    parsed_result : List
        A list of dictionaries
    num_sentence_chunk_size : int, optional
        The number of sentences to include in each chunk (default is 5).

    Returns
    -------
    List[dict]
        A list of dictionaries, each containing 'page_num', 'text', 'sentences', 'sentence_chunks', and 'num_chunks'.
    """

    chunks = []

    for page in parsed_result:
        item = {}
        item["page_num"] = page["page_num"]
        item["text"] = page["text"]
        item["token_count"] = page["token_count"]
        logger.debug(f"Processing page number {item['page_num']}.")
        doc = nlp(item["text"])
        item["sentences"] = [sent.text for sent in doc.sents]
        item["sentence_chunks"] = split_list(
            input_list=item["sentences"], slice_size=num_sentence_chunk_size
        )
        item["num_chunks"] = len(item["sentence_chunks"])
        chunks.append(item)
        logger.debug(f"Page {item['page_num']} processed with {item['num_chunks']} chunks.")

    return chunks


def get_individual_chunk(chunked_result: List[dict]) -> List[dict]:
    """
    Flatten the chunked sentences into independent text chunks.

    Parameters
    ----------
    chunked_result : List[dict]
        A list of dictionaries

    Returns
    -------
    List[dict]
        A list of dictionaries, each containing 'page_num' and 'sentence_chunk'.
    """
    independent_chunks = []
    for item in chunked_result:
        for sentence_chunk in item["sentence_chunks"]:
            chunk_dict = {
                "page_num": item["page_num"],
                "sentence_chunk": re.sub(r"\s+", " ", "".join(sentence_chunk)).strip(),
            }
            independent_chunks.append(chunk_dict)
    return independent_chunks
