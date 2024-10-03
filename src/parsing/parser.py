import logging

import fitz

from utils.utils import num_tokens_from_string, text_formatter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()


def get_document_text(doc_path: str):
    """
    Extract and clean text from the PDF document.

    Parameters
    ----------
    doc_path : str
        The path to the PDF document.

    Returns
    -------
    list of dict or None
        A list of dictionaries containing page number, formatted text, and token count for each page.
        Returns None if an error occurs.
    """
    try:
        doc = fitz.open(doc_path)
        doc_store = []
        for idx, page in enumerate(doc):
            text = page.get_text("text")
            formatted_text = text_formatter(text)
            token_count = num_tokens_from_string(formatted_text)
            doc_store.append({"page_num": idx, "text": formatted_text, "token_count": token_count})
        logger.info(f"Parsed {len(doc_store)} pages from document {doc_path}")
        return doc_store
    except Exception as e:
        logger.error(f"Error while parsing document: {str(e)}")
        return None
