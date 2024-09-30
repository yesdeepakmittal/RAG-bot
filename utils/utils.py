import logging
import tiktoken

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """
    Returns the number of tokens in a text string using the specified encoding.

    Parameters
    ----------
    string : str
        The input text string.
    encoding_name : str, optional
        The name of the encoding to use (default is "cl100k_base").

    Returns
    -------
    int
        The number of tokens in the input string.
    """
    logger.info("Calculating number of tokens for the given string with encoding: %s", encoding_name)
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    logger.info("Number of tokens: %d", num_tokens)
    return num_tokens

def text_formatter(text: str) -> str:
    """
    Cleans up and formats text by removing unwanted newlines and spaces.

    Parameters
    ----------
    text : str
        The input text string.

    Returns
    -------
    str
        The cleaned and formatted text string.
    """
    logger.info("Formatting text")
    formatted_text = text.replace("\n", " ").strip()
    logger.info("Formatted text: %s", formatted_text)
    return formatted_text
