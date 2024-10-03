import logging
import os

from src.chunking.chunking import get_individual_chunk, get_pdf_chunks
from src.embedding.embedder import get_embedding
from src.ingestion.indexing import ingest_chunk_es
from src.parsing.parser import get_document_text
from utils.utils import num_tokens_from_string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()


def run_ingestion_pipeline(client, ingest_flag=False):
    """
    Run the document ingestion pipeline.

    Parameters
    ----------
    client : Elasticsearch
        The Elasticsearch client instance.
    ingest_flag : bool, optional
        A flag to indicate whether to ingest the documents (default is False).
    """

    PDF_DIR = os.getenv("PDF_DIR")

    if ingest_flag:
        logger.info("Starting document ingestion.")
        for path in os.listdir(PDF_DIR):
            if path.endswith(".pdf"):
                doc_path = os.path.join(PDF_DIR, path)
                parsed_result = get_document_text(doc_path)
                chunked_result = get_pdf_chunks(parsed_result)
                individual_chunk = get_individual_chunk(chunked_result)

                for idx, chunk in enumerate(individual_chunk):
                    chunk_text = chunk["sentence_chunk"]
                    chunk_token_count = num_tokens_from_string(chunk_text)
                    embedding = get_embedding(chunk_text)

                    chunk_id = f"{doc_path.split('/')[-1]}_chunk_{idx}"
                    metadata = {
                        "chunk_id": chunk_id,
                        "page_num": chunk["page_num"],
                        "document_name": doc_path.split("/")[-1],
                        "chunk_token_count": chunk_token_count,
                    }

                    ingest_chunk_es(client, chunk_text, embedding, chunk_id, metadata)
                    logger.info(f"Ingested chunk {chunk_id} with metadata {metadata}")

                logger.info("****************************")
                logger.info(f"Ingested document {doc_path}")

        logger.info("Ingestion complete!")
