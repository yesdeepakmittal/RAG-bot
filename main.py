import os
import logging
import openai
import yaml

from src.chunking.chunking import get_individual_chunk, get_pdf_chunks
from src.embedding.embedder import get_embedding
from src.parsing.parser import get_document_text
from src.indexing.ingestion import ingest_chunk_es
from src.retrieval.retriever import query_chunks_es

from elasticsearch import Elasticsearch
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = Elasticsearch(
    cloud_id=os.getenv("ES_CLOUD_ID"),
    api_key=os.getenv("ES_API_KEY"),
)

index_name = os.getenv("ES_INDEX_NAME")
openai.api_key = os.getenv("OPENAI_API_KEY")  


def load_prompt_from_yaml(file_path):
    """
    Load the prompt template from a YAML file.

    Parameters
    ----------
    file_path : str
        The path to the YAML file containing the prompt template.

    Returns
    -------
    str
        The prompt template.
    """
    with open(file_path, 'r') as file:
        prompt_config = yaml.safe_load(file)
    return prompt_config['prompt']


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
    prompt_template = load_prompt_from_yaml('configs/prompt.yaml')
    prompt = prompt_template.format(context=context, query=query_text)
    logger.info(f"Generated prompt: {prompt}")

    logger.info("Generating response with GPT-3.5-turbo-instruct model.")
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=500,
        temperature=0.7,
        top_p=1.0,
        n=1
    )
    generated_text = response['choices'][0]['text'].strip()
    logger.info(f"Generated response: {generated_text}")
    return generated_text

if __name__ == "__main__":
    
    ingest_flag = False  

    PDF_DIR = "bin/finance docs"

    if ingest_flag:
        logger.info("Starting document ingestion.")
        for path in os.listdir(PDF_DIR):
            if path.endswith(".pdf"):
                doc_path = os.path.join(PDF_DIR, path)
                parsed_result = get_document_text(doc_path)
                chunked_result = get_pdf_chunks(parsed_result)
                individual_chunk = get_individual_chunk(chunked_result)

                for idx, chunk in enumerate(individual_chunk):
                    emb = get_embedding(chunk["sentence_chunk"])
                    chunk["dense_vector"] = emb

                    chunk_id = f"{doc_path.split('/')[-1]}_chunk_{idx}"
                    metadata = {
                        "chunk_id": chunk_id,
                        "page_num": chunk["page_num"],
                        "document_name": doc_path.split("/")[-1],
                    }

                    ingest_chunk_es(client, index_name, chunk["sentence_chunk"], emb, chunk_id, metadata)
                    logger.info(f"Ingested chunk {chunk_id} with metadata {metadata}")

                logger.info("****************************")
                logger.info(f"Ingested document {doc_path}")

        logger.info("Ingestion complete!")

    query_text = "Sabka Saath, Sabka Vikas"
    query_embedding = get_embedding(query_text)

    results = query_chunks_es(client, index_name, query_embedding, n_results=5)

    documents = [hit["_source"]["text_chunk"] for hit in results["hits"]["hits"]]
    response = generate_response_with_gpt(documents, query_text)

    print("GPT-3.5 Response:")
    print(response)
