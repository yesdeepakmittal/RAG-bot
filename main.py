import os
import logging
import chromadb
import openai
import chromadb.utils.embedding_functions as embedding_functions

from src.chunking.chunking import get_individual_chunk, get_pdf_chunks
from src.embedding.embedder import get_embedding
from src.parsing.parser import get_document_text
from src.indexing.ingestion import ingest_chunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

chroma_client = chromadb.Client()

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"], model_name="text-embedding-ada-002"
)

collection = chroma_client.get_or_create_collection(
    name="document_collection", embedding_function=openai_ef
)

openai.api_key = os.environ["OPENAI_API_KEY"]

def query_chromadb(query_text, n_results=5):
    """
    Query ChromaDB and return relevant documents.

    Parameters
    ----------
    query_text : str
        The text to query the database with.
    n_results : int, optional
        The number of results to return (default is 5).

    Returns
    -------
    dict
        A dictionary containing the query results.
    """
    logger.info(f"Querying ChromaDB with text: {query_text}")
    results = collection.query(query_texts=[query_text], n_results=n_results)
    logger.info(f"Query results: {results}")
    return results

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
    prompt = f"Use the following documents to answer the query:\n\n{context}\n\nQuery: {query_text}\nAnswer:"

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

                    ingest_chunk(chunk["sentence_chunk"], emb, chunk_id, metadata)
                    logger.info(f"Ingested chunk {chunk_id} with metadata {metadata}")

                logger.info("****************************")
                logger.info(f"Ingested document {doc_path}")

        logger.info("Ingestion complete!")

    
    query_text = "Sabka Saath, Sabka Vikas"
    results = query_chromadb(query_text, n_results=5)

    documents = [" ".join(result) for result in results['documents']]
    
    response = generate_response_with_gpt(documents, query_text)
    
    print("GPT-3.5 Response:")
    print(response)
