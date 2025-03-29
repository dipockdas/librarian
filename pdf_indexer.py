#!/usr/bin/env python3
"""
PDF Indexer for Book Library

This script scans a directory of PDF files, processes them with LangChain,
and stores their embeddings in a Qdrant vector database for later retrieval.
"""

import os
import argparse
import logging
import concurrent.futures
from pathlib import Path
from uuid import uuid4
from tqdm import tqdm
from dotenv import load_dotenv
from time import time

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Load environment variables from .env file
load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("PDFIndexer")

# Get default embedding model from environment 
DEFAULT_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

def find_pdf_files(directory):
    """Find all PDF files in the given directory and its subdirectories."""
    pdf_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    return pdf_files

def process_pdf(file_path, text_splitter):
    """Load and split a PDF file into chunks."""
    logger.info(f"Processing {file_path}")
    try:
        loader = PyPDFLoader(file_path)
        return loader.load_and_split(text_splitter)
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return []

def setup_qdrant(client, collection_name, vector_size=None):
    """Setup Qdrant collection with proper configuration."""
    # Delete collection if it exists
    try:
        client.delete_collection(collection_name=collection_name)
        logger.info(f"Deleted existing collection: {collection_name}")
    except Exception:
        logger.info(f"Collection {collection_name} does not exist yet")
    
    # If vector_size is not provided, get a sample embedding to determine the size
    if vector_size is None:
        logger.info("Determining vector size from model...")
        # Use the current embedding model that was passed in as an argument
        model_name = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        test_embeddings = OllamaEmbeddings(model=model_name, base_url=OLLAMA_HOST)
        vector_size = len(test_embeddings.embed_documents(["Test to determine vector size"])[0])
        logger.info(f"Detected vector size: {vector_size}")
    
    # Create new collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "content": VectorParams(size=vector_size, distance=Distance.COSINE)
        }
    )
    logger.info(f"Created new collection: {collection_name} with vector size {vector_size}")

def store_document_chunks(documents, embeddings, client, collection_name, batch_size=100):
    """Store document chunks in Qdrant."""
    
    total_documents = len(documents)
    logger.info(f"Storing {total_documents} document chunks in batches of {batch_size}")
    
    # Function to embed a batch of documents
    def embed_batch(batch):
        start_time = time()
        texts = [item.page_content for item in batch]
        try:
            # Batch embed all texts at once
            vectors = embeddings.embed_documents(texts)
            
            points = []
            for i, (item, vector) in enumerate(zip(batch, vectors)):
                id = str(uuid4())
                content = item.page_content
                source = item.metadata.get("source", "unknown")
                page = item.metadata.get("page", 0)
                
                vector_dict = {"content": vector}
                
                payload = {
                    "page_content": content,
                    "metadata": {
                        "id": id,
                        "page_content": content,
                        "source": source,
                        "page": page,
                    }
                }
                
                points.append(PointStruct(id=id, vector=vector_dict, payload=payload))
            
            elapsed = time() - start_time
            return points, elapsed
        except Exception as e:
            logger.error(f"Error embedding batch: {str(e)}")
            return [], 0
    
    # Process in batches
    total_batches = (total_documents + batch_size - 1) // batch_size
    total_time = 0
    processed = 0
    
    for i in range(0, total_documents, batch_size):
        batch_start = time()
        end_idx = min(i + batch_size, total_documents)
        batch = documents[i:end_idx]
        
        # Process this batch
        points, embed_time = embed_batch(batch)
        total_time += embed_time
        
        if points:
            try:
                client.upsert(
                    collection_name=collection_name,
                    wait=True,
                    points=points
                )
                processed += len(points)
            except Exception as e:
                logger.error(f"Error upserting points: {str(e)}")
        
        batch_time = time() - batch_start
        logger.info(f"Batch {i//batch_size + 1}/{total_batches}: Processed {len(points)} chunks " +
                   f"(Embedding: {embed_time:.2f}s, Total: {batch_time:.2f}s)")
    
    avg_time = total_time / total_batches if total_batches > 0 else 0
    logger.info(f"Successfully processed {processed}/{total_documents} chunks")
    logger.info(f"Average embedding time per batch: {avg_time:.2f}s")

def main():
    parser = argparse.ArgumentParser(description="Index PDF books into Qdrant vector database")
    parser.add_argument("--books-dir", type=str, default="./Books", help="Directory containing PDF books")
    parser.add_argument("--collection", type=str, default="books", help="Qdrant collection name")
    parser.add_argument("--chunk-size", type=int, default=512, help="Text chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Text chunk overlap")
    parser.add_argument("--batch-size", type=int, default=os.getenv("BATCH_SIZE", "200"), help="Batch size for Qdrant uploads")
    parser.add_argument("--qdrant-host", type=str, default=os.getenv("QDRANT_HOST", "localhost"), help="Qdrant server host")
    parser.add_argument("--qdrant-port", type=int, default=int(os.getenv("QDRANT_PORT", "6333")), help="Qdrant server port")
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL, help="Ollama model to use for embeddings")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of PDFs to process (for testing)")
    
    args = parser.parse_args()
    
    # Initialize components
    logger.info(f"Initializing embedding model: {args.embedding_model}")
    embeddings = OllamaEmbeddings(model=args.embedding_model, base_url=OLLAMA_HOST)
    
    logger.info(f"Connecting to Qdrant at {args.qdrant_host}:{args.qdrant_port}")
    client = QdrantClient(args.qdrant_host, port=args.qdrant_port)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Find all PDF files
    logger.info(f"Scanning for PDF files in {args.books_dir}")
    pdf_files = find_pdf_files(args.books_dir)
    
    if args.limit:
        logger.info(f"Limiting to first {args.limit} PDF files")
        pdf_files = pdf_files[:args.limit]
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    # Process PDFs in parallel
    from concurrent.futures import ThreadPoolExecutor
    import multiprocessing
    
    # Use at most 4 workers or number of CPUs, whichever is smaller
    max_workers = min(8, multiprocessing.cpu_count())
    logger.info(f"Processing PDFs with {max_workers} parallel workers")
    
    all_documents = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all PDF processing tasks
        future_to_pdf = {
            executor.submit(process_pdf, pdf_file, text_splitter): pdf_file 
            for pdf_file in pdf_files
        }
        
        # Process results as they complete
        for future in tqdm(
            concurrent.futures.as_completed(future_to_pdf), 
            total=len(pdf_files),
            desc="Processing PDFs"
        ):
            pdf_file = future_to_pdf[future]
            try:
                documents = future.result()
                all_documents.extend(documents)
                logger.info(f"Extracted {len(documents)} chunks from {pdf_file}")
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {str(e)}")
    
    logger.info(f"Total document chunks: {len(all_documents)}")
    
    # Store documents in Qdrant
    store_document_chunks(
        all_documents, 
        embeddings, 
        client, 
        args.collection,
        batch_size=args.batch_size
    )
    
    # Print collection info
    collection_info = client.get_collection(args.collection)
    logger.info(f"Collection '{args.collection}' now has {collection_info.points_count} points")
    logger.info("Indexing complete!")

if __name__ == "__main__":
    main()