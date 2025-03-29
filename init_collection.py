#!/usr/bin/env python3
"""
Initialize a Qdrant collection with correct vector settings.
"""

import os
import argparse
import logging
from dotenv import load_dotenv

from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Load environment variables
load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("InitCollection")

def main():
    parser = argparse.ArgumentParser(description="Initialize Qdrant collection")
    parser.add_argument("collection", type=str, help="Name of the collection to initialize")
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL, help="Embedding model to determine vector size")
    args = parser.parse_args()

    logger.info(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)

    logger.info(f"Using embedding model: {args.embedding_model}")
    embeddings = OllamaEmbeddings(model=args.embedding_model, base_url=OLLAMA_HOST)
    vector_size = len(embeddings.embed_documents(["Sample text for vector sizing"])[0])
    logger.info(f"Detected vector size: {vector_size}")

    try:
        client.delete_collection(collection_name=args.collection)
        logger.info(f"Deleted existing collection: {args.collection}")
    except Exception:
        logger.info(f"No existing collection named {args.collection} to delete")

    client.create_collection(
        collection_name=args.collection,
        vectors_config={
            "content": VectorParams(size=vector_size, distance=Distance.COSINE)
        }
    )
    logger.info(f"Created new collection: {args.collection} with vector size {vector_size}")

if __name__ == "__main__":
    main()