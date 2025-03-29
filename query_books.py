#!/usr/bin/env python3
"""
Book Library Query Tool

This script allows querying the Qdrant vector database containing book knowledge
using natural language questions.
"""

import os
import argparse
import logging
from dotenv import load_dotenv

from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_qdrant import Qdrant
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from qdrant_client import QdrantClient

# Load environment variables from .env file
load_dotenv()
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("BookQuery")

# Get default models from environment
DEFAULT_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
DEFAULT_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "mistral")
print("Using ${DEFAULT_LLM_MODEL} for LLM")

def main():
    parser = argparse.ArgumentParser(description="Query book knowledge from Qdrant vector database")
    parser.add_argument("--collection", type=str, default=os.getenv("QDRANT_COLLECTION", "books"), help="Qdrant collection name")
    parser.add_argument("--qdrant-host", type=str, default=os.getenv("QDRANT_HOST", "localhost"), help="Qdrant server host")
    parser.add_argument("--qdrant-port", type=int, default=int(os.getenv("QDRANT_PORT", "6333")), help="Qdrant server port")
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL, 
                       help="Ollama model for embeddings (should match what was used during indexing)")
    parser.add_argument("--llm-model", type=str, default=DEFAULT_LLM_MODEL, 
                       help="Ollama model for generating answers")
    parser.add_argument("--top-k", type=int, default=int(os.getenv("TOP_K", "5")), help="Number of documents to retrieve")
    parser.add_argument("--query", type=str, help="Question to ask (if not provided, will prompt)")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Initialize Ollama for embeddings and LLM
    logger.info(f"Initializing embedding model: {args.embedding_model}")
    logger.info(f"Initializing LLM model: {args.llm_model}")
    
    embeddings = OllamaEmbeddings(model=args.embedding_model, base_url=OLLAMA_HOST)
    llm = OllamaLLM(model=args.llm_model, base_url=OLLAMA_HOST)
    
    # Connect to Qdrant
    logger.info(f"Connecting to Qdrant at {args.qdrant_host}:{args.qdrant_port}")
    client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
    
    # Access the vector store
    logger.info(f"Accessing collection: {args.collection}")
    try:
        # Get collection info to verify it exists
        collection_info = client.get_collection(args.collection)
        logger.info(f"Collection has {collection_info.points_count} points")
        
        vectorstore = Qdrant(
            client=client,
            collection_name=args.collection,
            embeddings=embeddings,
            vector_name="content"
        )
    except Exception as e:
        logger.error(f"Error accessing collection: {str(e)}")
        return
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": args.top_k})
    
    # Create prompt template
    template = """
    You are a knowledgeable assistant who has read many books.
    Use the following information from books to answer the user's question.
    If you don't know the answer based on the provided context, just say that you don't know.
    Always cite the source of your information (book title) when possible.

    Question: {input}
    
    Context:
    {context}
    
    Answer:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create the chain
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    # Handle query
    if args.interactive:
        logger.info("Starting interactive mode. Type 'exit' to quit.")
        while True:
            query = input("\nEnter your question: ")
            if query.lower() in ['exit', 'quit', 'q']:
                break
                
            if not query.strip():
                continue
                
            try:
                print("\nSearching for relevant information...")
                result = retrieval_chain.invoke({"input": query})
                print("\nAnswer:")
                print(result["answer"])
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
    else:
        query = args.query or input("Enter your question: ")
        
        try:
            print("Searching for relevant information...")
            result = retrieval_chain.invoke({"input": query})
            print("\nAnswer:")
            print(result["answer"])
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()