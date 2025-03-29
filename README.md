# Fully local AI powered librarian

A system for indexing PDF books and creating a searchable knowledge base using vector embeddings and semantic search.

## Overview

This project provides tools to:

1. Initialize a vector database collection
2. Index PDF books and store their content as vector embeddings
3. Search the indexed books using natural language queries

The system uses:
- Ollama for embedding generation and LLM responses
- Qdrant as the vector database for storing and searching embeddings
- LangChain for document processing and chain orchestration

## Prerequisites

- Python 3.8+
- Ollama running locally or remotely (for embeddings and LLM)
- Qdrant running locally or remotely (for vector storage)

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start Qdrant server using Docker:
```bash
docker run -d -p 6333:6333 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant
```

4. Install Ollama and download needed models:
```bash
# Install Ollama from https://ollama.com/
ollama pull nomic-embed-text  # For embeddings
ollama pull mistral           # For LLM responses
```

5. Create a `.env` file with configuration (optional):

```
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_LLM_MODEL=mistral
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

## Usage

### 1. Initialize a Collection

Before indexing books, create a Qdrant collection:

```bash
python init_collection.py books
```

Options:
- `--embedding-model`: Specify the Ollama model to use for embeddings

### 2. Index PDF Books

Index PDF files from a directory:
1. Initialize components (embedding model, Qdrant client, text splitter)
2. Scan directories for PDF files
3. Process PDFs in parallel using a thread pool
4. For each PDF, load and split into text chunks
5. Collect all chunks from all PDFs
6. Generate embeddings in batches
7. Store the embeddings with metadata in Qdrant

```bash
python pdf_indexer.py --books-dir ./Books
```

Options:
- `--books-dir`: Directory containing PDF books (default: ./Books)
- `--collection`: Qdrant collection name (default: books)
- `--chunk-size`: Text chunk size (default: 512)
- `--chunk-overlap`: Text chunk overlap (default: 100)
- `--batch-size`: Batch size for Qdrant uploads
- `--embedding-model`: Ollama model to use for embeddings
- `--limit`: Limit number of PDFs to process (for testing)

### 3. Query the Knowledge Base

Ask questions about your indexed books:

```bash
python query_books.py --query "What does the book say about machine learning?"
```

For interactive mode:

```bash
python query_books.py --interactive
```

Options:
- `--collection`: Qdrant collection name (default: books)
- `--embedding-model`: Ollama model for embeddings
- `--llm-model`: Ollama model for generating answers
- `--top-k`: Number of documents to retrieve (default: 5)
- `--query`: Question to ask (if not provided, will prompt)
- `--interactive`: Run in interactive mode

## Environment Variables

The system supports configuration via environment variables:

- `OLLAMA_HOST`: URL of Ollama server
- `OLLAMA_EMBEDDING_MODEL`: Default embedding model
- `OLLAMA_LLM_MODEL`: Default LLM model
- `QDRANT_HOST`: Qdrant server host
- `QDRANT_PORT`: Qdrant server port
- `BATCH_SIZE`: Default batch size for uploads
- `TOP_K`: Default number of documents to retrieve

## Example Workflow

```bash
# Initialize collection
python init_collection.py books

# Index PDFs from the Books directory (start with a few for testing)
python pdf_indexer.py --books-dir ./Books --limit 5

# Ask a question
python query_books.py --query "What does Design Patterns book say about Factory Pattern?"

# Or use interactive mode
python query_books.py --interactive
```

## Notes
- For larger book collections, increase the batch size and consider using a more powerful machine
- You can experiment with different chunk sizes to optimize retrieval
- The quality of answers depends on the Ollama LLM model you use
