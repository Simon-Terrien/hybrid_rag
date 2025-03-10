# Quick Start Guide

This guide will help you get up and running with the Docling RAG system in just a few minutes.

## Installation

### 1. Set up environment

```bash
# Clone the repository
git clone https://github.com/yourusername/docling-rag.git
cd docling-rag

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare your documents

Place your documents in the `data` directory:

```bash
mkdir -p data/mydocs
cp /path/to/your/documents/*.pdf data/mydocs/
```

The system supports PDF, DOCX, TXT, HTML, and Markdown files.

## Basic Usage

### 1. Process documents

```bash
# Process all documents in the data directory
python main.py process

# Or process a specific directory
python main.py process --dir mydocs

# Or process a single file
python main.py process --file data/mydocs/document.pdf
```

### 2. Index documents

```bash
# Index all processed documents
python main.py index
```

### 3. Search for information

```bash
# Basic search
python main.py search "your search query"

# Advanced search with options
python main.py search "your search query" --top-k 10 --type hybrid
```

### 4. Ask questions

```bash
# Ask a question about your documents
python main.py ask "What are the key features of the product?"

# Get more context for your question
python main.py ask "What are the key features of the product?" --top-k 10
```

### 5. View system status

```bash
# Check the system status
python main.py system --status
```

## Using the API

### 1. Start the API server

```bash
# Start the server
python server.py

# Or specify host and port
python server.py --host 127.0.0.1 --port 8080
```

### 2. Access the API

The API documentation is available at http://localhost:8000/docs when the server is running.

### 3. Example API requests

#### Process a document

```bash
curl -X POST "http://localhost:8000/documents/process" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/document.pdf" \
  -F "force_reprocess=false"
```

#### Search for documents

```bash
curl -X POST "http://localhost:8000/search" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "your search query",
    "top_k": 5,
    "search_type": "hybrid",
    "rerank_results": true
  }'
```

#### Ask a question

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key features?",
    "top_k": 5,
    "search_type": "hybrid"
  }'
```

## Troubleshooting

### Common Issues

1. **No documents found**: Make sure your documents are in the `data` directory and have been processed.
2. **Search returns no results**: Try using a more general query, increasing `top_k`, or lowering the similarity threshold in `config.json`.
3. **Server won't start**: Check if port 8000 is already in use and try a different port with `--port`.

### Registry Repair

If you encounter issues with document or vector registries:

```bash
# Check registry status
python registry_check.py --check

# Repair registry
python registry_check.py --repair
```

## Next Steps

- [Read the full documentation](README.md)
- [Learn about configuration](CONFIGURATION.md)
- [Contribute to the project](CONTRIBUTING.md)