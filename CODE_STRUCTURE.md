# Code Structure Documentation

This document provides an overview of the Docling RAG system's code structure, component relationships, and architecture.

## Architecture Overview

The Docling RAG system follows a layered architecture pattern:

1. **Interface Layer**: CLI and API interfaces (`main.py`, `fastapi_app.py`, `server.py`)
2. **Orchestration Layer**: System coordination (`rag_orchestrator_api.py`, `docling_rag_orchestrator.py`)
3. **Processing Layer**: Document processing and search (`docling_processor.py`, `vector_indexer.py`, etc.)
4. **Storage Layer**: Document and vector storage management

```
User Interfaces (CLI, API)
       ↓    ↑
   Orchestrators
       ↓    ↑
Component Services/Processors
       ↓    ↑
  Data Management
```

## Directory Structure

```
docling-rag/
├── config.py                  # Configuration settings
├── config.json                # JSON configuration file
├── main.py                    # Command-line interface
├── fastapi_app.py             # FastAPI application
├── server.py                  # Standalone server script
├── rag_orchestrator_api.py    # API orchestrator
├── doclingroc/                # Document processing components
│   ├── __init__.py
│   ├── docling_processor.py   # Document processing with Docling
│   └── docling_rag_orchestrator.py  # Main RAG orchestrator
├── vectorproc/                # Vector processing components
│   ├── __init__.py
│   ├── vector_indexer.py      # Vector database management
│   └── semantic_search.py     # Semantic search implementation
├── searchproc/                # Search processing components
│   ├── __init__.py
│   └── hybrid_search.py       # Hybrid search implementation
├── utils/                     # Utility functions
│   ├── __init__.py
│   ├── document_registry_fix.py  # Document registry repair utility
│   └── common.py              # Common utility functions
├── data/                      # Document storage
├── processed_data/            # Processed document storage
│   ├── chunks/                # Chunked documents
│   └── docling/               # Docling output
└── vector_db/                 # Vector database storage
```

## Core Components

### Interface Components

#### Main CLI (`main.py`)
- **Purpose**: Provides command-line interface
- **Functions**: Process, index, search, ask, system status
- **Relationships**: Uses `RAGOrchestratorAPI`

#### FastAPI App (`fastapi_app.py` and `server.py`)
- **Purpose**: Provides REST API
- **Endpoints**: Document processing, indexing, search, QA
- **Relationships**: Uses `RAGOrchestratorAPI`

### Orchestrators

#### RAG Orchestrator API (`rag_orchestrator_api.py`)
- **Purpose**: High-level API for the RAG system
- **Functions**: API-friendly methods for document processing, search, QA
- **Relationships**: Uses `DoclingEnhancedRAGOrchestrator`

#### Docling RAG Orchestrator (`docling_rag_orchestrator.py`)
- **Purpose**: Core RAG system orchestration
- **Functions**: Document processing, indexing, search, QA
- **Relationships**: Uses processors, indexers, and search engines

### Processing Components

#### Docling Processor (`docling_processor.py`)
- **Purpose**: Document processing and chunking
- **Functions**: Process documents, chunk text, clean text
- **Relationships**: Used by `DoclingEnhancedRAGOrchestrator`

#### Vector Indexer (`vector_indexer.py`)
- **Purpose**: Vector database management
- **Functions**: Index documents, manage vector registry
- **Relationships**: Used by `DoclingEnhancedRAGOrchestrator`

#### Semantic Search (`semantic_search.py`)
- **Purpose**: Semantic search implementation
- **Functions**: Search for documents by meaning
- **Relationships**: Uses `VectorIndexer`

#### Hybrid Search (`hybrid_search.py`)
- **Purpose**: Combine semantic and lexical search
- **Functions**: Hybrid search, reranking
- **Relationships**: Uses `SemanticSearchEngine`

#### Contextual Reranker (`contextual_reranker.py`)
- **Purpose**: Rerank search results based on contextual relevance
- **Functions**: Rerank search results
- **Relationships**: Used by search engines

### Utilities

#### Configuration (`config.py`)
- **Purpose**: System configuration
- **Functions**: Load and manage configuration settings
- **Relationships**: Used by all components

#### Utilities (`utils.py`)
- **Purpose**: Common utility functions
- **Functions**: File operations, timing, error handling
- **Relationships**: Used by all components

## Component Relationships

### Data Flow

1. **Document Processing Flow**:
   ```
   DoclingProcessor → Document Registry → Chunks
   ```

2. **Indexing Flow**:
   ```
   Chunks → VectorIndexer → Vector Database → Vector Registry
   ```

3. **Search Flow**:
   ```
   Query → SemanticSearch/HybridSearch → Results → Reranker → Ranked Results
   ```

4. **QA Flow**:
   ```
   Question → Search → Context Retrieval → LLM → Answer
   ```

### Dependency Graph

```
FastAPI App / CLI
    ↓
RAGOrchestratorAPI
    ↓
DoclingEnhancedRAGOrchestrator
    ↓
┌───────┬─────────┬──────────┐
↓       ↓         ↓          ↓
DoclingProcessor  VectorIndexer  SemanticSearch  HybridSearch
                      ↑              ↑               ↑
                      └──────────────┘               │
                                                     │
                                      ContextualReranker
```

## Key Classes and Methods

### DoclingProcessor

Key methods:
- `process_document(file_path)`: Process a single document
- `process_directory(directory)`: Process all documents in a directory

### VectorIndexer

Key methods:
- `index_document(document_id)`: Index a specific document
- `index_all_documents()`: Index all processed documents
- `get_stats()`: Get vector database statistics

### SemanticSearchEngine

Key methods:
- `search(query, top_k)`: Perform semantic search
- `get_document_context(document_id)`: Get context for a document

### HybridSearchEngine

Key methods:
- `search(query, top_k)`: Perform hybrid search
- `_combine_search_results()`: Combine semantic and lexical results

### DoclingEnhancedRAGOrchestrator

Key methods:
- `process_documents(source_paths)`: Process documents
- `index_documents(force_reindex)`: Index documents
- `search(query, top_k)`: Search for documents
- `ask(question, top_k)`: Answer questions using document context

### RAGOrchestratorAPI

Key methods:
- `process_document(file_path)`: Process a document (async)
- `index_documents(force_reindex)`: Index documents (async)
- `search(query, top_k)`: Search for documents (async)
- `ask(question, top_k)`: Answer questions (async)

## Configuration

The system is configured through multiple files:

1. **config.py**: Core configuration implementation
2. **config.json**: User-editable configuration
3. **.env**: Environment-specific settings

Key configuration sections:
- **paths**: Data and storage directories
- **embedding**: Model settings
- **vector_db**: Vector database settings
- **chunking**: Document chunking parameters
- **search**: Search parameters
- **llm**: Language model settings

## Storage Structure

### Document Registry (`document_registry.json`)
- Maps document paths to metadata
- Tracks document IDs, hashes, and chunk files

### Vector Registry (`vector_registry.json`)
- Tracks vector database metadata
- Records indexed documents and chunk counts

### Document Storage
- **data/**: Raw documents
- **processed_data/chunks/**: Processed document chunks
- **processed_data/docling/**: Docling-specific output

### Vector Storage
- **vector_db/**: Vector database files
- **vector_db/index.faiss**: FAISS index (if using FAISS)
- **vector_db/chroma/**: Chroma database (if using Chroma)

## Extension Points

The system is designed to be extended in several ways:

1. **New Document Processors**: Add new processors in `doclingroc/`
2. **New Search Methods**: Add new search implementations in `searchproc/`
3. **New Vector Stores**: Add support for additional vector databases in `vectorproc/`
4. **New LLM Integrations**: Add support for different LLMs in the orchestrator

To add extensions, follow the existing patterns and interfaces, then update the orchestrator to use your new components.