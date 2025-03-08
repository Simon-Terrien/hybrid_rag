# RAG System Documentation

## Overview

This document provides comprehensive documentation for the enhanced RAG (Retrieval-Augmented Generation) system with Docling and Ollama integration. The system is designed to process various document formats, extract and index content, and enable natural language queries with accurate responses based on the document content.

## System Architecture

The RAG system consists of several key components:

1. **Document Processing** - Extracts, cleans, and chunks document content
2. **Vector Indexing** - Creates searchable vector representations of document chunks 
3. **Search Engine** - Retrieves relevant documents based on queries
4. **Document Chat** - Provides conversational interface for questions and answers
5. **Orchestrator** - Coordinates all system components

The system also includes optional enhancements:
- **Docling Integration** - Improves document processing with advanced layout understanding
- **Ollama Integration** - Provides local LLM capabilities for generating responses

## Component Details

### 1. Document Processing

The document processing pipeline handles the extraction and preparation of content from various file formats:

#### EnhancedDocumentProcessor

```python
from enhanced_document_processor import EnhancedDocumentProcessor

processor = EnhancedDocumentProcessor(
    data_dir=DATA_DIR,
    processed_dir=PROCESSED_DIR
)

# Process a single document
result = processor.process_document(Path("data/document.pdf"))

# Process all documents in a directory
results = processor.process_directory()
```

**Key Features:**
- Text extraction from multiple document formats (PDF, DOCX, TXT)
- Text cleaning and normalization
- Metadata extraction and enrichment
- Semantic chunking for improved context preservation
- Document registry management

#### DoclingProcessor (Optional)

If Docling is available, the system can use its advanced document processing capabilities:

```python
from docling_processor import DoclingProcessor

processor = DoclingProcessor(
    data_dir=DATA_DIR,
    processed_dir=PROCESSED_DIR,
    enable_enrichments=True
)

# Process a document with Docling
result = processor.process_document(Path("data/complex_document.pdf"))
```

**Key Features:**
- Layout analysis for complex documents
- Table structure recognition
- Code and formula understanding
- Image classification and description
- Rich document structure preservation

### 2. Vector Indexing

The vector indexing component creates searchable representations of document chunks:

```python
from vector_indexer import VectorIndexer

indexer = VectorIndexer(
    processed_dir=PROCESSED_DIR,
    vector_db_dir=VECTOR_DB_DIR,
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    vector_db_type="faiss"
)

# Index all processed documents
result = indexer.index_all_documents()

# Get indexing statistics
stats = indexer.get_stats()
```

**Key Features:**
- Support for multiple embedding models
- Support for FAISS and Chroma vector databases
- Efficient batch processing
- Document registry synchronization
- Automatic embeddings computation

### 3. Search Engine

The search engine provides document retrieval capabilities:

#### SemanticSearchEngine

```python
from semantic_search import SemanticSearchEngine

search_engine = SemanticSearchEngine(
    vector_indexer=indexer,
    top_k=5,
    similarity_threshold=0.7
)

# Search documents
results = search_engine.search("What is hybrid search?")
```

**Key Features:**
- Semantic similarity search
- Metadata filtering
- Document grouping
- Source attribution

#### HybridSearchEngine

```python
from hybrid_search import HybridSearchEngine

hybrid_search = HybridSearchEngine(
    semantic_search_engine=search_engine,
    similarity_threshold=0.7,
    semantic_weight=0.7
)

# Search using both semantic and lexical matching
results = hybrid_search.search(
    "What is hybrid search?",
    semantic_weight=0.6  # Adjust the balance between semantic and lexical search
)
```

**Key Features:**
- Combined semantic and lexical search
- BM25 and TF-IDF lexical matching
- Adjustable weighting between search methods
- Improved recall for rare terms and keywords

#### ContextualReranker (Optional)

```python
from contextual_reranker import ContextualReranker

reranker = ContextualReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    enabled=True
)

# Rerank search results for better precision
reranked_results = reranker.rerank(
    query="What is hybrid search?",
    results=search_results["results"],
    top_k=5
)
```

**Key Features:**
- Cross-encoder based reranking
- Improved precision in search results
- Optional activation for performance optimization

### 4. Document Chat

The document chat component provides a conversational interface:

```python
from DocumentChat import DocumentChat

document_chat = DocumentChat(
    search_engine=hybrid_search,
    use_ollama=True,
    use_langchain=True
)

# Create a new conversation
conversation_id = document_chat.new_conversation()

# Ask a question
response = document_chat.ask("What is hybrid search?")

# Get conversation history
conversation = document_chat.get_conversation(conversation_id)
```

**Key Features:**
- Conversation management
- Question-answering capabilities
- Source attribution
- Conversation history
- Ollama integration for LLM responses

### 5. RAG Orchestrator

The orchestrator coordinates all system components:

#### RAGPipelineOrchestrator (Base)

```python
from RAGPipelineOrchestrator import RAGPipelineOrchestrator

orchestrator = RAGPipelineOrchestrator()

# Process documents
orchestrator.process_documents(subdirectory="papers")

# Index documents
orchestrator.index_documents()

# Search documents
results = orchestrator.search("What is hybrid search?")

# Ask questions
response = orchestrator.ask("How does hybrid search work?")
```

**Key Features:**
- Component management and coordination
- Asynchronous task processing
- System state tracking
- Configuration management

#### DoclingEnhancedRAGOrchestrator (Enhanced)

```python
from docling_rag_orchestrator import DoclingEnhancedRAGOrchestrator

orchestrator = DoclingEnhancedRAGOrchestrator(
    use_docling=True,
    use_langchain=True,
    use_hybrid_search=True,
    use_reranker=True
)

# Process and index in one operation
result = orchestrator.process_and_index(
    source_paths=["documents/report.pdf"],
    force_reprocess=True
)
```

**Key Features:**
- All features of base orchestrator
- Docling integration
- LangChain integration
- End-to-end processing and querying

## Configuration

The configuration system is highly flexible and customizable:

```python
from config import get_config, save_config

# Get current configuration
config = get_config()

# Update configuration
config["embedding"]["model"] = "all-mpnet-base-v2"
save_config(config)
```

Key configuration sections:
- **paths** - File system paths for data, processing, and indexing
- **embedding** - Embedding model selection and settings
- **vector_db** - Vector database settings
- **chunking** - Document chunking parameters
- **search** - Search and retrieval parameters
- **llm** - Language model settings
- **docling_integration** - Docling-specific settings
- **ollama_integration** - Ollama-specific settings

## Command-Line Interface

The system provides a comprehensive command-line interface:

```bash
# Initialize the system
python main_rag.py init

# Process documents
python main_rag.py process --dir documents

# Index documents
python main_rag.py index

# Search documents
python main_rag.py search "hybrid search" --top-k 5 --type hybrid

# Ask questions
python main_rag.py ask "How does hybrid search work?"

# Check system status
python main_rag.py status

# Update configuration
python main_rag.py config update --docling True --reranker True
```

## Workflows

### Document Processing Workflow

1. **Document Loading**: Load document from file system or URL
2. **Text Extraction**: Extract raw text from the document
3. **Text Cleaning**: Normalize and clean the extracted text
4. **Metadata Extraction**: Extract and enhance document metadata
5. **Chunking**: Split the document into semantically meaningful chunks
6. **Storage**: Store chunks and metadata in the processed directory
7. **Registry Update**: Update the document registry with new document information

### Search and Retrieval Workflow

1. **Query Analysis**: Analyze the user's query
2. **Semantic Search**: Convert query to vector and find similar document chunks
3. **Lexical Search** (optional): Perform keyword-based search using BM25/TF-IDF
4. **Result Combination**: Combine semantic and lexical results (for hybrid search)
5. **Reranking** (optional): Rerank results using a cross-encoder for better precision
6. **Result Formatting**: Group by document and format results for presentation

### Question Answering Workflow

1. **Query Processing**: Process the user's question
2. **Document Retrieval**: Retrieve relevant document chunks
3. **Context Building**: Build a context from retrieved chunks
4. **LLM Generation**: Generate an answer using an LLM (Ollama or other)
5. **Source Attribution**: Identify and attribute sources used in the answer
6. **Response Formatting**: Format the response with the answer and sources
7. **Conversation Update**: Update conversation history

## Integration with Docling

Docling provides advanced document understanding capabilities:

1. **Layout Analysis**: Understands document structure and layout
2. **Table Extraction**: Accurately extracts and preserves table structures
3. **Formula Recognition**: Understands mathematical formulas
4. **Code Understanding**: Preserves code blocks with syntax
5. **Image Analysis**: Classifies and describes images in documents

Integration is optional and can be enabled/disabled through configuration.

## Integration with Ollama

Ollama provides local LLM capabilities:

1. **Local Inference**: Run LLMs locally without external API calls
2. **Model Selection**: Choose from various available models
3. **Parameter Control**: Adjust temperature, top_p, and other generation parameters
4. **Streaming**: Stream responses token by token for better UX

Integration is optional and can be enabled/disabled through configuration.

## Error Handling and Fallbacks

The system is designed with robust error handling and fallback mechanisms:

1. **Component Initialization**: If a component fails to initialize, the system logs the error and falls back to simpler alternatives
2. **Optional Components**: Enhanced components like Docling and Ollama are optional and the system works without them
3. **Method Inspection**: The system checks for method availability before calling them
4. **Exception Logging**: All exceptions are logged for debugging
5. **Graceful Degradation**: When LLM is unavailable, the system provides simpler responses based on document retrieval

## System Requirements

- Python 3.8+
- PyTorch (for embeddings and LLMs)
- sentence-transformers
- FAISS or Chroma for vector storage
- Docker (optional, for Ollama)

### Optional Dependencies:
- Docling and langchain-docling for enhanced document processing
- Ollama for local LLM inference
- CrossEncoders for reranking

## Performance Considerations

- **GPU Acceleration**: Use GPU for embedding generation when available
- **Batch Processing**: Process documents and embeddings in batches
- **Caching**: Enable embedding caching to avoid redundant computation
- **Chunking Strategy**: Adjust chunking parameters based on document types
- **LLM Selection**: Choose appropriate LLM models based on performance requirements

## Troubleshooting

Common issues and solutions:

1. **Import Errors**: Use the feature flags in main_rag.py to disable problematic components
2. **Memory Issues**: Reduce batch sizes and use smaller embedding models
3. **Performance Issues**: Enable caching, use faster models, disable reranking for faster results
4. **Integration Issues**: Check availability of required packages and correct API endpoints

## Conclusion

This RAG system provides a comprehensive solution for document processing, indexing, searching, and question answering. The modular design allows for flexible deployment and customization, while the integration with Docling and Ollama enhances the capabilities for complex documents and local inference.