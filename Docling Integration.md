# Docling Integration Guide for Enhanced RAG System

This guide provides instructions for integrating Docling with your enhanced RAG system to significantly improve document processing and information retrieval.

## What is Docling?

Docling is a powerful document conversion tool that parses PDF, DOCX, HTML, and other document formats into a rich, unified representation that preserves document layout, tables, formula, code, and other structured elements. It's designed to make documents ready for generative AI workflows like Retrieval-Augmented Generation (RAG).

## Benefits of Integrating Docling

1. **Advanced Document Parsing**: Docling provides superior parsing of complex document formats, preserving layout, tables, and hierarchical structure.

2. **Rich Metadata Extraction**: Extracts detailed metadata including document structure, headers, and page information.

3. **Smart Chunking Options**: Built-in chunkers like `HybridChunker` and `HierarchicalChunker` offer advanced document segmentation that respects document structure.

4. **Content Enrichment**: Special handling for code, formulas, images, and tables improves the quality of extracted information.

5. **LangChain Integration**: Seamless integration with LangChain through the `DoclingLoader` component.

## Implementation Prerequisites

Before implementing Docling integration, ensure you have:

1. Installed the required packages:
   ```bash
   pip install docling langchain-docling
   ```

2. Understood your document types and processing needs.

3. Set up your enhanced RAG system as described in the main implementation guide.

## Integration Components

We've created three key components to integrate Docling with your existing RAG system:

1. **DoclingProcessor**: A wrapper for Docling's document processing capabilities that fits into your existing pipeline.

2. **LangChainRAGAdapter**: Integrates LangChain with Docling and your existing RAG components.

3. **DoclingEnhancedRAGOrchestrator**: A unified orchestrator that gives you the flexibility to use Docling, LangChain, or your existing RAG components as needed.

## Implementation Steps

### 1. Add the Docling Processor

The `DoclingProcessor` handles document conversion using Docling:

```python
from docling_processor import DoclingProcessor

# Initialize the processor
processor = DoclingProcessor(
    data_dir=DATA_DIR,
    processed_dir=PROCESSED_DIR,
    enable_enrichments=True  # Enable code, formula, and image enrichments
)

# Process a document
result = processor.process_document("path/to/document.pdf")
print(f"Generated {result['chunk_count']} chunks")

# Process all documents in a directory
results = processor.process_directory()
```

### 2. Integrate with LangChain

The `LangChainRAGAdapter` connects Docling with LangChain components:

```python
from langchain_integration import LangChainRAGAdapter

# Initialize the adapter
adapter = LangChainRAGAdapter(
    data_dir=DATA_DIR,
    processed_dir=PROCESSED_DIR,
    vector_db_dir=VECTOR_DB_DIR,
    use_docling=True
)

# Process and index documents
adapter.process_and_index_documents(
    source_paths=["path/to/document.pdf", "path/to/another.docx"],
    vector_store_id="my_documents"
)

# Ask a question
answer = adapter.ask(
    question="What is the main topic of the document?",
    top_k=3
)
print(answer["answer"])
```

### 3. Use the Enhanced Orchestrator

The `DoclingEnhancedRAGOrchestrator` provides a unified interface to all components:

```python
from docling_rag_orchestrator import DoclingEnhancedRAGOrchestrator

# Initialize the orchestrator
orchestrator = DoclingEnhancedRAGOrchestrator(
    use_docling=True,        # Use Docling for document processing
    use_langchain=False,     # Don't use LangChain for retrieval/QA
    use_hybrid_search=True,  # Use your hybrid search for retrieval
    use_reranker=True        # Use your contextual reranker
)

# Process documents with Docling
orchestrator.process_documents(
    source_paths=["path/to/document.pdf"],
    use_docling=True
)

# Index with your vector indexer
orchestrator.index_documents(
    use_langchain=False
)

# Search using your hybrid search
results = orchestrator.search(
    query="important concepts",
    search_type="hybrid", 
    use_langchain=False
)
```

### Using Docling's Advanced Chunkers

Docling provides specialized chunkers that respect document structure:

```python
from docling.chunking import HybridChunker, HierarchicalChunker
from docling.document_converter import DocumentConverter

# Convert document
converter = DocumentConverter()
result = converter.convert("path/to/document.pdf")
doc = result.document

# Hybrid chunker (balances content and structure)
chunker = HybridChunker(tokenizer="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v1")
chunks = list(chunker.chunk(doc))

# Hierarchical chunker (preserves document hierarchy)
h_chunker = HierarchicalChunker()
hierarchical_chunks = list(h_chunker.chunk(doc))
```

### Processing Complex Documents

For technical papers, manuals, or other complex documents, enable all enrichments:

```python
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

# Configure pipeline with all enrichments
pipeline_options = PdfPipelineOptions(
    do_code_enrichment=True,
    do_formula_enrichment=True,
    do_picture_classification=True,
    do_picture_description=True,
    do_table_structure=True
)

# Configure converter
converter = DocumentConverter(format_options={
    InputFormat.PDF: {
        "pipeline_options": pipeline_options
    }
})

# Convert document
result = converter.convert("technical_paper.pdf")
```

## Best Practices for Docling Integration

### Document Processing

1. **Match Processing to Document Types**: 
   - Use Docling for PDF, DOCX, and other complex formats
   - Your enhanced processor works well for plain text and markdown

2. **Enable Appropriate Enrichments**:
   - For code-heavy documents, enable `do_code_enrichment`
   - For academic papers, enable `do_formula_enrichment`
   - For visual content, enable `do_picture_classification`

3. **Evaluate Chunking Strategies**:
   - Compare Docling's chunkers with your semantic chunker on your documents
   - Some documents benefit from hierarchy-aware chunking, others from semantic boundaries

### Information Retrieval

1. **Leverage Rich Metadata**:
   - Use the heading information Docling extracts for more precise filtering
   - Page numbers can help users locate information in original documents

2. **Combine Retrieval Methods**:
   - Docling's structural understanding + Hybrid Search semantic/lexical capabilities
   - Filter by document structure before semantic search for better performance

3. **Custom Prompting**:
   - Adjust prompt templates to leverage Docling's structural information
   - Include headings and page numbers in context for better grounding

## Troubleshooting

### Common Issues and Solutions

1. **Memory Consumption**:
   - Docling models can consume significant memory for large documents
   - Process large documents in batches or reduce the quality of enrichments

2. **Processing Speed**:
   - Enable only the enrichments you need
   - Use CPU-optimized models for deployment scenarios

3. **Integration Conflicts**:
   - Ensure compatible versions between Docling and LangChain
   - Use the adapter pattern to isolate potential conflicts

### Error Handling

The provided integration components include robust error handling. For additional debugging, enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Tuning

For production deployments, consider these optimization strategies:

1. **Selective Processing**:
   - Only enable Docling enrichments needed for your use case
   - Process documents once and cache results

2. **Parallel Processing**:
   - Process multiple documents concurrently using a worker pool
   - Example implementation in `DoclingProcessor`

3. **Model Selection**:
   - Use smaller embedding models for faster processing
   - Consider quantized models for reduced memory footprint

4. **Chunking Optimization**:
   - Adjust chunk sizes based on your LLM's context window
   - Balance between chunk size and retrieval accuracy

## Conclusion

Integrating Docling with your enhanced RAG system provides significant improvements in document processing, information extraction, and retrieval accuracy. The flexible architecture allows you to use the best components for your specific needs.

For more information on Docling's capabilities, refer to:
- Docling documentation: https://ds4sd.github.io/docling/
- LangChain-Docling integration: https://python.langchain.com/docs/integrations/document_loaders/docling

The implementation components provide a seamless way to leverage Docling's document understanding capabilities while keeping the enhanced features you've already built like hybrid search and contextual reranking.
edRAGOrchestrator(
    use_docling=True,  # Use Docling for document processing
    use_langchain=True,  # Use LangChain for RAG operations
    use_hybrid_search=True  # Use hybrid search from your enhanced system
)

# Process and index documents
orchestrator.process_and_index(
    source_paths=["path/to/document.pdf"],
    force_reprocess=True
)

# Ask a question
response = orchestrator.ask(
    question="Explain the main concept in section 2.",
    top_k=5
)
print(response["answer"])
```

## Configuration Options

### Docling Processing Options

- **enable_enrichments**: Enable specialized processing for code, formulas, and images
- **use_docling_chunking**: Use Docling's built-in chunkers instead of your semantic chunker

### LangChain Integration Options

- **embedding_model_name**: Specify which embedding model to use
- **vector_db_type**: Choose between "faiss" and "chroma" vector stores
- **prompt_template**: Customize the prompt template for question answering

### Orchestrator Options

- **use_docling**: Toggle Docling document processing
- **use_langchain**: Toggle LangChain RAG components
- **use_hybrid_search**: Toggle hybrid search from your enhanced system
- **use_reranker**: Toggle contextual reranking

## Advanced Usage

### Combining Docling with Hybrid Search

One powerful configuration is to use Docling for document processing, your hybrid search for retrieval, and LangChain for the LLM interactions:

```python
orchestrator = DoclingEnhanc