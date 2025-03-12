# Docling RAG System

A powerful Retrieval-Augmented Generation (RAG) system built with Python that leverages document processing, vector-based search, and LLM integration to provide accurate answers based on your document collection.

## 🌟 Features

- **Powerful Document Processing**: Automatic processing of various document formats including PDF, DOCX, HTML, TXT and MD
- **Semantic Search**: Find relevant information using meaning-based search powered by embeddings
- **Hybrid Search**: Combine semantic and lexical search for improved accuracy
- **Context-Aware Answers**: Generate answers to questions based on document context
- **Flexible API**: Use via command line or REST API
- **Customizable**: Configurable components and parameters for different use cases

## 📋 Quick Start

### Prerequisites

- Python 3.8+
- [Optional] GPU with CUDA support for faster processing

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/docling-rag.git
cd docling-rag

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
# install spacy lib needed
python -m spacy download fr_core_news_lg
```

### Basic Usage

1. Process your documents:
   ```bash
   python main.py process --dir your_documents_directory
   ```

2. Index the processed documents:
   ```bash
   python main.py index
   ```

3. Search for information:
   ```bash
   python main.py search "your search query"
   ```

4. Ask questions about your documents:
   ```bash
   python main.py ask "your question"
   ```

5. Start the API server:
   ```bash
   python server.py
   ```

## 📚 Documentation

For more detailed documentation, see:

- [Installation Guide](docs/INSTALL.md)
- [Code Structure](docs/CODE_STRUCTURE.md)
- [API Reference](docs/API.md)
- [Configuration Guide](docs/CONFIGURATION.md)
- [Contribution Guidelines](docs/CONTRIBUTING.md)

## 🧪 Example

```python
# Python API usage example
from rag_orchestrator_api import RAGOrchestratorAPI
import asyncio

async def main():
    # Create orchestrator
    rag = RAGOrchestratorAPI()
    
    # Process a document
    result = await rag.process_document("path/to/document.pdf")
    print(f"Document processed: {result['document_id']}")
    
    # Index documents
    await rag.index_documents()
    
    # Search for information
    search_results = await rag.search("vector databases")
    
    # Ask a question
    answer = await rag.ask("How do vector databases work?")
    print(f"Answer: {answer['answer']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🔧 Configuration

Configuration is managed through `config.py` and `config.json`. Key settings include:

- **Data Paths**: Configure where documents and processed data are stored
- **Embedding Model**: Select the embedding model for vectorization
- **Vector Database**: Choose between FAISS or Chroma
- **Chunking Parameters**: Configure how documents are split into chunks
- **Search Settings**: Adjust search parameters for optimal results

## 🚀 Advanced Features

- **Cross-Encoder Reranking**: Improve search result ranking
- **Hybrid Search**: Combine semantic and lexical search capabilities
- **Document Enrichment**: Extract code, formulas, and images from documents
- **Custom Chunking**: Semantic-based document splitting for better context

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- Docling for document processing components
- LangChain for vector store and embeddings integrations
- HuggingFace for transformer models
- FastAPI for the REST API framework