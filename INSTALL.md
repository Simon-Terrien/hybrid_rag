# Installation Guide

This guide provides detailed installation instructions for the Docling RAG system.

## System Requirements

- Python 3.8+ (3.9 or 3.10 recommended)
- 8GB RAM minimum (16GB+ recommended)
- 2GB free disk space for code and dependencies
- [Optional] NVIDIA GPU with CUDA support for faster processing
- [Optional] 10GB+ free disk space for vector database storage (depends on your document collection size)

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/docling-rag.git
cd docling-rag
```

### 2. Set Up Virtual Environment

#### On Linux/macOS:
```bash
python -m venv .venv
source .venv/bin/activate
```

#### On Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

#### Basic Installation
```bash
pip install -r requirements.txt
```

#### With GPU Support
```bash
pip install -r requirements-cuda.txt
```

### 4. Install Additional Dependencies

#### Docling 
Docling requires some specific dependencies:
```bash
pip install PyPDF2 docx2txt
```

#### OCR Support (Optional)
If you need OCR support for image text extraction:
```bash
pip install pytesseract
```

**Note**: You also need to install Tesseract OCR:
- On Ubuntu: `sudo apt install tesseract-ocr`
- On macOS: `brew install tesseract`
- On Windows: Download and install from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

### 5. Model Installation

By default, the system will download models when first used. If you want to pre-download the models:

```bash
python -m scripts.download_models
```

### 6. Configure Environment

Create a `.env` file in the project root with your configuration:

```
# Data directories
DATA_DIR=data
PROCESSED_DIR=processed_data
VECTOR_DB_DIR=vector_db

# Embedding model
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
USE_GPU=true

# Vector database
VECTOR_DB_TYPE=faiss

# Search configuration
DEFAULT_TOP_K=5
SIMILARITY_THRESHOLD=0.7
```

### 7. Verify Installation

Run the system check to verify your installation:

```bash
python main.py system --status
```

You should see output indicating all components are available.

## Docker Installation (Alternative)

If you prefer using Docker:

### 1. Build the Docker Image

```bash
docker build -t docling-rag .
```

### 2. Run the Container

```bash
docker run -p 8000:8000 -v $(pwd)/data:/app/data -v $(pwd)/processed_data:/app/processed_data -v $(pwd)/vector_db:/app/vector_db docling-rag
```

## Troubleshooting

### Common Issues

#### Package installation errors
If you encounter errors installing packages:
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

#### Memory errors during indexing
If you encounter memory errors during indexing, try:
1. Reduce the batch size in `config.py` (`BATCH_SIZE` setting)
2. Process fewer documents at once

#### CUDA issues
If you face CUDA-related errors:
1. Ensure your NVIDIA drivers are up to date
2. Check CUDA and cuDNN compatibility with your PyTorch version
3. Try setting `USE_GPU=false` in your `.env` file to fall back to CPU

### Getting Help

If you encounter issues not covered here:
1. Check the [Troubleshooting Guide](docs/TROUBLESHOOTING.md)
2. Open an issue on GitHub with:
   - Your environment details
   - Error messages
   - Steps to reproduce