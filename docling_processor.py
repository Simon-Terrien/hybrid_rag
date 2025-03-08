from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import time
from datetime import datetime

# Docling imports
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

# Import our existing components
from docling.datamodel.base_models import FormatBackendConfig
from semantic_chunker import SemanticChunker
from config import (
    DATA_DIR, PROCESSED_DIR, VECTOR_DB_DIR, logger,
    ROOT_DIR
)
from utils import (
    compute_file_hash, extract_metadata, load_json, save_json, generate_document_id, 
    generate_chunk_id, filter_complex_metadata, timed, log_exceptions
)

class DoclingProcessor:
    """Wrapper for Docling document processing capabilities"""
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
        enable_enrichments: bool = False,
        use_docling_chunking: bool = True
    ):
        """
        Initialise the processeur Docling.
        
        Args:
            data_dir: Répertoire des données brutes (défaut: depuis config)
            processed_dir: Répertoire des données traitées (défaut: depuis config)
            enable_enrichments: Activer les enrichissements Docling (code, formules, images, etc.)
            use_docling_chunking: Utiliser les chunkers intégrés à Docling
        """
        self.data_dir = data_dir or DATA_DIR
        self.processed_dir = processed_dir or PROCESSED_DIR
        self.enable_enrichments = enable_enrichments
        self.use_docling_chunking = use_docling_chunking
        
        # Create necessary directories
        self.processed_dir.mkdir(exist_ok=True, parents=True)
        (self.processed_dir / "docling").mkdir(exist_ok=True, parents=True)
        (self.processed_dir / "chunks").mkdir(exist_ok=True, parents=True)
        
        # Initialize Docling converter with appropriate options
        pipeline_options = PdfPipelineOptions(
            do_code_enrichment=enable_enrichments,
            do_formula_enrichment=enable_enrichments,
            do_picture_classification=enable_enrichments,
            do_table_structure=True  # Enable table structure extraction by default
        )
                
        # Configure PDF format options
        pdf_format_option = {
            InputFormat.PDF: FormatBackendConfig(
                backend="pdf",  # This is the key component missing
                pipeline_options=pipeline_options
            )
        }

        # Initialize DocumentConverter
        self.converter = DocumentConverter(format_options=pdf_format_option)
        
        # Initialize Docling chunker if enabled
        if use_docling_chunking:
            self.chunker = HybridChunker(tokenizer="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v1")
        
        logger.info(f"DoclingProcessor initialized: enable_enrichments={enable_enrichments}, use_docling_chunking={use_docling_chunking}")
    
    @timed(logger=logger)
    @log_exceptions(logger=logger)
    def process_document(self, file_path: Path, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Process a document using Docling.
        
        Args:
            file_path: Path to the document
            force_reprocess: Force reprocessing even if already processed
            
        Returns:
            Document processing result information
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Le fichier {file_path} n'existe pas")
        
        # Generate document ID based on hash
        doc_hash = compute_file_hash(file_path)
        document_id = generate_document_id(file_path, doc_hash)
        rel_path = str(file_path.relative_to(self.data_dir))
        
        # Output paths
        docling_file = self.processed_dir / "docling" / f"{document_id}.json"
        chunk_file = self.processed_dir / "chunks" / f"{document_id}.json"
        
        # Check if already processed and hash matches (unless force_reprocess)
        if not force_reprocess and docling_file.exists():
            logger.info(f"Document {rel_path} already processed with Docling")
            # We could check hash here but skip for simplicity
            
            # Load the already processed chunks
            chunks = load_json(chunk_file)
            return {
                "document_id": document_id,
                "file_path": str(file_path),
                "hash": doc_hash,
                "chunk_count": len(chunks),
                "chunk_file": str(chunk_file.relative_to(self.processed_dir)),
                "docling_file": str(docling_file.relative_to(self.processed_dir))
            }
        
        logger.info(f"Processing document with Docling: {rel_path}")
        
        # Convert the document using Docling
        start_time = time.time()
        conversion_result = self.converter.convert(str(file_path))
        docling_document = conversion_result.document
        
        # Extract basic metadata
        doc_metadata = extract_metadata(file_path)
        
        # Add Docling-specific metadata
        docling_metadata = {
            "docling_version": docling_document.version,
            "page_count": len(docling_document.get_pages())
        }
        doc_metadata.update(docling_metadata)
        
        # Export the Docling document to JSON for storage
        docling_json = docling_document.export_to_dict()
        save_json(docling_json, docling_file)
        
        # Process chunks either with Docling's chunker or convert to text for our semantic chunker
        if self.use_docling_chunking:
            # Use Docling's built-in chunker
            raw_chunks = list(self.chunker.chunk(docling_document))
            
            # Format chunks with appropriate metadata
            chunks = []
            for i, chunk in enumerate(raw_chunks):
                chunk_id = generate_chunk_id(document_id, i)
                
                # Prepare chunk metadata (combining document metadata and chunk-specific metadata)
                chunk_metadata = doc_metadata.copy()
                chunk_metadata.update({
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "chunk_index": i,
                    "total_chunks": len(raw_chunks)
                })
                
                # Add Docling-specific metadata from the chunk
                if "meta" in chunk and isinstance(chunk["meta"], dict):
                    # Extract headings for better context
                    if "headings" in chunk["meta"]:
                        chunk_metadata["headings"] = chunk["meta"]["headings"]
                    
                    # Extract page numbers if available
                    if "doc_items" in chunk["meta"]:
                        page_numbers = set()
                        for item in chunk["meta"]["doc_items"]:
                            if "prov" in item:
                                for prov in item["prov"]:
                                    if "page_no" in prov:
                                        page_numbers.add(prov["page_no"])
                        
                        if page_numbers:
                            chunk_metadata["page_numbers"] = sorted(list(page_numbers))
                
                # Create the formatted chunk
                chunks.append({
                    "text": chunk["text"],
                    "metadata": filter_complex_metadata(chunk_metadata)
                })
        else:
            # Convert Docling document to markdown for processing with our semantic chunker
            markdown_text = docling_document.export_to_markdown()
            
            # Use our semantic chunker
            semantic_chunker = SemanticChunker()
            chunking_result = semantic_chunker.split_text(markdown_text)
            
            # Format chunks
            chunks = []
            for i, chunk in enumerate(chunking_result):
                chunk_id = generate_chunk_id(document_id, i)
                
                # Prepare chunk metadata
                chunk_metadata = doc_metadata.copy()
                chunk_metadata.update({
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "chunk_index": i,
                    "total_chunks": len(chunking_result)
                })
                
                # Merge with chunk-specific metadata
                if "metadata" in chunk and isinstance(chunk["metadata"], dict):
                    chunk_metadata.update(chunk["metadata"])
                
                # Create the formatted chunk
                chunks.append({
                    "text": chunk["text"],
                    "metadata": filter_complex_metadata(chunk_metadata)
                })
        
        # Save the chunks
        save_json(chunks, chunk_file)
        
        processing_time = time.time() - start_time
        logger.info(f"Document {rel_path} processed with Docling in {processing_time:.2f}s: {len(chunks)} chunks generated")
        
        return {
            "document_id": document_id,
            "file_path": str(file_path),
            "hash": doc_hash,
            "chunk_count": len(chunks),
            "chunk_file": str(chunk_file.relative_to(self.processed_dir)),
            "docling_file": str(docling_file.relative_to(self.processed_dir)),
            "processing_time_seconds": processing_time
        }
    
    @timed(logger=logger)
    def process_directory(self, subdirectory: Optional[str] = None, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Process all documents in the data directory.
        
        Args:
            subdirectory: Optional subdirectory to process
            force_reprocess: Force reprocessing even if already processed
            
        Returns:
            Processing results
        """
        base_dir = self.data_dir
        if subdirectory:
            base_dir = base_dir / subdirectory
            logger.info(f"Processing subdirectory {base_dir}")
            if not base_dir.exists():
                raise FileNotFoundError(f"Le sous-répertoire {subdirectory} n'existe pas")
        
        # Collect all files with supported extensions
        all_files = []
        supported_extensions = ['.pdf', '.docx', '.html', '.txt', '.md']
        for ext in supported_extensions:
            all_files.extend(list(base_dir.glob(f"**/*{ext}")))
        
        logger.info(f"Processing {len(all_files)} files with Docling in {base_dir}")
        
        results = {
            "total_documents": len(all_files),
            "processed_documents": 0,
            "failed_documents": 0,
            "documents": []
        }
        
        # Process each file
        for file_path in all_files:
            try:
                doc_result = self.process_document(file_path, force_reprocess)
                results["processed_documents"] += 1
                results["documents"].append({
                    "path": str(file_path.relative_to(self.data_dir)),
                    "document_id": doc_result["document_id"],
                    "chunk_count": doc_result["chunk_count"],
                    "status": "processed"
                })
            except Exception as e:
                logger.error(f"Error processing {file_path} with Docling: {str(e)}")
                results["failed_documents"] += 1
                results["documents"].append({
                    "path": str(file_path.relative_to(self.data_dir)),
                    "error": str(e),
                    "status": "failed"
                })
        
        # Save processing results
        save_json(results, self.processed_dir / "docling_processing_results.json")
        
        logger.info(f"Docling processing completed: {results['processed_documents']} processed, "
                  f"{results['failed_documents']} failed")
        
        return results