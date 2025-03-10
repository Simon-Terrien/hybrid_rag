from pathlib import Path
from typing import  Dict, Any, Optional
import time
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption
)
# Docling imports
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
# Import our existing components
from semantic_chunker import SemanticChunker
from config import (
    DATA_DIR, PROCESSED_DIR,  logger
    
)
from utils import (
    compute_file_hash, extract_metadata, load_json, save_json, generate_document_id, 
    generate_chunk_id, filter_complex_metadata, timed, log_exceptions
)
import re
import unicodedata
import nltk
from typing import List, Optional, Dict, Any

class TextCleaner:
    """
    A comprehensive text cleaning utility with various preprocessing methods.
    """
    def __init__(
        self, 
        min_token_length: int = 2, 
        max_token_length: int = 30,
        varremove_stopwords: bool = True,
        do_stemming: bool = False,
        do_lemmatization: bool = False
    ):
        """
        Initialize the TextCleaner with configurable options.
        
        Args:
            min_token_length: Minimum length of tokens to keep
            max_token_length: Maximum length of tokens to keep
            remove_stopwords: Whether to remove stopwords
            do_stemming: Whether to apply stemming
            do_lemmatization: Whether to apply lemmatization
        """
        # Check for optional dependencies
        self.NLTK_AVAILABLE = True
        self.SPACY_AVAILABLE = True
        self.CONTRACTIONS_AVAILABLE = True
        
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize
            from nltk.stem import PorterStemmer, WordNetLemmatizer
            
            self.stopwords_list = set(stopwords.words('english'))
            self.word_tokenize = word_tokenize
            self.stemmer = PorterStemmer()
            self.lemmatizer = WordNetLemmatizer()
        except ImportError:
            self.NLTK_AVAILABLE = False
            self.stopwords_list = []
            self.word_tokenize = lambda x: x.split()
            self.stemmer = None
            self.lemmatizer = None
        
        try:
            import spacy
            self.nlp = spacy.load('en_core_web_sm')
        except ImportError:
            self.SPACY_AVAILABLE = False
            self.nlp = None
        
        try:
            from contractions import contractions_dict
            self.contractions_dict = contractions_dict
        except ImportError:
            self.CONTRACTIONS_AVAILABLE = False
            self.contractions_dict = {}
        
        # Preprocessing configurations
        self.min_token_length = min_token_length
        self.max_token_length = max_token_length
        self.varremove_stopwords = varremove_stopwords
        self.do_stemming = do_stemming
        self.do_lemmatization = do_lemmatization
    
    def clean_text(self, text: str, domain: Optional[str] = None) -> str:
        """
        Apply a comprehensive set of text cleaning steps.
        
        Args:
            text: Input text to clean
            domain: Optional domain-specific processing (medical, legal, finance, technical)
        
        Returns:
            Cleaned text
        """
        # Domain-specific preprocessing first
        if domain == 'medical':
            text = self.process_medical(text)
        elif domain == 'legal':
            text = self.process_legal(text)
        elif domain == 'finance':
            text = self.process_finance(text)
        elif domain == 'technical':
            text = self.process_technical(text)
        
        # Apply general cleaning steps
        text = self.strip_html(text)
        text = self.remove_accents(text)
        
        if self.CONTRACTIONS_AVAILABLE:
            text = self.expand_contractions(text)
        
        text = self.remove_special_chars(text)
        text = self.remove_extra_whitespace(text)
        
        # Optional text processing steps
        if self.varremove_stopwords:
            text = self.remove_stopwords(text)
        
        if self.do_lemmatization:
            text = self.lemmatize(text)
        elif self.do_stemming:
            text = self.stem(text)
        
        text = self.filter_by_length(text)
        text = self.lowercase(text)
        
        return text
    
    def strip_html(self, text: str) -> str:
        """Remove HTML tags from text"""
        return re.sub(r'<.*?>', '', text)
    
    def lowercase(self, text: str) -> str:
        """Convert text to lowercase"""
        return text.lower()
    
    def remove_accents(self, text: str) -> str:
        """Remove accents from text"""
        return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    
    def expand_contractions(self, text: str) -> str:
        """Expand contractions (e.g., don't -> do not)"""
        if not self.contractions_dict:
            return text
            
        def replace(match):
            match_str = match.group(0)
            if match_str.lower() in self.contractions_dict:
                return self.contractions_dict[match_str.lower()]
            return match_str
            
        return re.sub(r'\b\w+\'(?:t|s|d|ll|m|re|ve)\b', replace, text)
    
    def remove_special_chars(self, text: str) -> str:
        """Remove special characters"""
        # Keep alphanumeric, spaces, and basic punctuation
        return re.sub(r'[^a-zA-Z0-9\s.,;:!?()[\]{}"\'-]', '', text)
    
    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords"""
        if not self.stopwords_list:
            return text
            
        # First tokenize
        if self.NLTK_AVAILABLE:
            tokens = self.word_tokenize(text)
            filtered_tokens = [word for word in tokens if word.lower() not in self.stopwords_list]
            return ' '.join(filtered_tokens)
        else:
            # Simple tokenization fallback
            words = text.split()
            filtered_words = [word for word in words if word.lower() not in self.stopwords_list]
            return ' '.join(filtered_words)
    
    def remove_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace"""
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading and trailing whitespace
        return text.strip()
    
    def stem(self, text: str) -> str:
        """Apply stemming to words"""
        if not self.stemmer or not self.NLTK_AVAILABLE:
            return text
            
        tokens = self.word_tokenize(text)
        stemmed_tokens = [self.stemmer.stem(word) for word in tokens]
        return ' '.join(stemmed_tokens)
    
    def lemmatize(self, text: str) -> str:
        """Apply lemmatization to words"""
        if self.nlp and self.SPACY_AVAILABLE:
            # Use spaCy for lemmatization
            doc = self.nlp(text)
            return ' '.join([token.lemma_ for token in doc])
        elif self.lemmatizer and self.NLTK_AVAILABLE:
            # Use NLTK for lemmatization
            tokens = self.word_tokenize(text)
            lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
            return ' '.join(lemmatized_tokens)
        else:
            return text
    
    def filter_by_length(self, text: str) -> str:
        """Filter tokens by length"""
        if self.NLTK_AVAILABLE:
            tokens = self.word_tokenize(text)
            filtered_tokens = [
                word for word in tokens 
                if len(word) >= self.min_token_length and len(word) <= self.max_token_length
            ]
            return ' '.join(filtered_tokens)
        else:
            # Simple tokenization fallback
            words = text.split()
            filtered_words = [
                word for word in words 
                if len(word) >= self.min_token_length and len(word) <= self.max_token_length
            ]
            return ' '.join(filtered_words)
    
    # Domain-specific processing methods (same as in the previous implementation)
    def process_medical(self, text: str) -> str:
        replacements = {
            r'\bmg\b': 'milligrams',
            r'\bdr\b': 'doctor',
            r'\bdrs\b': 'doctors',
            r'\bpt\b': 'patient',
            r'\bpts\b': 'patients',
            r'\btx\b': 'treatment',
            r'\bdx\b': 'diagnosis',
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def process_legal(self, text: str) -> str:
        replacements = {
            r'\bplf\b': 'plaintiff',
            r'\bdef\b': 'defendant',
            r'\bv\.\b': 'versus',
            r'\bart\.\b': 'article',
            r'\bsec\.\b': 'section',
            r'\bno\.\b': 'number',
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def process_finance(self, text: str) -> str:
        replacements = {
            r'\$': 'dollar ',
            r'€': 'euro ',
            r'£': 'pound ',
            r'\bq[1-4]\b': 'quarter',
            r'\bfy\b': 'fiscal year',
            r'\brev\b': 'revenue',
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def process_technical(self, text: str) -> str:
        # Handle code snippets differently
        code_blocks = re.findall(r'```.*?```', text, re.DOTALL)
        for i, block in enumerate(code_blocks):
            # Replace code blocks with placeholders
            text = text.replace(block, f"CODE_BLOCK_{i}")
        
        # Process the text normally
        text = self.remove_extra_whitespace(text)
        
        # Put code blocks back
        for i, block in enumerate(code_blocks):
            text = text.replace(f"CODE_BLOCK_{i}", block.strip('`'))
        
        return text
    

class DoclingProcessor:
    """Wrapper for Docling document processing capabilities"""
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
        enable_enrichments: bool = False,
        use_docling_chunking: bool = True,
        cleaning_config: Optional[Dict[str, Any]] = None,
        domain: Optional[str] = None
    ):
        """
        Initialise the DoclingProcessor.
        
        Args:
            data_dir: Directory for raw data (default: from config)
            processed_dir: Directory for processed data (default: from config)
            enable_enrichments: Enable Docling enrichments (code, formulas, images, etc.)
            use_docling_chunking: Use Docling's integrated chunkers
            cleaning_config: Configuration for text cleaning
            domain: Optional domain for domain-specific text processing
        """
        self.data_dir = data_dir or DATA_DIR
        self.processed_dir = processed_dir or PROCESSED_DIR
        self.enable_enrichments = enable_enrichments
        self.use_docling_chunking = use_docling_chunking
        self.domain = domain
        
        # Create necessary directories
        self.processed_dir.mkdir(exist_ok=True, parents=True)
        (self.processed_dir / "docling").mkdir(exist_ok=True, parents=True)
        (self.processed_dir / "chunks").mkdir(exist_ok=True, parents=True)
        
        # Initialize text cleaner
        cleaning_config = cleaning_config or {}
        self.text_cleaner = TextCleaner(**cleaning_config)
        
        # Configure PDF pipeline options
        pipeline_options = PdfPipelineOptions(
            do_code_enrichment=enable_enrichments,
            do_formula_enrichment=enable_enrichments,
            do_picture_classification=enable_enrichments,
            do_table_structure=True  # Enable table structure extraction by default
        )
        
        # Configure PDF format options
        pdf_format_option = {
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend
            )
        }

        # Initialize DocumentConverter
        self.converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options=pdf_format_option
        )
        
        # Initialize Docling chunker if enabled
        if use_docling_chunking:
            self.chunker = HybridChunker(
                tokenizer="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        
        logger.info(f"DoclingProcessor initialized: enable_enrichments={enable_enrichments}, use_docling_chunking={use_docling_chunking}")
        
    @timed(logger=logger)
    @log_exceptions(logger=logger)
    def process_document(self, file_path: Path, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Process a document using Docling with robust error handling.
        
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
        
        try:
            # Convert the document using Docling
            start_time = time.time()
            conversion_result = self.converter.convert(str(file_path))
            docling_document = conversion_result.document
            
            # Convert Docling document to markdown 
            markdown_text = docling_document.export_to_markdown()
            
            # Clean the markdown text
            cleaned_markdown = self.text_cleaner.clean_text(
                markdown_text, 
                domain=self.domain
            )
        except Exception as conversion_error:
            logger.warning(f"Docling conversion failed for {rel_path}: {conversion_error}")
            
            # Fallback: use a basic text extraction method
            try:
                # Try to extract text using PyPDF2 or another library
                import PyPDF2
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    markdown_text = "\n\n".join([page.extract_text() for page in pdf_reader.pages])
                
                # Clean the extracted text
                cleaned_markdown = self.text_cleaner.clean_text(
                    markdown_text, 
                    domain=self.domain
                )
            except Exception as fallback_error:
                logger.error(f"Fallback text extraction failed for {rel_path}: {fallback_error}")
                raise
        
        # Process chunks 
        try:
            if self.use_docling_chunking:
                # Use our semantic chunker as a fallback
                semantic_chunker = SemanticChunker()
                raw_chunks = semantic_chunker.split_text(cleaned_markdown)
            else:
                # Use our semantic chunker on cleaned markdown
                semantic_chunker = SemanticChunker()
                raw_chunks = semantic_chunker.split_text(cleaned_markdown)
        except Exception as chunking_error:
            logger.warning(f"Chunking failed for {rel_path}: {chunking_error}")
            # Fallback to splitting by paragraphs or lines
            raw_chunks = [
                {"text": chunk} 
                for chunk in cleaned_markdown.split("\n\n") 
                if chunk.strip()
            ]
        
        # Format chunks
        chunks = []
        for i, chunk in enumerate(raw_chunks):
            chunk_id = generate_chunk_id(document_id, i)
            
            # Prepare chunk metadata
            chunk_metadata = {
                "document_id": document_id,
                "chunk_id": chunk_id,
                "chunk_index": i,
                "total_chunks": len(raw_chunks)
            }
            
            # Create the formatted chunk
            chunks.append({
                "text": chunk.get("text", chunk) if isinstance(chunk, dict) else chunk,
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