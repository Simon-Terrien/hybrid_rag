from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
import spacy
import re
from typing import List, Dict, Any, Optional

class SemanticChunker:
    """Improved chunking system using semantic boundaries"""
    
    def __init__(self, 
                 base_chunk_size: int = 1000,
                 base_chunk_overlap: int = 200,
                 use_spacy: bool = False,
                 use_markdown_headers: bool = True):
        """
        Initialize the semantic chunker.
        
        Args:
            base_chunk_size: Base size for chunks
            base_chunk_overlap: Base overlap between chunks
            use_spacy: Whether to use spaCy for NER-based chunking
            use_markdown_headers: Whether to use markdown headers for splitting
        """
        self.base_chunk_size = base_chunk_size
        self.base_chunk_overlap = base_chunk_overlap
        self.use_spacy = use_spacy
        self.use_markdown_headers = use_markdown_headers
        
        # Load spaCy model if needed
        if use_spacy:
            try:
                self.nlp = spacy.load("fr_core_news_md")  # Using French model
            except:
                print("Installing French spaCy model...")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "fr_core_news_md"])
                self.nlp = spacy.load("fr_core_news_md")
    
    def _detect_document_type(self, text: str) -> str:
        """
        Detect the type of document based on content analysis.
        
        Args:
            text: The document text
            
        Returns:
            Document type: 'code', 'markdown', 'text'
        """
        # Check for code patterns
        code_patterns = [
            r'def\s+\w+\s*\(.*\)\s*:',  # Python function
            r'function\s+\w+\s*\(.*\)\s*{',  # JavaScript function
            r'class\s+\w+[\s\w]*{',  # Class definition
            r'import\s+[\w\s,{}]+\s+from\s+[\'"]',  # Import statements
            r'<\?php',  # PHP
            r'#!/usr/bin'  # Shebang
        ]
        
        for pattern in code_patterns:
            if re.search(pattern, text):
                return 'code'
        
        # Check for markdown
        markdown_patterns = [
            r'^#{1,6}\s+\w+',  # Headers
            r'^\s*[-*+]\s+\w+',  # List items
            r'!\[.*\]\(.*\)',  # Images
            r'\[.*\]\(.*\)'  # Links
        ]
        
        markdown_matches = 0
        for pattern in markdown_patterns:
            if re.search(pattern, text, re.MULTILINE):
                markdown_matches += 1
        
        if markdown_matches >= 2:
            return 'markdown'
        
        return 'text'
    
    def _adapt_chunking_parameters(self, text: str, doc_type: str) -> Dict[str, int]:
        """
        Adapts chunking parameters based on document type and content.
        
        Args:
            text: The document text
            doc_type: The detected document type
            
        Returns:
            Dictionary with adapted chunk_size and chunk_overlap
        """
        # Base values
        chunk_size = self.base_chunk_size
        chunk_overlap = self.base_chunk_overlap
        
        # Adjust based on document type
        if doc_type == 'code':
            # Smaller chunks for code with more overlap
            chunk_size = int(chunk_size * 0.7)
            chunk_overlap = int(chunk_size * 0.3)
        elif doc_type == 'markdown':
            # Standard chunks for markdown
            pass
        else:
            # Analyze text density
            avg_word_length = sum(len(word) for word in text.split()) / max(1, len(text.split()))
            
            if avg_word_length > 10:  # Technical content with long words
                chunk_size = int(chunk_size * 0.8)  # Smaller chunks
            elif len(text) > 50000:  # Long document
                chunk_size = int(chunk_size * 1.2)  # Larger chunks
        
        return {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }
    
    def _split_with_spacy(self, text: str, chunk_params: Dict[str, int]) -> List[str]:
        """
        Split text using spaCy NER to maintain entity context.
        
        Args:
            text: The document text
            chunk_params: Chunking parameters
            
        Returns:
            List of text chunks
        """
        doc = self.nlp(text)
        
        # Find entity boundaries
        entity_boundaries = set()
        for ent in doc.ents:
            entity_boundaries.add(ent.start_char)
            entity_boundaries.add(ent.end_char)
        
        # Also add sentence boundaries
        sentence_boundaries = set()
        for sent in doc.sents:
            sentence_boundaries.add(sent.start_char)
            sentence_boundaries.add(sent.end_char)
        
        # Combine all boundaries and sort
        all_boundaries = sorted(list(entity_boundaries.union(sentence_boundaries)))
        
        # Create initial segments based on sentences and entities
        segments = []
        for i in range(len(all_boundaries) - 1):
            segment = text[all_boundaries[i]:all_boundaries[i+1]]
            if segment.strip():
                segments.append(segment)
        
        # Combine segments into chunks of appropriate size
        chunks = []
        current_chunk = ""
        
        for segment in segments:
            if len(current_chunk) + len(segment) <= chunk_params["chunk_size"]:
                current_chunk += segment
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = segment
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_with_markdown(self, text: str) -> List[str]:
        """
        Split text using markdown headers.
        
        Args:
            text: The document text
            
        Returns:
            List of text chunks
        """
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        return markdown_splitter.split_text(text)
    
    def split_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into semantically meaningful chunks.
        
        Args:
            text: The document text
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        # Detect document type
        doc_type = self._detect_document_type(text)
        
        # Adapt chunking parameters
        chunk_params = self._adapt_chunking_parameters(text, doc_type)
        
        # Choose splitting strategy based on document type
        if doc_type == 'markdown' and self.use_markdown_headers:
            md_chunks = self._split_with_markdown(text)
            chunks = []
            for chunk in md_chunks:
                chunks.append({
                    "text": chunk.page_content,
                    "metadata": {
                        "doc_type": doc_type,
                        "headers": chunk.metadata
                    }
                })
            return chunks
        
        elif self.use_spacy and doc_type != 'code':
            raw_chunks = self._split_with_spacy(text, chunk_params)
            chunks = []
            for i, chunk in enumerate(raw_chunks):
                chunks.append({
                    "text": chunk,
                    "metadata": {
                        "doc_type": doc_type,
                        "chunk_index": i,
                        "semantic_chunk": True
                    }
                })
            return chunks
        
        else:
            # Fall back to RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_params["chunk_size"],
                chunk_overlap=chunk_params["chunk_overlap"],
                separators=["\n\n", "\n", ".", " ", ""]
            )
            
            docs = text_splitter.create_documents([text])
            chunks = []
            for i, doc in enumerate(docs):
                chunks.append({
                    "text": doc.page_content,
                    "metadata": {
                        "doc_type": doc_type,
                        "chunk_index": i,
                        "semantic_chunk": False
                    }
                })
            
            return chunks