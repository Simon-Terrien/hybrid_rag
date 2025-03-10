from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import time
from datetime import datetime
import logging
import traceback

# Import RAG components
from doclingroc.docling_processor import DoclingProcessor

# Import our components
from utils import log_exceptions, timed
from vectorproc.vector_indexer import VectorIndexer
from searchproc.semantic_search import SemanticSearchEngine
from searchproc.hybrid_search import HybridSearchEngine
from searchproc.contextual_reranker import ContextualReranker

# Import configuration
from config import (
    DATA_DIR, PROCESSED_DIR, VECTOR_DB_DIR, 
    LLM_CONFIG, SEARCH_CONFIG, logger
)

class DoclingEnhancedRAGOrchestrator:
    """
    Enhanced RAG orchestrator that combines Docling document processing 
    with existing RAG pipeline.
    """
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
        vector_db_dir: Optional[Path] = None,
        use_docling: bool = True,
        use_hybrid_search: bool = True,
        use_reranker: bool = True
    ):
        """Initialize the Docling-enhanced RAG orchestrator."""
        # Set directories
        self.data_dir = data_dir or DATA_DIR
        self.processed_dir = processed_dir or PROCESSED_DIR
        self.vector_db_dir = vector_db_dir or VECTOR_DB_DIR
        
        # Set component flags
        self.use_docling = use_docling
        self.use_hybrid_search = use_hybrid_search
        self.use_reranker = use_reranker
        
        logger.info(f"Initializing DoclingEnhancedRAGOrchestrator with settings:")
        logger.info(f"  - data_dir: {self.data_dir}")
        logger.info(f"  - processed_dir: {self.processed_dir}")
        logger.info(f"  - vector_db_dir: {self.vector_db_dir}")
        logger.info(f"  - use_docling: {self.use_docling}")
        logger.info(f"  - use_hybrid_search: {self.use_hybrid_search}")
        logger.info(f"  - use_reranker: {self.use_reranker}")
        
        # Initialize components
        self._docling_processor = None
        self._vector_indexer = None
        self._semantic_search = None
        self._hybrid_search = None
        self._reranker = None
        self._base_orchestrator = None
        
        # System state
        self.system_state = {
            "status": "initialized",
            "last_update": datetime.now().isoformat(),
            "components": {
                "docling": {"status": "not_initialized" if use_docling else "disabled"},
                "base_orchestrator": {"status": "not_initialized"}
            },
            "statistics": {
                "total_documents": 0,
                "total_chunks": 0,
                "total_queries": 0,
                "last_processing_time": None,
                "last_indexing_time": None
            }
        }
        
        logger.info(f"DoclingEnhancedRAGOrchestrator initialized with: use_docling={use_docling}, "
                    f"use_hybrid_search={use_hybrid_search}")
    
    def _update_system_state(self, component: str, status: str, **kwargs):
        """Update system state for a component"""
        logger.info(f"Updating system state for component '{component}' to '{status}'")
        
        if component not in self.system_state["components"]:
            self.system_state["components"][component] = {"status": "not_initialized"}
            
        old_status = self.system_state["components"][component]["status"]
        self.system_state["components"][component]["status"] = status
        self.system_state["components"][component].update(kwargs)
        self.system_state["last_update"] = datetime.now().isoformat()
        
        # Log any additional component state information
        for key, value in kwargs.items():
            logger.info(f"  - {component}.{key}: {value}")
        
        # Update global status
        any_failed = any(comp["status"] == "failed" for comp in self.system_state["components"].values())
        all_ready = all(comp["status"] in ["ready", "disabled"] for comp in self.system_state["components"].values())
        
        old_global_status = self.system_state["status"]
        
        if any_failed:
            self.system_state["status"] = "partially_failed"
        elif all_ready:
            self.system_state["status"] = "ready"
        else:
            self.system_state["status"] = "initializing"
            
        if old_global_status != self.system_state["status"]:
            logger.info(f"Global system state changed: {old_global_status} -> {self.system_state['status']}")
    
    @property
    def docling_processor(self) -> Optional[DoclingProcessor]:
        """Get or initialize the Docling processor"""
        if self.use_docling and self._docling_processor is None:
            logger.info("Lazy-loading Docling processor...")
            try:
                logger.info(f"Initializing DoclingProcessor with data_dir={self.data_dir}, processed_dir={self.processed_dir}")
                self._docling_processor = DoclingProcessor(
                    data_dir=self.data_dir,
                    processed_dir=self.processed_dir,
                    enable_enrichments=True  # Enable code, formula, and image enrichments
                )
                logger.info("DoclingProcessor initialization successful")
                self._update_system_state("docling", "ready")
            except Exception as e:
                logger.error(f"Error initializing Docling processor: {str(e)}")
                logger.error(f"Exception type: {type(e).__name__}")
                logger.error("Stack trace:")
                traceback.print_exc()
                self._update_system_state("docling", "failed", error=str(e))
                logger.error("Docling processor initialization failed")
        elif not self.use_docling:
            logger.info("Docling processor is disabled by configuration")
        else:
            logger.debug("Using existing Docling processor instance")
        
        return self._docling_processor
    
    @property
    def vector_indexer(self) -> VectorIndexer:
        """Get or initialize the vector indexer"""
        if self._vector_indexer is None:
            logger.info("Lazy-loading vector indexer...")
            try:
                # First, check if document registry exists
                from utils import save_json
                import os
                from config import DOCUMENT_REGISTRY_PATH
                
                logger.info(f"Checking for document registry at: {DOCUMENT_REGISTRY_PATH}")
                
                # Create document registry if it doesn't exist
                if not os.path.exists(DOCUMENT_REGISTRY_PATH):
                    logger.info(f"Document registry not found. Creating new registry at {DOCUMENT_REGISTRY_PATH}")
                    # Create parent directories if they don't exist
                    os.makedirs(os.path.dirname(DOCUMENT_REGISTRY_PATH), exist_ok=True)
                    
                    # Create an empty registry
                    empty_registry = {
                        "documents": {},
                        "last_updated": datetime.now().isoformat()
                    }
                    save_json(empty_registry, DOCUMENT_REGISTRY_PATH)
                    logger.info("Empty document registry created successfully")
                else:
                    logger.info("Document registry found")
                
                # Now initialize the vector indexer
                logger.info(f"Initializing VectorIndexer with processed_dir={self.processed_dir}, vector_db_dir={self.vector_db_dir}")
                self._vector_indexer = VectorIndexer(
                    processed_dir=self.processed_dir,
                    vector_db_dir=self.vector_db_dir
                )
                logger.info("VectorIndexer initialized successfully")
                self._update_system_state("vector_indexer", "ready")
            except Exception as e:
                logger.error(f"Error initializing vector indexer: {str(e)}")
                logger.error(f"Exception type: {type(e).__name__}")
                logger.error(f"Stack trace:")
                traceback.print_exc()
                logger.error("Vector indexer initialization failed")
                self._update_system_state("vector_indexer", "failed", error=str(e))
        
        return self._vector_indexer
    
    @property
    def semantic_search(self) -> SemanticSearchEngine:
        """Get or initialize the semantic search engine"""
        if self._semantic_search is None:
            logger.info("Lazy-loading semantic search engine...")
            try:
                logger.info("Initializing SemanticSearchEngine")
                self._semantic_search = SemanticSearchEngine(
                    vector_indexer=self.vector_indexer
                )
                logger.info("SemanticSearchEngine initialized successfully")
                self._update_system_state("semantic_search", "ready")
            except Exception as e:
                logger.error(f"Error initializing semantic search: {str(e)}")
                logger.error(f"Exception type: {type(e).__name__}")
                logger.error("Stack trace:")
                traceback.print_exc()
                logger.error("Semantic search engine initialization failed")
                self._update_system_state("semantic_search", "failed", error=str(e))
        else:
            logger.debug("Using existing semantic search engine instance")
        
        return self._semantic_search
    
    @property
    def hybrid_search(self) -> HybridSearchEngine:
        """Get or initialize the hybrid search engine"""
        if self.use_hybrid_search and self._hybrid_search is None:
            logger.info("Lazy-loading hybrid search engine...")
            try:
                semantic_weight = SEARCH_CONFIG.get("semantic_weight", 0.5)
                logger.info(f"Initializing HybridSearchEngine with semantic_weight={semantic_weight}")
                self._hybrid_search = HybridSearchEngine(
                    semantic_search_engine=self.semantic_search,
                    semantic_weight=semantic_weight
                )
                logger.info("HybridSearchEngine initialized successfully")
                self._update_system_state("hybrid_search", "ready")
            except Exception as e:
                logger.error(f"Error initializing hybrid search: {str(e)}")
                logger.error(f"Exception type: {type(e).__name__}")
                logger.error("Stack trace:")
                traceback.print_exc()
                logger.error("Hybrid search engine initialization failed")
                self._update_system_state("hybrid_search", "failed", error=str(e))
        elif not self.use_hybrid_search:
            logger.info("Hybrid search is disabled by configuration")
            self._update_system_state("hybrid_search", "disabled")
        else:
            logger.debug("Using existing hybrid search engine instance")
        
        return self._hybrid_search
    
    @property
    def reranker(self) -> ContextualReranker:
        """Get or initialize the reranker"""
        if self.use_reranker and self._reranker is None:
            logger.info("Lazy-loading contextual reranker...")
            try:
                logger.info("Initializing ContextualReranker")
                self._reranker = ContextualReranker(
                    enabled=self.use_reranker
                )
                logger.info("ContextualReranker initialized successfully")
                self._update_system_state("reranker", "ready")
            except Exception as e:
                logger.error(f"Error initializing reranker: {str(e)}")
                logger.error(f"Exception type: {type(e).__name__}")
                logger.error("Stack trace:")
                traceback.print_exc()
                logger.error("Contextual reranker initialization failed")
                self._update_system_state("reranker", "failed", error=str(e))
        elif not self.use_reranker:
            logger.info("Contextual reranker is disabled by configuration")
            self._update_system_state("reranker", "disabled")
        else:
            logger.debug("Using existing contextual reranker instance")
        
        return self._reranker
    
    @timed(logger=logger)
    @log_exceptions(logger=logger)
    def process_documents(
        self, 
        source_paths: Optional[Union[str, List[str], Path, List[Path]]] = None,
        subdirectory: Optional[str] = None,
        force_reprocess: bool = False,
        use_docling: Optional[bool] = None,
        async_mode: bool = False
    ) -> Dict[str, Any]:
        """Process documents using either Docling or the base processor."""
        start_time = time.time()
        
        logger.info(f"Starting document processing with parameters:")
        logger.info(f"  - source_paths: {source_paths}")
        logger.info(f"  - subdirectory: {subdirectory}")
        logger.info(f"  - force_reprocess: {force_reprocess}")
        logger.info(f"  - async_mode: {async_mode}")
        
        # Determine whether to use Docling
        use_docling_proc = use_docling if use_docling is not None else self.use_docling
        logger.info(f"Using Docling processor: {use_docling_proc}")
        
        if not use_docling_proc:
            logger.info("Falling back to base orchestrator for document processing")
            # Process with base orchestrator instead
            if not source_paths and not subdirectory:
                logger.info("No source paths or subdirectory specified, processing all documents")
                subdirectory = ""  # Process all documents in data_dir
            
            return self.base_orchestrator.process_documents(
                subdirectory=subdirectory,
                force_reprocess=force_reprocess,
                async_mode=async_mode
            )
        
        # Process with Docling
        logger.info("Getting Docling processor instance")
        processor = self.docling_processor
        
        if processor is None:
            logger.error("Docling processor could not be initialized")
            return {
                "status": "failed",
                "error": "Docling processor initialization failed",
                "processor": "docling",
                "total_documents": 0,
                "processed_documents": 0,
                "failed_documents": 0,
                "documents": []
            }
            
        result = {}
        
        if source_paths:
            logger.info(f"Processing specific file(s)")
            # Process specific files
            if not isinstance(source_paths, list):
                source_paths = [source_paths]
                logger.info(f"Converted single path to list: {source_paths}")
            
            # Convert to Path objects
            paths = [Path(p) if isinstance(p, str) else p for p in source_paths]
            logger.info(f"Converted to Path objects: {paths}")
            
            # Process each file
            processed_docs = []
            for path in paths:
                if not path.is_absolute():
                    # Relative to data_dir
                    path = self.data_dir / path
                    logger.info(f"Converting to absolute path: {path}")
                
                logger.info(f"Processing document: {path}")
                try:
                    doc_result = processor.process_document(path, force_reprocess)
                    logger.info(f"Document processed successfully: {path}")
                    logger.info(f"  - document_id: {doc_result['document_id']}")
                    logger.info(f"  - chunk_count: {doc_result['chunk_count']}")
                    
                    processed_docs.append({
                        "path": str(path),
                        "document_id": doc_result["document_id"],
                        "chunk_count": doc_result["chunk_count"],
                        "status": "processed"
                    })
                except Exception as e:
                    logger.error(f"Error processing {path} with Docling: {str(e)}")
                    logger.error(f"Exception type: {type(e).__name__}")
                    logger.error("Stack trace:")
                    traceback.print_exc()
                    
                    processed_docs.append({
                        "path": str(path),
                        "error": str(e),
                        "status": "failed"
                    })
            
            processed_count = sum(1 for d in processed_docs if d["status"] == "processed")
            failed_count = sum(1 for d in processed_docs if d["status"] == "failed")
            
            logger.info(f"Document processing summary:")
            logger.info(f"  - Total documents: {len(source_paths)}")
            logger.info(f"  - Processed successfully: {processed_count}")
            logger.info(f"  - Failed: {failed_count}")
            
            result = {
                "processor": "docling",
                "total_documents": len(source_paths),
                "processed_documents": processed_count,
                "failed_documents": failed_count,
                "documents": processed_docs,
                "processing_time_seconds": time.time() - start_time
            }
        else:
            # Process directory
            logger.info(f"Processing directory: {subdirectory or 'root data directory'}")
            result = processor.process_directory(subdirectory, force_reprocess)
            result["processor"] = "docling"
            
            logger.info(f"Directory processing complete:")
            logger.info(f"  - Total documents: {result['total_documents']}")
            logger.info(f"  - Processed successfully: {result['processed_documents']}")
            logger.info(f"  - Failed: {result['failed_documents']}")
        
        # Update system state
        self.system_state["statistics"]["last_processing_time"] = datetime.now().isoformat()
        self.system_state["statistics"]["total_documents"] += result.get("processed_documents", 0)
        
        logger.info(f"Document processing completed in {time.time() - start_time:.2f} seconds")
        return result
    
    @timed(logger=logger)
    @log_exceptions(logger=logger)
    def index_documents(
        self, 
        force_reindex: bool = False,
        async_mode: bool = False
    ) -> Dict[str, Any]:
        """Index processed documents in the vector database."""
        logger.info(f"Starting document indexing with parameters:")
        logger.info(f"  - force_reindex: {force_reindex}")
        logger.info(f"  - async_mode: {async_mode}")
        
        if async_mode:
            logger.info("Running in async mode, returning task ID immediately")
            # Start async indexing and return immediately
            task_id = f"index_{int(time.time())}"
            return {"task_id": task_id, "status": "indexing", "message": "Indexing documents in progress"}
        
        # Check if vector_indexer is initialized
        logger.info("Getting vector indexer instance")
        vector_indexer = self.vector_indexer
        
        if vector_indexer is None:
            logger.error("Vector indexer could not be initialized")
            return {
                "status": "failed",
                "error": "Vector indexer could not be initialized",
                "documents": [],
                "indexed_documents": 0,
                "failed_documents": 0
            }
        
        # Perform indexing
        try:
            logger.info(f"Starting vector indexing with force_reindex={force_reindex}")
            result = vector_indexer.index_all_documents(force_reindex=force_reindex)
            
            # Log indexing results
            logger.info(f"Indexing completed:")
            logger.info(f"  - Total documents: {result.get('total_documents', 0)}")
            logger.info(f"  - Indexed documents: {result.get('indexed_documents', 0)}")
            logger.info(f"  - Skipped documents: {result.get('skipped_documents', 0)}")
            logger.info(f"  - Failed documents: {result.get('failed_documents', 0)}")
            
            # Update statistics
            logger.info("Retrieving vector database statistics")
            stats = vector_indexer.get_stats()
            logger.info(f"Vector DB stats:")
            logger.info(f"  - Total documents: {stats['total_documents']}")
            logger.info(f"  - Total chunks: {stats['total_chunks']}")
            
            self.system_state["statistics"]["last_indexing_time"] = datetime.now().isoformat()
            self.system_state["statistics"]["total_documents"] = stats["total_documents"]
            self.system_state["statistics"]["total_chunks"] = stats["total_chunks"]
            
            # Rebuild lexical indices for hybrid search
            if self._hybrid_search:
                logger.info("Rebuilding lexical indices for hybrid search")
                self.hybrid_search.rebuild_indices()
                logger.info("Lexical indices rebuilt successfully")
            
            logger.info("Document indexing completed successfully")
            return result
        except Exception as e:
            logger.error(f"Failed to index documents: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error("Stack trace:")
            traceback.print_exc()
            
            return {
                "status": "failed",
                "error": str(e),
                "documents": [],
                "indexed_documents": 0,
                "failed_documents": 0
            }
 
    @timed(logger=logger)
    @log_exceptions(logger=logger)
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        search_type: str = "hybrid",
        rerank_results: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Search for documents using semantic or hybrid search with logging."""
        
        logger.info(f"Received search query: '{query}' | Top K: {top_k} | Search Type: {search_type}")

        # Determine whether to use the reranker
        use_reranker = rerank_results if rerank_results is not None else self.use_reranker
        logger.debug(f"Search config -> Reranker: {use_reranker}")
        
        # Select the search engine
        if search_type == "hybrid" and self.use_hybrid_search:
            search_engine = self.hybrid_search
            logger.info("Using hybrid search engine.")
        else:
            search_engine = self.semantic_search
            logger.info("Using semantic search engine.")
        
        try:
            # Perform search
            logger.info("Executing search...")
            result = search_engine.search(
                query=query,
                top_k=top_k,
                filters=filter_dict,
                return_all_chunks=True
            )
            logger.info(f"Search completed successfully. Found {len(result.get('results', []))} results.")
            
            # Apply reranking if requested
            if use_reranker and self._reranker and "results" in result:
                logger.info("Applying reranking to search results...")
                result["results"] = self.reranker.rerank(
                    query=query,
                    results=result["results"],
                    top_k=top_k
                )
                logger.info("Reranking applied successfully.")
            
            result["searcher"] = search_type

        except Exception as e:
            logger.error(f"Error occurred during search: {str(e)}", exc_info=True)
            result = {"query": query, "searcher": search_type, "error": str(e)}

        # Update statistics
        self.system_state["statistics"]["total_queries"] += 1
        logger.info(f"Search query logged. Total queries so far: {self.system_state['statistics']['total_queries']}")

        return result
    
    @timed(logger=logger)
    @log_exceptions(logger=logger)
    def ask(
        self,
        question: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        search_type: str = "hybrid",
        prompt_template: Optional[str] = None,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Ask a question using the RAG system."""
        # 1. Search for relevant context
        search_results = self.search(
            query=question,
            top_k=top_k,
            filter_dict=filter_dict,
            search_type=search_type
        )
        
        # 2. Extract the relevant text passages
        context_texts = []
        sources = []
        
        if "results" in search_results:
            for result in search_results["results"]:
                context_texts.append(result["text"])
                sources.append({
                    "text": result["text"],
                    "metadata": result["metadata"]
                })
        
        # 3. Generate an answer using the context
        # This is a simplified version
        combined_context = "\n\n".join(context_texts)
        
        # In a real implementation, you would use an LLM like this:
        # answer = self._generate_answer(question, combined_context, prompt_template)
        
        # Placeholder
        answer = f"Based on the {len(context_texts)} context passages, I would answer: [LLM would generate answer here]"
        
        # Format the result
        result = {
            "question": question,
            "answer": answer,
            "sources": sources,
            "processing_time_seconds": search_results.get("processing_time_seconds", 0)
        }
        
        return result
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current system state"""
        # Update stats if vector indexer is initialized
        if self._vector_indexer:
            stats = self.vector_indexer.get_stats()
            self.system_state["statistics"]["total_documents"] = stats["total_documents"]
            self.system_state["statistics"]["total_chunks"] = stats["total_chunks"]
        
        return self.system_state
