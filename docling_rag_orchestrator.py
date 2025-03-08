from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import time
from datetime import datetime
import logging
import traceback
import json

# Import RAG components
from docling_processor import DoclingProcessor
from langchain_integration import LangChainRAGAdapter
from rag_pipeline_orchestrator import RAGPipelineOrchestrator

# Import existing components if needed
from vector_indexer import VectorIndexer
from semantic_search import SemanticSearchEngine
from hybrid_search import HybridSearchEngine
from contextual_reranker import ContextualReranker

# Import configuration
from config import (
    DATA_DIR, PROCESSED_DIR, VECTOR_DB_DIR, 
    LLM_CONFIG, SEARCH_CONFIG, logger
)

class DoclingEnhancedRAGOrchestrator:
    """
    Enhanced RAG orchestrator that combines Docling document processing 
    with our existing RAG pipeline and LangChain integration.
    """
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
        vector_db_dir: Optional[Path] = None,
        use_docling: bool = True,
        use_langchain: bool = True,
        use_hybrid_search: bool = True,
        use_reranker: bool = True
    ):
        """
        Initialize the Docling-enhanced RAG orchestrator.
        
        Args:
            data_dir: Raw data directory (default: from config)
            processed_dir: Processed data directory (default: from config)
            vector_db_dir: Vector database directory (default: from config)
            use_docling: Whether to use Docling for document processing
            use_langchain: Whether to use LangChain for RAG operations
            use_hybrid_search: Whether to use hybrid search
            use_reranker: Whether to use contextual reranking
        """
        self.data_dir = data_dir or DATA_DIR
        self.processed_dir = processed_dir or PROCESSED_DIR
        self.vector_db_dir = vector_db_dir or VECTOR_DB_DIR
        self.use_docling = use_docling
        self.use_langchain = use_langchain
        self.use_hybrid_search = use_hybrid_search
        self.use_reranker = use_reranker
        
        # Initialize components to None (lazy loading)
        self._docling_processor = None
        self._langchain_adapter = None
        self._base_orchestrator = None
        
        # System state
        self.system_state = {
            "status": "initialized",
            "last_update": datetime.now().isoformat(),
            "components": {
                "docling": {"status": "not_initialized" if use_docling else "disabled"},
                "langchain": {"status": "not_initialized" if use_langchain else "disabled"},
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
        
        logger.info(f"DoclingEnhancedRAGOrchestrator initialized: use_docling={use_docling}, "
                  f"use_langchain={use_langchain}, use_hybrid_search={use_hybrid_search}")
    
    def _update_system_state(self, component: str, status: str, **kwargs):
        """Update system state for a component"""
        self.system_state["components"][component]["status"] = status
        self.system_state["components"][component].update(kwargs)
        self.system_state["last_update"] = datetime.now().isoformat()
        
        # Update global status
        any_failed = any(comp["status"] == "failed" for comp in self.system_state["components"].values())
        all_ready = all(comp["status"] in ["ready", "disabled"] for comp in self.system_state["components"].values())
        
        if any_failed:
            self.system_state["status"] = "partially_failed"
        elif all_ready:
            self.system_state["status"] = "ready"
        else:
            self.system_state["status"] = "initializing"
    
    @property
    def docling_processor(self) -> Optional[DoclingProcessor]:
        """Get or initialize the Docling processor"""
        if self.use_docling and self._docling_processor is None:
            try:
                logger.info("Initializing Docling processor")
                self._docling_processor = DoclingProcessor(
                    data_dir=self.data_dir,
                    processed_dir=self.processed_dir,
                    enable_enrichments=True  # Enable code, formula, and image enrichments
                )
                self._update_system_state("docling", "ready")
            except Exception as e:
                logger.error(f"Error initializing Docling processor: {str(e)}")
                self._update_system_state("docling", "failed", error=str(e))
                traceback.print_exc()
        
        return self._docling_processor
    
    @property
    def langchain_adapter(self) -> Optional[LangChainRAGAdapter]:
        """Get or initialize the LangChain adapter"""
        if self.use_langchain and self._langchain_adapter is None:
            try:
                logger.info("Initializing LangChain RAG adapter")
                self._langchain_adapter = LangChainRAGAdapter(
                    data_dir=self.data_dir,
                    processed_dir=self.processed_dir,
                    vector_db_dir=self.vector_db_dir,
                    embedding_model_name=SEARCH_CONFIG.get("embedding_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v1"),
                    vector_db_type=SEARCH_CONFIG.get("vector_db_type", "faiss"),
                    use_docling=self.use_docling
                )
                self._update_system_state("langchain", "ready")
            except Exception as e:
                logger.error(f"Error initializing LangChain adapter: {str(e)}")
                self._update_system_state("langchain", "failed", error=str(e))
                traceback.print_exc()
        
        return self._langchain_adapter
    
    @property
    def base_orchestrator(self) -> RAGPipelineOrchestrator:
        """Get or initialize the base RAG orchestrator"""
        if self._base_orchestrator is None:
            try:
                logger.info("Initializing base RAG orchestrator")
                self._base_orchestrator = RAGPipelineOrchestrator(
                    data_dir=self.data_dir,
                    processed_dir=self.processed_dir,
                    vector_db_dir=self.vector_db_dir
                )
                self._update_system_state("base_orchestrator", "ready")
            except Exception as e:
                logger.error(f"Error initializing base RAG orchestrator: {str(e)}")
                self._update_system_state("base_orchestrator", "failed", error=str(e))
                traceback.print_exc()
        
        return self._base_orchestrator
    
    def process_documents(
        self, 
        source_paths: Optional[Union[str, List[str], Path, List[Path]]] = None,
        subdirectory: Optional[str] = None,
        force_reprocess: bool = False,
        use_docling: Optional[bool] = None,
        async_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Process documents using either Docling or the base processor.
        
        Args:
            source_paths: Specific document path(s) to process
            subdirectory: Subdirectory in data_dir to process
            force_reprocess: Force reprocessing even if already processed
            use_docling: Whether to use Docling (overrides instance setting)
            async_mode: Whether to process asynchronously
            
        Returns:
            Processing results
        """
        start_time = time.time()
        
        # Determine whether to use Docling
        use_docling_proc = use_docling if use_docling is not None else self.use_docling
        
        result = {}
        
        if use_docling_proc:
            # Process with Docling
            processor = self.docling_processor
            
            if source_paths:
                # Process specific files
                if not isinstance(source_paths, list):
                    source_paths = [source_paths]
                
                # Convert to Path objects
                paths = [Path(p) if isinstance(p, str) else p for p in source_paths]
                
                # Process each file
                processed_docs = []
                for path in paths:
                    if not path.is_absolute():
                        # Relative to data_dir
                        path = self.data_dir / path
                    
                    try:
                        doc_result = processor.process_document(path, force_reprocess)
                        processed_docs.append({
                            "path": str(path),
                            "document_id": doc_result["document_id"],
                            "chunk_count": doc_result["chunk_count"],
                            "status": "processed"
                        })
                    except Exception as e:
                        logger.error(f"Error processing {path} with Docling: {str(e)}")
                        processed_docs.append({
                            "path": str(path),
                            "error": str(e),
                            "status": "failed"
                        })
                
                result = {
                    "processor": "docling",
                    "total_documents": len(source_paths),
                    "processed_documents": sum(1 for d in processed_docs if d["status"] == "processed"),
                    "failed_documents": sum(1 for d in processed_docs if d["status"] == "failed"),
                    "documents": processed_docs,
                    "processing_time_seconds": time.time() - start_time
                }
            else:
                # Process directory
                result = processor.process_directory(subdirectory, force_reprocess)
                result["processor"] = "docling"
        else:
            # Process with base orchestrator
            if source_paths:
                logger.warning("Base orchestrator does not support processing specific source paths. Using subdirectory instead.")
            
            result = self.base_orchestrator.process_documents(
                subdirectory=subdirectory,
                force_reprocess=force_reprocess,
                async_mode=async_mode
            )
            result["processor"] = "base"
        
        # Update system state
        self.system_state["statistics"]["last_processing_time"] = datetime.now().isoformat()
        self.system_state["statistics"]["total_documents"] += result.get("processed_documents", 0)
        
        return result
    
    def index_documents(
        self,
        source_paths: Optional[Union[str, List[str], Path, List[Path]]] = None,
        force_reindex: bool = False,
        use_langchain: Optional[bool] = None,
        vector_store_id: str = "default",
        async_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Index documents using either LangChain or the base indexer.
        
        Args:
            source_paths: Specific document path(s) to index (LangChain only)
            force_reindex: Force reindexing even if already indexed
            use_langchain: Whether to use LangChain (overrides instance setting)
            vector_store_id: Identifier for the vector store (LangChain only)
            async_mode: Whether to index asynchronously (base indexer only)
            
        Returns:
            Indexing results
        """
        start_time = time.time()
        
        # Determine whether to use LangChain
        use_langchain_idx = use_langchain if use_langchain is not None else self.use_langchain
        
        result = {}
        
        if use_langchain_idx:
            # Index with LangChain
            adapter = self.langchain_adapter
            
            if source_paths:
                # Index specific files
                result = adapter.process_and_index_documents(
                    source_paths=source_paths,
                    vector_store_id=vector_store_id,
                    use_docling=self.use_docling
                )
                result["indexer"] = "langchain"
            else:
                # Index all processed documents
                logging.warning("LangChain indexer requires specific source paths. Please provide source_paths.")
                # Future implementation could scan processed_dir for documents
                result = {
                    "indexer": "langchain",
                    "error": "No source paths provided for LangChain indexer",
                    "status": "failed"
                }
        else:
            # Index with base orchestrator
            if source_paths:
                logger.warning("Base orchestrator does not support indexing specific source paths.")
            
            result = self.base_orchestrator.index_documents(
                force_reindex=force_reindex,
                async_mode=async_mode
            )
            result["indexer"] = "base"
        
        # Update system state
        self.system_state["statistics"]["last_indexing_time"] = datetime.now().isoformat()
        
        if "document_count" in result:
            self.system_state["statistics"]["total_chunks"] = result["document_count"]
        elif "indexed_documents" in result:
            self.system_state["statistics"]["total_chunks"] = result.get("total_chunks", 0)
        
        return result
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        search_type: str = "hybrid",
        use_langchain: Optional[bool] = None,
        vector_store_id: str = "default",
        rerank_results: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Search for documents using either LangChain or the base search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_dict: Dictionary of metadata filters
            search_type: Type of search ("semantic", "hybrid")
            use_langchain: Whether to use LangChain (overrides instance setting)
            vector_store_id: Identifier for the vector store (LangChain only)
            rerank_results: Whether to use contextual reranking
            
        Returns:
            Search results
        """
        # Determine whether to use LangChain
        use_langchain_search = use_langchain if use_langchain is not None else self.use_langchain
        
        # Determine whether to use reranker
        use_reranker = rerank_results if rerank_results is not None else self.use_reranker
        
        result = {}
        
        if use_langchain_search:
            # Search with LangChain
            adapter = self.langchain_adapter
            
            try:
                # Load vector store if not already loaded
                if adapter.vector_store is None:
                    adapter.load_vector_store(vector_store_id)
                
                # Perform search
                docs = adapter.search(
                    query=query,
                    top_k=top_k,
                    filter_dict=filter_dict
                )
                
                # Format results
                results = []
                for i, doc in enumerate(docs):
                    results.append({
                        "rank": i + 1,
                        "text": doc.page_content,
                        "metadata": doc.metadata
                    })
                
                result = {
                    "query": query,
                    "searcher": "langchain",
                    "total_results": len(results),
                    "results": results
                }
                
            except Exception as e:
                logger.error(f"Error searching with LangChain: {str(e)}")
                result = {
                    "query": query,
                    "searcher": "langchain",
                    "error": str(e),
                    "status": "failed"
                }
        else:
            # Search with base orchestrator
            result = self.base_orchestrator.search(
                query=query,
                top_k=top_k,
                filters=filter_dict,
                search_type=search_type,
                rerank_results=use_reranker
            )
            result["searcher"] = "base"
        
        # Update stats
        self.system_state["statistics"]["total_queries"] += 1
        
        return result
    
    def ask(
        self,
        question: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        use_langchain: Optional[bool] = None,
        vector_store_id: str = "default",
        prompt_template: Optional[str] = None,
        streaming: bool = False,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ask a question using either LangChain or the base QA.
        
        Args:
            question: Question to ask
            top_k: Number of context documents to retrieve
            filter_dict: Dictionary of metadata filters
            use_langchain: Whether to use LangChain (overrides instance setting)
            vector_store_id: Identifier for the vector store (LangChain only)
            prompt_template: Custom prompt template (LangChain only)
            streaming: Whether to stream the response (LangChain only)
            conversation_id: Conversation ID for base QA
            
        Returns:
            Answer and source documents
        """
        # Determine whether to use LangChain
        use_langchain_qa = use_langchain if use_langchain is not None else self.use_langchain
        
        result = {}
        
        if use_langchain_qa:
            # QA with LangChain
            adapter = self.langchain_adapter
            
            try:
                # Load vector store if not already loaded
                if adapter.vector_store is None:
                    adapter.load_vector_store(vector_store_id)
                
                # Perform QA
                result = adapter.ask(
                    question=question,
                    top_k=top_k,
                    filter_dict=filter_dict,
                    prompt_template=prompt_template,
                    streaming=streaming
                )
                
                # Format for consistent output
                sources = []
                for doc in result["source_documents"]:
                    sources.append({
                        "text": doc.page_content,
                        "metadata": doc.metadata
                    })
                
                qa_result = {
                    "question": question,
                    "answer": result["answer"],
                    "sources": sources,
                    "qa_system": "langchain",
                    "processing_time_seconds": result["processing_time_seconds"]
                }
                
                result = qa_result
                
            except Exception as e:
                logger.error(f"Error with LangChain QA: {str(e)}")
                result = {
                    "question": question,
                    "qa_system": "langchain",
                    "error": str(e),
                    "status": "failed"
                }
        else:
            # QA with base orchestrator
            search_params = {
                "top_k": top_k,
                "filters": filter_dict,
                "search_type": "hybrid",
                "rerank_results": self.use_reranker
            }
            
            result = self.base_orchestrator.ask(
                question=question,
                conversation_id=conversation_id,
                search_params=search_params
            )
            result["qa_system"] = "base"
        
        # Update stats
        self.system_state["statistics"]["total_queries"] += 1
        
        return result
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current system state"""
        # Update from base orchestrator if available
        if self._base_orchestrator is not None:
            base_state = self.base_orchestrator.get_state()
            self.system_state["statistics"].update(base_state["statistics"])
        
        return self.system_state
    
    def process_and_index(
        self,
        source_paths: Optional[Union[str, List[str], Path, List[Path]]] = None,
        subdirectory: Optional[str] = None,
        force_reprocess: bool = False,
        force_reindex: bool = False,
        use_docling: Optional[bool] = None,
        use_langchain: Optional[bool] = None,
        vector_store_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Process and index documents in a single operation.
        
        Args:
            source_paths: Specific document path(s) to process
            subdirectory: Subdirectory in data_dir to process
            force_reprocess: Force reprocessing even if already processed
            force_reindex: Force reindexing even if already indexed
            use_docling: Whether to use Docling (overrides instance setting)
            use_langchain: Whether to use LangChain (overrides instance setting)
            vector_store_id: Identifier for the vector store (LangChain only)
            
        Returns:
            Processing and indexing results
        """
        start_time = time.time()
        
        # Process documents
        process_result = self.process_documents(
            source_paths=source_paths,
            subdirectory=subdirectory,
            force_reprocess=force_reprocess,
            use_docling=use_docling
        )
        
        # Index documents
        index_result = self.index_documents(
            source_paths=source_paths,
            force_reindex=force_reindex,
            use_langchain=use_langchain,
            vector_store_id=vector_store_id
        )
        
        # Combine results
        result = {
            "processing": process_result,
            "indexing": index_result,
            "total_time_seconds": time.time() - start_time
        }
        
        return result