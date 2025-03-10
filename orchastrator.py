from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import time
from datetime import datetime
import os
import json
import traceback

# Import RAG components
from doclingroc.docling_rag_orchestrator import DoclingEnhancedRAGOrchestrator

# Import configuration
from config import (
    DATA_DIR, PROCESSED_DIR, VECTOR_DB_DIR, 
    DOCUMENT_REGISTRY_PATH, logger, SEARCH_CONFIG
)

class RAGOrchestrator:
    """
    Main API orchestrator for the RAG system.
    This class serves as a facade for the underlying RAG components
    and will be easy to convert to a FastAPI application.
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
        """Initialize the RAG API orchestrator"""
        logger.info("Initializing RAGOrchestratorAPI")
        
        # Set directories
        self.data_dir = data_dir or DATA_DIR
        self.processed_dir = processed_dir or PROCESSED_DIR
        self.vector_db_dir = vector_db_dir or VECTOR_DB_DIR
        
        # Ensure document registry exists
        self._ensure_document_registry()
        
        # Initialize the RAG orchestrator
        self.orchestrator = DoclingEnhancedRAGOrchestrator(
            data_dir=data_dir,
            processed_dir=processed_dir,
            vector_db_dir=vector_db_dir,
            use_docling=use_docling,
            use_hybrid_search=use_hybrid_search,
            use_reranker=use_reranker
        )
        
        # Initialize task tracking
        self.tasks = {}
        
        logger.info(f"RAGOrchestratorAPI initialized successfully")
    
    def _ensure_document_registry(self):
        """Ensure the document registry file exists"""
        from utils import save_json
        
        if not os.path.exists(DOCUMENT_REGISTRY_PATH):
            logger.info(f"Creating document registry at {DOCUMENT_REGISTRY_PATH}")
            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(DOCUMENT_REGISTRY_PATH), exist_ok=True)
            
            # Create an empty registry
            empty_registry = {
                "documents": {},
                "last_updated": datetime.now().isoformat()
            }
            save_json(empty_registry, DOCUMENT_REGISTRY_PATH)
            logger.info("Empty document registry created successfully")
    
    # Document Processing API
    
    async def process_document(
        self, 
        file_path: Union[str, Path],
        force_reprocess: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single document
        
        Args:
            file_path: Path to the document file
            force_reprocess: Whether to force reprocessing even if already processed
            
        Returns:
            Processing result
        """
        logger.info(f"API: Processing document {file_path}")
        
        try:
            file_path = Path(file_path) if isinstance(file_path, str) else file_path
            
            # Process the document
            result = self.orchestrator.process_documents(
                source_paths=[file_path],
                force_reprocess=force_reprocess
            )
            
            # Format the result for API
            api_result = {
                "status": "success" if result.get("processed_documents", 0) > 0 else "failed",
                "document_id": result.get("documents", [{}])[0].get("document_id", None) if result.get("documents") else None,
                "processing_time": result.get("processing_time_seconds", 0),
                "details": result
            }
            
            return api_result
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error("Stack trace:")
            traceback.print_exc()
            
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def process_directory(
        self,
        directory: Optional[str] = None,
        force_reprocess: bool = False
    ) -> Dict[str, Any]:
        """
        Process all documents in a directory
        
        Args:
            directory: Subdirectory to process (relative to data_dir)
            force_reprocess: Whether to force reprocessing
            
        Returns:
            Processing results
        """
        logger.info(f"API: Processing directory {directory or 'root'}")
        
        try:
            # Process the directory
            result = self.orchestrator.process_documents(
                subdirectory=directory,
                force_reprocess=force_reprocess
            )
            
            # Format the result for API
            api_result = {
                "status": "success",
                "processed_count": result.get("processed_documents", 0),
                "failed_count": result.get("failed_documents", 0),
                "processing_time": result.get("processing_time_seconds", 0),
                "details": result
            }
            
            return api_result
            
        except Exception as e:
            logger.error(f"Error processing directory {directory}: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error("Stack trace:")
            traceback.print_exc()
            
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    # Indexing API
    
    async def index_documents(self, force_reindex: bool = False) -> Dict[str, Any]:
        """
        Index all processed documents
        
        Args:
            force_reindex: Whether to force reindexing
            
        Returns:
            Indexing results
        """
        logger.info(f"API: Indexing documents with force_reindex={force_reindex}")
        
        try:
            # Index documents
            result = self.orchestrator.index_documents(force_reindex=force_reindex)
            
            if result.get("status") == "failed":
                return {
                    "status": "error",
                    "error": result.get("error", "Unknown indexing error"),
                    "details": result
                }
            
            # Format the result for API
            api_result = {
                "status": "success",
                "indexed_count": result.get("indexed_documents", 0),
                "skipped_count": result.get("skipped_documents", 0),
                "failed_count": result.get("failed_documents", 0),
                "total_count": result.get("total_documents", 0),
                "processing_time": result.get("total_processing_time", 0),
                "details": result
            }
            
            return api_result
            
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error("Stack trace:")
            traceback.print_exc()
            
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    # Search API
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        search_type: str = "hybrid",
        rerank_results: bool = True
    ) -> Dict[str, Any]:
        """
        Search for documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_dict: Metadata filters
            search_type: Search type ("hybrid", "semantic")
            rerank_results: Whether to rerank results
            
        Returns:
            Search results
        """
        logger.info(f"API: Searching for '{query}' with type={search_type}, top_k={top_k}")
        
        try:
            # Perform search
            result = self.orchestrator.search(
                query=query,
                top_k=top_k,
                filter_dict=filter_dict,
                search_type=search_type,
                rerank_results=rerank_results
            )
            
            # Format the result for API
            api_result = {
                "status": "success",
                "query": query,
                "result_count": result.get("total_results", 0),
                "processing_time": result.get("processing_time_seconds", 0),
                "results": result.get("results", []),
                "details": {
                    "search_type": search_type,
                    "reranking": rerank_results
                }
            }
            
            return api_result
            
        except Exception as e:
            logger.error(f"Error searching for '{query}': {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error("Stack trace:")
            traceback.print_exc()
            
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    # QA API
    
    async def ask(
        self,
        question: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        search_type: str = "hybrid",
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ask a question using the RAG system
        
        Args:
            question: Question to ask
            top_k: Number of context passages to retrieve
            filter_dict: Metadata filters
            search_type: Search type ("hybrid", "semantic")
            conversation_id: Optional conversation ID for context
            
        Returns:
            Answer and source passages
        """
        logger.info(f"API: Answering question '{question}'")
        
        try:
            # Get answer
            result = self.orchestrator.ask(
                question=question,
                top_k=top_k,
                filter_dict=filter_dict,
                search_type=search_type,
                conversation_id=conversation_id
            )
            
            # Format the result for API
            api_result = {
                "status": "success",
                "question": question,
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "processing_time": result.get("processing_time_seconds", 0),
                "conversation_id": conversation_id
            }
            
            return api_result
            
        except Exception as e:
            logger.error(f"Error answering question '{question}': {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error("Stack trace:")
            traceback.print_exc()
            
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    # System status API
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get the current system status
        
        Returns:
            System status information
        """
        logger.info("API: Getting system status")
        
        try:
            # Get system state
            state = self.orchestrator.get_state()
            
            # Get vector DB stats if available
            vector_stats = {}
            if hasattr(self.orchestrator, "vector_indexer") and self.orchestrator.vector_indexer:
                try:
                    vector_stats = self.orchestrator.vector_indexer.get_stats()
                except Exception as e:
                    logger.warning(f"Could not get vector DB stats: {str(e)}")
            
            # Combine into API result
            api_result = {
                "status": "success",
                "system_status": state.get("status", "unknown"),
                "components": state.get("components", {}),
                "statistics": {
                    **state.get("statistics", {}),
                    "vector_db": vector_stats
                },
                "last_update": state.get("last_update", datetime.now().isoformat())
            }
            
            return api_result
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error("Stack trace:")
            traceback.print_exc()
            
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    # Document management API
    
    async def get_document_info(self, document_id: str) -> Dict[str, Any]:
        """
        Get information about a specific document
        
        Args:
            document_id: Document ID
            
        Returns:
            Document information
        """
        logger.info(f"API: Getting info for document {document_id}")
        
        try:
            # Get document registry
            from utils import load_json
            registry = load_json(DOCUMENT_REGISTRY_PATH)
            
            # Find the document
            document_info = None
            for doc_path, doc_info in registry.get("documents", {}).items():
                if doc_info.get("document_id") == document_id:
                    document_info = {
                        "document_id": document_id,
                        "file_path": doc_path,
                        "metadata": doc_info.get("metadata", {}),
                        "hash": doc_info.get("hash", ""),
                        "chunk_file": doc_info.get("chunk_file", "")
                    }
                    break
            
            if not document_info:
                return {
                    "status": "error",
                    "error": f"Document not found: {document_id}"
                }
            
            # Get chunk information if available
            chunk_file_path = self.processed_dir / document_info.get("chunk_file", "")
            chunks = []
            
            if chunk_file_path.exists():
                try:
                    chunks = load_json(chunk_file_path)
                    document_info["chunk_count"] = len(chunks)
                except Exception as e:
                    logger.warning(f"Could not load chunks for document {document_id}: {str(e)}")
            
            # Format the result for API
            api_result = {
                "status": "success",
                "document": document_info,
                "chunks": {
                    "count": len(chunks),
                    "path": str(chunk_file_path)
                }
            }
            
            return api_result
            
        except Exception as e:
            logger.error(f"Error getting document info for {document_id}: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error("Stack trace:")
            traceback.print_exc()
            
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def list_documents(
        self, 
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        List all documents in the system
        
        Args:
            limit: Maximum number of documents to return
            offset: Offset for pagination
            
        Returns:
            List of documents
        """
        logger.info(f"API: Listing documents with limit={limit}, offset={offset}")
        
        try:
            # Get document registry
            from utils import load_json
            registry = load_json(DOCUMENT_REGISTRY_PATH)
            
            # Get all documents
            documents = []
            for doc_path, doc_info in registry.get("documents", {}).items():
                documents.append({
                    "document_id": doc_info.get("document_id", ""),
                    "file_path": doc_path,
                    "metadata": doc_info.get("metadata", {}),
                    "hash": doc_info.get("hash", "")
                })
            
            # Apply pagination
            total_count = len(documents)
            documents = documents[offset:offset+limit]
            
            # Format the result for API
            api_result = {
                "status": "success",
                "documents": documents,
                "pagination": {
                    "total": total_count,
                    "limit": limit,
                    "offset": offset,
                    "has_more": (offset + limit) < total_count
                }
            }
            
            return api_result
            
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error("Stack trace:")
            traceback.print_exc()
            
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    # Helper to convert to FastAPI
    def create_fastapi_app(self):
        """
        Create a FastAPI application from this orchestrator
        
        Returns:
            FastAPI application
        """
        from fastapi import FastAPI, File, UploadFile, Form, Query, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel, Field
        from typing import List, Optional, Dict, Any
        
        # Create FastAPI app
        app = FastAPI(
            title="RAG API",
            description="API for RAG (Retrieval Augmented Generation) system",
            version="1.0.0"
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Define models
        class SearchQuery(BaseModel):
            query: str = Field(..., description="Search query")
            top_k: int = Field(5, description="Number of results to return")
            filter_dict: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
            search_type: str = Field("hybrid", description="Search type")
            rerank_results: bool = Field(True, description="Whether to rerank results")
        
        class QuestionQuery(BaseModel):
            question: str = Field(..., description="Question to ask")
            top_k: int = Field(5, description="Number of context passages to retrieve")
            filter_dict: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
            search_type: str = Field("hybrid", description="Search type")
            conversation_id: Optional[str] = Field(None, description="Conversation ID")
        
        # Define endpoints
        @app.post("/documents/process")
        async def api_process_document(file: UploadFile = File(...), force_reprocess: bool = Form(False)):
            """Process a single document"""
            # Save uploaded file
            temp_file_path = Path(self.data_dir) / "uploads" / file.filename
            os.makedirs(temp_file_path.parent, exist_ok=True)
            
            with open(temp_file_path, "wb") as f:
                f.write(await file.read())
            
            # Process the document
            result = await self.process_document(temp_file_path, force_reprocess)
            
            if result.get("status") == "error":
                raise HTTPException(status_code=500, detail=result.get("error"))
            
            return result
        
        @app.post("/documents/process-directory")
        async def api_process_directory(directory: Optional[str] = None, force_reprocess: bool = False):
            """Process all documents in a directory"""
            result = await self.process_directory(directory, force_reprocess)
            
            if result.get("status") == "error":
                raise HTTPException(status_code=500, detail=result.get("error"))
            
            return result
        
        @app.post("/documents/index")
        async def api_index_documents(force_reindex: bool = False):
            """Index all processed documents"""
            result = await self.index_documents(force_reindex)
            
            if result.get("status") == "error":
                raise HTTPException(status_code=500, detail=result.get("error"))
            
            return result
        
        @app.post("/search")
        async def api_search(query: SearchQuery):
            """Search for documents"""
            result = await self.search(
                query=query.query,
                top_k=query.top_k,
                filter_dict=query.filter_dict,
                search_type=query.search_type,
                rerank_results=query.rerank_results
            )
            
            if result.get("status") == "error":
                raise HTTPException(status_code=500, detail=result.get("error"))
            
            return result
        
        @app.post("/ask")
        async def api_ask(query: QuestionQuery):
            """Ask a question using the RAG system"""
            result = await self.ask(
                question=query.question,
                top_k=query.top_k,
                filter_dict=query.filter_dict,
                search_type=query.search_type,
                conversation_id=query.conversation_id
            )
            
            if result.get("status") == "error":
                raise HTTPException(status_code=500, detail=result.get("error"))
            
            return result
        
        @app.get("/status")
        async def api_get_system_status():
            """Get the current system status"""
            result = await self.get_system_status()
            
            if result.get("status") == "error":
                raise HTTPException(status_code=500, detail=result.get("error"))
            
            return result
        
        @app.get("/documents/{document_id}")
        async def api_get_document_info(document_id: str):
            """Get information about a specific document"""
            result = await self.get_document_info(document_id)
            
            if result.get("status") == "error":
                raise HTTPException(status_code=404 if "not found" in result.get("error", "") else 500, 
                                 detail=result.get("error"))
            
            return result
        
        @app.get("/documents")
        async def api_list_documents(limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)):
            """List all documents in the system"""
            result = await self.list_documents(limit, offset)
            
            if result.get("status") == "error":
                raise HTTPException(status_code=500, detail=result.get("error"))
            
            return result
        
        return app

# Example usage in main.py
if __name__ == "__main__":
    # Create the orchestrator
    orchestrator = RAGOrchestrator()
    
    # Process documents in the data directory
    import asyncio
    
    async def main():
        # Process documents
        result = await orchestrator.process_directory()
        print(f"Processing result: {result['status']}")
        
        # Index documents
        result = await orchestrator.index_documents()
        print(f"Indexing result: {result['status']}")
        
        # Search for documents
        result = await orchestrator.search("vector database")
        print(f"Search result: {result['status']}, {result.get('result_count')} results")
        
        # Ask a question
        result = await orchestrator.ask("How does the vector indexer work?")
        print(f"Answer: {result.get('answer')[:100]}...")
    
    asyncio.run(main())