from pathlib import Path
import time
from typing import List, Dict, Any, Optional, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

# Import configuration and utilities
from config import (
    PROCESSED_DIR, VECTOR_DB_DIR, SEARCH_CONFIG, 
    logger
)
from utils import (
    load_json, timed, log_exceptions
)

class HybridSearchEngine:
    """
    Hybrid search engine combining semantic and lexical search.
    """
    
    def __init__(
        self,
        semantic_search_engine, 
        similarity_threshold: float = 0.2,
        semantic_weight: float = 0.5
    ):
        """
        Initialize the hybrid search engine.
        
        Args:
            semantic_search_engine: Instance of SemanticSearchEngine
            similarity_threshold: Minimum similarity threshold (0-1)
            semantic_weight: Weight of semantic search (0-1), the rest for lexical
        """
        self.semantic_search = semantic_search_engine
        self.similarity_threshold = similarity_threshold
        self.semantic_weight = semantic_weight
        self.lexical_weight = 1.0 - semantic_weight
        
        # Initialize lexical indices
        self.tfidf_vectorizer = None
        self.bm25_index = None
        self.document_texts = []
        self.document_ids = []
        
        # Build lexical indices
        self._build_lexical_indices()
        
        logger.info(f"HybridSearchEngine initialized: semantic_weight={semantic_weight}, "
                  f"lexical_weight={self.lexical_weight}")
    
    @timed(logger=logger)
    def _build_lexical_indices(self):
        """Builds lexical indices (TF-IDF and BM25) for all documents"""
        # Load chunks from all documents
        documents = []
        document_ids = []
        document_texts = []
        
        # Traverse the document registry via the vector indexer
        for doc_path, doc_info in self.semantic_search.vector_indexer.document_registry["documents"].items():
            document_id = doc_info["document_id"]
            chunk_file_path = PROCESSED_DIR / doc_info["chunk_file"]
            
            if not chunk_file_path.exists():
                logger.warning(f"Chunk file not found: {chunk_file_path}")
                continue
            
            # Load chunks
            try:
                chunks_data = load_json(chunk_file_path)
                
                for chunk in chunks_data:
                    documents.append({
                        "text": chunk["text"],
                        "metadata": chunk["metadata"]
                    })
                    document_ids.append(chunk["metadata"]["chunk_id"])
                    document_texts.append(chunk["text"])
            except Exception as e:
                logger.error(f"Error loading chunks for {document_id}: {str(e)}")
        
        if not document_texts:
            logger.warning("No documents to index for lexical search")
            return
        
        # Build TF-IDF index
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            max_df=0.85,
            min_df=2,
            ngram_range=(1, 2)
        )
        self.tfidf_vectorizer.fit(document_texts)
        
        # Build BM25 index
        # Simple tokenization for BM25
        tokenized_corpus = [text.lower().split() for text in document_texts]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        
        # Store texts and IDs for search
        self.document_texts = document_texts
        self.document_ids = document_ids
        
        logger.info(f"Lexical indices built: {len(document_texts)} chunks indexed")
    
    @timed(logger=logger)
    @log_exceptions(logger=logger)
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        return_all_chunks: bool = False,
        semantic_weight: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Performs a hybrid search (semantic + lexical).
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Metadata filters
            return_all_chunks: If True, returns all chunks, otherwise groups by document
            semantic_weight: Optional weight for semantic component (0-1)
            
        Returns:
            Hybrid search results
        """
        start_time = time.time()
        
        # Adjust weight if necessary
        current_semantic_weight = semantic_weight if semantic_weight is not None else self.semantic_weight
        current_lexical_weight = 1.0 - current_semantic_weight
        
        # Perform semantic search
        semantic_results = self.semantic_search.search(
            query=query,
            top_k=top_k * 2,  # Get more results for fusion
            filters=filters,
            return_all_chunks=True
        )
        
        # Perform lexical search (BM25 and TF-IDF)
        lexical_results = self._perform_lexical_search(query, top_k * 2)
        
        # Merge results and reorder by combined score
        combined_results = self._combine_search_results(
            semantic_results["results"],
            lexical_results,
            current_semantic_weight,
            current_lexical_weight
        )
        
        # Filter by minimum similarity
        filtered_results = [result for result in combined_results if result["combined_score"] >= self.similarity_threshold]
        
        # Limit to top_k results
        final_results = filtered_results[:top_k]
        
        # Prepare results
        search_results = {
            "query": query,
            "total_results": len(final_results),
            "processing_time_seconds": round(time.time() - start_time, 3),
            "filters": filters,
            "top_k": top_k,
            "semantic_weight": current_semantic_weight,
            "lexical_weight": current_lexical_weight
        }
        
        if return_all_chunks:
            # Return all individual chunks
            search_results["results"] = final_results
        else:
            # Group results by document
            search_results["results"] = self._group_results_by_document(final_results)
        
        return search_results
    
    def _perform_lexical_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Performs lexical search combining TF-IDF and BM25.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of lexical search results
        """
        # If no documents have been indexed
        if not self.document_texts or not self.bm25_index or not self.tfidf_vectorizer:
            logger.warning("No lexical index available for search")
            return []
        
        # BM25 search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        # Normalize BM25 scores between 0 and 1
        if max(bm25_scores) > 0:
            bm25_scores = bm25_scores / max(bm25_scores)
        
        # TF-IDF search
        try:
            tfidf_query_vector = self.tfidf_vectorizer.transform([query])
            tfidf_doc_vectors = self.tfidf_vectorizer.transform(self.document_texts)
            
            # Calculate cosine similarity
            tfidf_scores = (tfidf_doc_vectors @ tfidf_query_vector.T).toarray().flatten()
        except Exception as e:
            logger.error(f"Error during TF-IDF search: {str(e)}")
            tfidf_scores = np.zeros_like(bm25_scores)
        
        # Combine scores (weighted average)
        combined_lexical_scores = 0.6 * bm25_scores + 0.4 * tfidf_scores
        
        # Sort documents by score and get indices of top_k
        top_indices = np.argsort(combined_lexical_scores)[::-1][:top_k]
        
        # Build results
        results = []
        for i, idx in enumerate(top_indices):
            if combined_lexical_scores[idx] > 0:  # Ignore documents without matches
                # Get metadata via vector indexer
                chunk_id = self.document_ids[idx]
                document_id, chunk_index = chunk_id.split('-')
                
                # Search document in registry
                document_metadata = {}
                chunk_metadata = {}
                
                # Traverse registry to find metadata
                for doc_path, doc_info in self.semantic_search.vector_indexer.document_registry["documents"].items():
                    if doc_info["document_id"] == document_id:
                        document_metadata = doc_info["metadata"]
                        # Load chunk metadata
                        chunk_file_path = PROCESSED_DIR / doc_info["chunk_file"]
                        try:
                            chunks_data = load_json(chunk_file_path)
                            for chunk in chunks_data:
                                if chunk["metadata"]["chunk_id"] == chunk_id:
                                    chunk_metadata = chunk["metadata"]
                                    break
                        except Exception as e:
                            logger.error(f"Error loading metadata for {chunk_id}: {str(e)}")
                        break
                
                results.append({
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "text": self.document_texts[idx],
                    "metadata": chunk_metadata,
                    "lexical_score": float(combined_lexical_scores[idx]),
                    "bm25_score": float(bm25_scores[idx]),
                    "tfidf_score": float(tfidf_scores[idx]),
                    "rank": i + 1
                })
        
        return results
    
    def _combine_search_results(
        self,
        semantic_results: List[Dict[str, Any]],
        lexical_results: List[Dict[str, Any]],
        semantic_weight: float,
        lexical_weight: float
    ) -> List[Dict[str, Any]]:
        """
        Combines semantic and lexical search results.
        
        Args:
            semantic_results: Semantic search results
            lexical_results: Lexical search results
            semantic_weight: Weight for semantic results
            lexical_weight: Weight for lexical results
            
        Returns:
            List of combined results
        """
        # Create dictionary for all chunks found by both methods
        combined_chunks = {}
        
        # Add semantic results
        for result in semantic_results:
            chunk_id = result["chunk_id"]
            combined_chunks[chunk_id] = {
                "chunk_id": chunk_id,
                "document_id": result["document_id"],
                "text": result["text"],
                "metadata": result["metadata"],
                "semantic_score": result["similarity"],
                "lexical_score": 0.0,
                "combined_score": semantic_weight * result["similarity"],
                "semantic_rank": result["rank"],
                "lexical_rank": None
            }
        
        # Add or merge lexical results
        for result in lexical_results:
            chunk_id = result["chunk_id"]
            if chunk_id in combined_chunks:
                # Merge with existing semantic result
                combined_chunks[chunk_id]["lexical_score"] = result["lexical_score"]
                combined_chunks[chunk_id]["lexical_rank"] = result["rank"]
                # Update combined score
                combined_chunks[chunk_id]["combined_score"] = (
                    semantic_weight * combined_chunks[chunk_id]["semantic_score"] +
                    lexical_weight * result["lexical_score"]
                )
            else:
                # Add new lexical result
                combined_chunks[chunk_id] = {
                    "chunk_id": chunk_id,
                    "document_id": result["document_id"],
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "semantic_score": 0.0,
                    "lexical_score": result["lexical_score"],
                    "combined_score": lexical_weight * result["lexical_score"],
                    "semantic_rank": None,
                    "lexical_rank": result["rank"]
                }
        
        # Convert to list and sort by combined score
        combined_results = list(combined_chunks.values())
        combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # Add combined rank
        for i, result in enumerate(combined_results):
            result["rank"] = i + 1
        
        return combined_results
    
    def _group_results_by_document(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Groups results by document.
        
        Args:
            results: List of individual chunk results
            
        Returns:
            List of results grouped by document
        """
        # Dictionary to group chunks by document
        documents = {}
        
        for result in results:
            document_id = result["document_id"]
            
            if document_id not in documents:
                # Get document metadata via semantic search engine
                doc_metadata = self.semantic_search._get_document_metadata(document_id)
                
                documents[document_id] = {
                    "document_id": document_id,
                    "metadata": doc_metadata,
                    "max_score": result["combined_score"],
                    "avg_score": result["combined_score"],
                    "chunks": [],
                    "best_rank": result["rank"]
                }
            
            # Update document statistics
            doc_entry = documents[document_id]
            doc_entry["max_score"] = max(doc_entry["max_score"], result["combined_score"])
            doc_entry["avg_score"] = (doc_entry["avg_score"] * len(doc_entry["chunks"]) + result["combined_score"]) / (len(doc_entry["chunks"]) + 1)
            doc_entry["best_rank"] = min(doc_entry["best_rank"], result["rank"])
            
            # Add the chunk
            doc_entry["chunks"].append(result)
        
        # Convert to list and sort by maximum score
        grouped_results = list(documents.values())
        grouped_results.sort(key=lambda x: x["max_score"], reverse=True)
        
        return grouped_results
    
    def rebuild_indices(self):
        """Rebuilds the lexical indices"""
        self._build_lexical_indices()
        
    def get_document_context(self, document_id: str, chunk_id: Optional[str] = None, window_size: int = 3):
        """
        Delegate document context retrieval to the semantic search engine
        
        Args:
            document_id: Document ID
            chunk_id: Chunk ID
            window_size: Context window size
            
        Returns:
            Document context
        """
        return self.semantic_search.get_document_context(document_id, chunk_id, window_size)