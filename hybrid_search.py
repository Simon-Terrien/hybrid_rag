from pathlib import Path
import time
from typing import List, Dict, Any, Optional, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

# Importation de la configuration et des utilitaires
from config import (
    PROCESSED_DIR, VECTOR_DB_DIR, SEARCH_CONFIG, 
    logger
)
from utils import (
    load_json, timed, log_exceptions
)

class HybridSearchEngine:
    """
    Moteur de recherche hybride combinant recherche sémantique et recherche lexicale.
    """
    
    def __init__(
        self,
        semantic_search_engine, 
        similarity_threshold: float = 0.5,
        semantic_weight: float = 0.7
    ):
        """
        Initialise le moteur de recherche hybride.
        
        Args:
            semantic_search_engine: Instance de SemanticSearchEngine
            similarity_threshold: Seuil minimal de similarité (0-1)
            semantic_weight: Poids de la recherche sémantique (0-1), le reste pour lexical
        """
        self.semantic_search = semantic_search_engine
        self.similarity_threshold = similarity_threshold
        self.semantic_weight = semantic_weight
        self.lexical_weight = 1.0 - semantic_weight
        
        # Initialisation des index lexicaux
        self.tfidf_vectorizer = None
        self.bm25_index = None
        self.document_texts = []
        self.document_ids = []
        
        # Construire les index lexicaux
        self._build_lexical_indices()
        
        logger.info(f"HybridSearchEngine initialisé: semantic_weight={semantic_weight}, "
                  f"lexical_weight={self.lexical_weight}")
    
    @timed(logger=logger)
    def _build_lexical_indices(self):
        """Construit les index lexicaux (TF-IDF et BM25) pour tous les documents"""
        # Charger les chunks de tous les documents
        documents = []
        document_ids = []
        document_texts = []
        
        # Parcourir le registre des documents via l'indexeur vectoriel
        for doc_path, doc_info in self.semantic_search.vector_indexer.document_registry["documents"].items():
            document_id = doc_info["document_id"]
            chunk_file_path = PROCESSED_DIR / doc_info["chunk_file"]
            
            if not chunk_file_path.exists():
                logger.warning(f"Fichier de chunks non trouvé: {chunk_file_path}")
                continue
            
            # Charger les chunks
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
                logger.error(f"Erreur lors du chargement des chunks de {document_id}: {str(e)}")
        
        if not document_texts:
            logger.warning("Aucun document à indexer pour la recherche lexicale")
            return
        
        # Construire l'index TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            max_df=0.85,
            min_df=2,
            ngram_range=(1, 2)
        )
        self.tfidf_vectorizer.fit(document_texts)
        
        # Construire l'index BM25
        # Tokenisation simple pour BM25
        tokenized_corpus = [text.lower().split() for text in document_texts]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        
        # Stocker les textes et ID pour la recherche
        self.document_texts = document_texts
        self.document_ids = document_ids
        
        logger.info(f"Index lexicaux construits: {len(document_texts)} chunks indexés")
    
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
        Effectue une recherche hybride (sémantique + lexicale).
        
        Args:
            query: Requête de recherche
            top_k: Nombre de résultats à retourner
            filters: Filtres de métadonnées
            return_all_chunks: Si True, retourne tous les chunks, sinon regroupe par document
            semantic_weight: Poids optionnel pour la partie sémantique (0-1)
            
        Returns:
            Résultats de la recherche hybride
        """
        start_time = time.time()
        
        # Ajuster le poids si nécessaire
        current_semantic_weight = semantic_weight if semantic_weight is not None else self.semantic_weight
        current_lexical_weight = 1.0 - current_semantic_weight
        
        # Effectuer la recherche sémantique
        semantic_results = self.semantic_search.search(
            query=query,
            top_k=top_k * 2,  # Récupérer plus de résultats pour la fusion
            filters=filters,
            return_all_chunks=True
        )
        
        # Effectuer la recherche lexicale (BM25 et TF-IDF)
        lexical_results = self._perform_lexical_search(query, top_k * 2)
        
        # Fusionner les résultats et réordonner par score combiné
        combined_results = self._combine_search_results(
            semantic_results["results"],
            lexical_results,
            current_semantic_weight,
            current_lexical_weight
        )
        
        # Filtrer par similarité minimale
        filtered_results = [result for result in combined_results if result["combined_score"] >= self.similarity_threshold]
        
        # Limiter aux top_k résultats
        final_results = filtered_results[:top_k]
        
        # Préparer les résultats
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
            # Retourner tous les chunks individuels
            search_results["results"] = final_results
        else:
            # Regrouper les résultats par document
            search_results["results"] = self._group_results_by_document(final_results)
        
        return search_results
    
    def _perform_lexical_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Effectue une recherche lexicale en combinant TF-IDF et BM25.
        
        Args:
            query: Requête de recherche
            top_k: Nombre de résultats à retourner
            
        Returns:
            Liste des résultats lexicaux
        """
        # Si aucun document n'a été indexé
        if not self.document_texts or not self.bm25_index or not self.tfidf_vectorizer:
            logger.warning("Aucun index lexical disponible pour la recherche")
            return []
        
        # Recherche BM25
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        # Normaliser les scores BM25 entre 0 et 1
        if max(bm25_scores) > 0:
            bm25_scores = bm25_scores / max(bm25_scores)
        
        # Recherche TF-IDF
        try:
            tfidf_query_vector = self.tfidf_vectorizer.transform([query])
            tfidf_doc_vectors = self.tfidf_vectorizer.transform(self.document_texts)
            
            # Calculer la similarité cosinus
            tfidf_scores = (tfidf_doc_vectors @ tfidf_query_vector.T).toarray().flatten()
        except Exception as e:
            logger.error(f"Erreur lors de la recherche TF-IDF: {str(e)}")
            tfidf_scores = np.zeros_like(bm25_scores)
        
        # Combiner les scores (moyenne pondérée)
        combined_lexical_scores = 0.6 * bm25_scores + 0.4 * tfidf_scores
        
        # Trier les documents par score et obtenir les indices des top_k
        top_indices = np.argsort(combined_lexical_scores)[::-1][:top_k]
        
        # Construire les résultats
        results = []
        for i, idx in enumerate(top_indices):
            if combined_lexical_scores[idx] > 0:  # Ignorer les documents sans correspondance
                # Récupérer les métadonnées via l'indexeur vectoriel
                chunk_id = self.document_ids[idx]
                document_id, chunk_index = chunk_id.split('-')
                
                # Rechercher le document dans le registre
                document_metadata = {}
                chunk_metadata = {}
                
                # Parcourir le registre pour trouver les métadonnées
                for doc_path, doc_info in self.semantic_search.vector_indexer.document_registry["documents"].items():
                    if doc_info["document_id"] == document_id:
                        document_metadata = doc_info["metadata"]
                        # Charger les métadonnées du chunk
                        chunk_file_path = PROCESSED_DIR / doc_info["chunk_file"]
                        try:
                            chunks_data = load_json(chunk_file_path)
                            for chunk in chunks_data:
                                if chunk["metadata"]["chunk_id"] == chunk_id:
                                    chunk_metadata = chunk["metadata"]
                                    break
                        except Exception as e:
                            logger.error(f"Erreur lors du chargement des métadonnées de {chunk_id}: {str(e)}")
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
        Combine les résultats sémantiques et lexicaux.
        
        Args:
            semantic_results: Résultats de la recherche sémantique
            lexical_results: Résultats de la recherche lexicale
            semantic_weight: Poids des résultats sémantiques
            lexical_weight: Poids des résultats lexicaux
            
        Returns:
            Liste des résultats combinés
        """
        # Créer un dictionnaire pour tous les chunks trouvés par les deux méthodes
        combined_chunks = {}
        
        # Ajouter les résultats sémantiques
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
        
        # Ajouter ou fusionner les résultats lexicaux
        for result in lexical_results:
            chunk_id = result["chunk_id"]
            if chunk_id in combined_chunks:
                # Fusionner avec un résultat sémantique existant
                combined_chunks[chunk_id]["lexical_score"] = result["lexical_score"]
                combined_chunks[chunk_id]["lexical_rank"] = result["rank"]
                # Mettre à jour le score combiné
                combined_chunks[chunk_id]["combined_score"] = (
                    semantic_weight * combined_chunks[chunk_id]["semantic_score"] +
                    lexical_weight * result["lexical_score"]
                )
            else:
                # Ajouter un nouveau résultat lexical
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
        
        # Convertir en liste et trier par score combiné
        combined_results = list(combined_chunks.values())
        combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # Ajouter le rang combiné
        for i, result in enumerate(combined_results):
            result["rank"] = i + 1
        
        return combined_results
    
    def _group_results_by_document(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Regroupe les résultats par document.
        
        Args:
            results: Liste des résultats individuels par chunk
            
        Returns:
            Liste des résultats regroupés par document
        """
        # Dictionnaire pour regrouper les chunks par document
        documents = {}
        
        for result in results:
            document_id = result["document_id"]
            
            if document_id not in documents:
                # Récupérer les métadonnées du document via le moteur de recherche sémantique
                doc_metadata = self.semantic_search._get_document_metadata(document_id)
                
                documents[document_id] = {
                    "document_id": document_id,
                    "metadata": doc_metadata,
                    "max_score": result["combined_score"],
                    "avg_score": result["combined_score"],
                    "chunks": [],
                    "best_rank": result["rank"]
                }
            
            # Mettre à jour les statistiques du document
            doc_entry = documents[document_id]
            doc_entry["max_score"] = max(doc_entry["max_score"], result["combined_score"])
            doc_entry["avg_score"] = (doc_entry["avg_score"] * len(doc_entry["chunks"]) + result["combined_score"]) / (len(doc_entry["chunks"]) + 1)
            doc_entry["best_rank"] = min(doc_entry["best_rank"], result["rank"])
            
            # Ajouter le chunk
            doc_entry["chunks"].append(result)
        
        # Convertir en liste et trier par score maximum
        grouped_results = list(documents.values())
        grouped_results.sort(key=lambda x: x["max_score"], reverse=True)
        
        return grouped_results
    
    def rebuild_indices(self):
        """Reconstruit les index lexicaux"""
        self._build_lexical_indices()