from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import time
from datetime import datetime
import json

# Composants LangChain pour la recherche
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Importation de la configuration et des utilitaires
from config import (
    PROCESSED_DIR, VECTOR_DB_DIR, EMBEDDING_MODEL, USE_GPU,
    VECTOR_DB_TYPE, SEARCH_CONFIG, DOCUMENT_REGISTRY_PATH,
    VECTOR_REGISTRY_PATH, logger
)
from utils import (
    load_json, timed, log_exceptions, retry,
    filter_complex_metadata
)

# Importer notre indexeur vectoriel pour réutiliser ses fonctions
from vector_indexer import VectorIndexer

class SemanticSearchEngine:
    """Moteur de recherche sémantique pour la base de connaissances"""
    
    def __init__(
        self,
        vector_indexer: Optional[VectorIndexer] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ):
        """
        Initialise le moteur de recherche sémantique.
        
        Args:
            vector_indexer: Instance de VectorIndexer (si None, en crée une nouvelle)
            top_k: Nombre de résultats à retourner (défaut: depuis SEARCH_CONFIG)
            similarity_threshold: Seuil de similarité minimal (défaut: depuis SEARCH_CONFIG)
        """
        # Si aucun indexeur n'est fourni, en créer un nouveau
        self.vector_indexer = vector_indexer or VectorIndexer()
        
        # Paramètres de recherche
        self.top_k = top_k or SEARCH_CONFIG.get("default_top_k", 5)
        self.similarity_threshold = similarity_threshold or SEARCH_CONFIG.get("similarity_threshold", 0.3)
        self.max_tokens_per_chunk = SEARCH_CONFIG.get("max_tokens_per_chunk", 1000)
        
        # Métadonnées à booster lors de la recherche
        self.metadata_boost = SEARCH_CONFIG.get("metadata_fields_boost", {})
        
        # Historique des recherches
        self.search_history = []
        
        logger.info(f"SemanticSearchEngine initialisé: top_k={self.top_k}, "
                  f"similarity_threshold={self.similarity_threshold}")
    
    @timed(logger=logger)
    @log_exceptions(logger=logger)
    def search(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        return_all_chunks: bool = False
    ) -> Dict[str, Any]:
        """
        Effectue une recherche sémantique.
        
        Args:
            query: Requête de recherche
            top_k: Nombre de résultats à retourner (si None, utilise self.top_k)
            filters: Filtres de métadonnées (e.g. {"source": "document1.pdf"})
            return_all_chunks: Si True, retourne tous les chunks, sinon regroupe par document
            
        Returns:
            Résultats de la recherche
        """
        start_time = time.time()
        k = top_k or self.top_k
        
        # Vérifier que l'indexeur vectoriel est initialisé
        if self.vector_indexer.vector_db is None:
            raise ValueError("La base vectorielle n'est pas initialisée")
        
        # Effectuer la recherche
        logger.info(f"Recherche: '{query}' (top_k={k}, filtres={filters})")
        
        # Adapter les filtres au format spécifique du backend
        backend_filters = self._adapt_filters_to_backend(filters)
        
        # Exécuter la recherche
        if self.vector_indexer.vector_db_type == "faiss":
            # FAISS ne supporte pas les filtres natifs, on filtre après
            raw_results = self.vector_indexer.vector_db.similarity_search_with_score(
                query, k=k * 3 if filters else k  # Récupérer plus de résultats pour le filtrage post-recherche
            )
            
            # Filtrer manuellement si nécessaire
            if filters:
                filtered_results = []
                for doc, score in raw_results:
                    if self._matches_filters(doc.metadata, filters):
                        filtered_results.append((doc, score))
                
                raw_results = filtered_results[:k]  # Limiter au top_k après filtrage
                
        else:  # chroma
            # Chroma supporte le filtrage natif
            raw_results = self.vector_indexer.vector_db.similarity_search_with_score(
                query, k=k, filter=backend_filters
            )
        
        # Normaliser les scores (0 à 1, où 1 est le meilleur)
        normalized_results = []
        for doc, score in raw_results:
            # Convertir la distance en similarité normalisée selon le backend
            if self.vector_indexer.vector_db_type == "faiss":
                # Pour FAISS avec cosine, la distance est entre 0 et 2, où 0 est le meilleur
                similarity = 1 - (score / 2)
            else:  # chroma
                # Pour Chroma, le score est déjà une similarité
                similarity = score
            
            # Appliquer le boost basé sur les métadonnées
            boosted_similarity = self._apply_metadata_boost(doc.metadata, similarity)
            
            # Filtrer basé sur le seuil de similarité
            if boosted_similarity >= self.similarity_threshold:
                normalized_results.append((doc, boosted_similarity))
        
        # Trier les résultats par score décroissant
        normalized_results.sort(key=lambda x: x[1], reverse=True)
        
        # Préparer les résultats
        search_results = {
            "query": query,
            "total_results": len(normalized_results),
            "processing_time_seconds": round(time.time() - start_time, 3),
            "filters": filters,
            "top_k": k
        }
        
        if return_all_chunks:
            # Retourner tous les chunks individuels
            search_results["results"] = [
                {
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "document_id": doc.metadata.get("document_id"),
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity": similarity,
                    "rank": i + 1
                }
                for i, (doc, similarity) in enumerate(normalized_results)
            ]
        else:
            # Regrouper les résultats par document
            search_results["results"] = self._group_results_by_document(normalized_results)
        
        # Ajouter à l'historique des recherches
        self.search_history.append({
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "num_results": len(normalized_results),
            "filters": filters,
            "top_k": k,
            "processing_time": search_results["processing_time_seconds"]
        })
        
        return search_results
    
    def _adapt_filters_to_backend(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Adapte les filtres au format spécifique du backend.
        
        Args:
            filters: Filtres de métadonnées génériques
            
        Returns:
            Filtres adaptés au backend spécifique
        """
        if not filters:
            return None
        
        if self.vector_indexer.vector_db_type == "faiss":
            # FAISS ne supporte pas les filtres natifs
            return None
        elif self.vector_indexer.vector_db_type == "chroma":
            # Format de filtre Chroma
            chroma_filters = {}
            
            for key, value in filters.items():
                # Gestion des valeurs multiples
                if isinstance(value, list):
                    chroma_filters[key] = {"$in": value}
                # Gestion des opérateurs de comparaison (préfixés par $ dans le filtre d'entrée)
                elif isinstance(value, dict) and all(k.startswith("$") for k in value.keys()):
                    chroma_filters[key] = value
                # Valeur simple (égalité)
                else:
                    chroma_filters[key] = {"$eq": value}
            
            return chroma_filters
        
        return filters  # Par défaut, retourner les filtres tels quels
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """
        Vérifie si les métadonnées correspondent aux filtres.
        
        Args:
            metadata: Métadonnées du document
            filters: Filtres à appliquer
            
        Returns:
            True si les métadonnées correspondent aux filtres, False sinon
        """
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            # Valeur simple (égalité)
            if not isinstance(value, (list, dict)):
                if metadata[key] != value:
                    return False
            
            # Liste de valeurs possibles
            elif isinstance(value, list):
                if metadata[key] not in value:
                    return False
            
            # Dictionnaire d'opérateurs
            elif isinstance(value, dict):
                for op, op_value in value.items():
                    if op == "$eq" and metadata[key] != op_value:
                        return False
                    elif op == "$ne" and metadata[key] == op_value:
                        return False
                    elif op == "$in" and metadata[key] not in op_value:
                        return False
                    elif op == "$nin" and metadata[key] in op_value:
                        return False
                    elif op == "$gt" and not (metadata[key] > op_value):
                        return False
                    elif op == "$gte" and not (metadata[key] >= op_value):
                        return False
                    elif op == "$lt" and not (metadata[key] < op_value):
                        return False
                    elif op == "$lte" and not (metadata[key] <= op_value):
                        return False
        
        return True
    
    def _apply_metadata_boost(self, metadata: Dict[str, Any], similarity: float) -> float:
        """
        Applique un boost au score de similarité basé sur les métadonnées.
        
        Args:
            metadata: Métadonnées du document
            similarity: Score de similarité initial (0-1)
            
        Returns:
            Score de similarité boosté
        """
        if not self.metadata_boost:
            return similarity
        
        boost_factor = 1.0
        
        for field, factor in self.metadata_boost.items():
            if field in metadata:
                # Le boost ne s'applique que si le champ est présent
                boost_factor *= factor
        
        # Limiter le boost pour éviter des scores > 1
        return min(similarity * boost_factor, 1.0)
    
    def _group_results_by_document(self, results: List[Tuple[Document, float]]) -> List[Dict[str, Any]]:
        """
        Regroupe les résultats par document.
        
        Args:
            results: Liste de tuples (document, score)
            
        Returns:
            Liste de résultats regroupés par document
        """
        # Dictionnaire pour regrouper les chunks par document
        documents = {}
        
        for i, (doc, similarity) in enumerate(results):
            document_id = doc.metadata.get("document_id")
            
            if document_id not in documents:
                # Récupérer les métadonnées du document depuis le registre
                doc_metadata = self._get_document_metadata(document_id)
                
                documents[document_id] = {
                    "document_id": document_id,
                    "metadata": doc_metadata,
                    "max_similarity": similarity,
                    "avg_similarity": similarity,
                    "chunks": [],
                    "best_rank": i + 1
                }
            
            # Mettre à jour les statistiques du document
            doc_entry = documents[document_id]
            doc_entry["max_similarity"] = max(doc_entry["max_similarity"], similarity)
            doc_entry["avg_similarity"] = (doc_entry["avg_similarity"] * len(doc_entry["chunks"]) + similarity) / (len(doc_entry["chunks"]) + 1)
            doc_entry["best_rank"] = min(doc_entry["best_rank"], i + 1)
            
            # Ajouter le chunk
            doc_entry["chunks"].append({
                "chunk_id": doc.metadata.get("chunk_id"),
                "text": doc.page_content,
                "metadata": doc.metadata,
                "similarity": similarity,
                "rank": i + 1
            })
        
        # Convertir en liste et trier par similarité maximale
        grouped_results = list(documents.values())
        grouped_results.sort(key=lambda x: x["max_similarity"], reverse=True)
        
        return grouped_results
    
    def _get_document_metadata(self, document_id: str) -> Dict[str, Any]:
        """
        Récupère les métadonnées d'un document.
        
        Args:
            document_id: Identifiant du document
            
        Returns:
            Métadonnées du document
        """
        # Parcourir le registre des documents pour trouver les métadonnées
        for doc_path, doc_info in self.vector_indexer.document_registry["documents"].items():
            if doc_info["document_id"] == document_id:
                return doc_info["metadata"]
        
        # Si non trouvé, retourner un dictionnaire vide
        return {}
    
    @timed(logger=logger)
    def get_document_context(self, document_id: str, chunk_id: Optional[str] = None, window_size: int = 3) -> Dict[str, Any]:
        """
        Récupère le contexte autour d'un chunk ou d'un document.
        
        Args:
            document_id: Identifiant du document
            chunk_id: Identifiant du chunk (si None, retourne tout le document)
            window_size: Nombre de chunks avant et après à inclure
            
        Returns:
            Contexte du document
        """
        # Trouver le fichier de chunks
        document_entry = None
        for doc_path, doc_info in self.vector_indexer.document_registry["documents"].items():
            if doc_info["document_id"] == document_id:
                document_entry = doc_info
                break
        
        if not document_entry:
            raise ValueError(f"Document non trouvé avec l'ID: {document_id}")
        
        chunk_file_path = PROCESSED_DIR / document_entry["chunk_file"]
        if not chunk_file_path.exists():
            raise FileNotFoundError(f"Fichier de chunks non trouvé: {chunk_file_path}")
        
        # Charger tous les chunks du document
        with open(chunk_file_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        # Si aucun chunk spécifique n'est demandé, retourner tout le document
        if not chunk_id:
            return {
                "document_id": document_id,
                "metadata": document_entry["metadata"],
                "chunks": chunks_data,
                "total_chunks": len(chunks_data)
            }
        
        # Trouver l'index du chunk spécifié
        chunk_index = -1
        for i, chunk in enumerate(chunks_data):
            if chunk["metadata"]["chunk_id"] == chunk_id:
                chunk_index = i
                break
        
        if chunk_index == -1:
            raise ValueError(f"Chunk non trouvé avec l'ID: {chunk_id}")
        
        # Calculer les indices de début et de fin pour le contexte
        start_index = max(0, chunk_index - window_size)
        end_index = min(len(chunks_data), chunk_index + window_size + 1)
        
        # Extraire le contexte
        context_chunks = chunks_data[start_index:end_index]
        
        return {
            "document_id": document_id,
            "metadata": document_entry["metadata"],
            "central_chunk_id": chunk_id,
            "central_chunk_index": chunk_index,
            "window_size": window_size,
            "context": context_chunks,
            "total_chunks": len(chunks_data)
        }
    
    def get_search_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Récupère l'historique des recherches.
        
        Args:
            limit: Nombre maximum d'entrées à retourner
            
        Returns:
            Historique des recherches
        """
        return self.search_history[-limit:]
    
    def clear_search_history(self) -> None:
        """Efface l'historique des recherches."""
        self.search_history = []
        logger.info("Historique des recherches effacé")

if __name__ == "__main__":
    # Initialiser le processeur de documents pour traiter le PDF
    from DocumentProcessor import DocumentProcessor
    
    # Traiter le document PDF
    processor = DocumentProcessor()
    
    # Si votre PDF est dans le répertoire data, indexez-le
    pdf_results = processor.process_directory()
    print(f"Documents traités: {pdf_results['processed_documents']}")
    
    # Initialiser l'indexeur et forcer la réindexation
    indexer = VectorIndexer()
    index_results = indexer.index_all_documents(force_reindex=True)
    print(f"Indexation terminée: {index_results['indexed_documents']} documents indexés")
    
    # Afficher les statistiques
    stats = indexer.get_stats()
    print(f"Total des documents: {stats['total_documents']}")
    print(f"Total des chunks: {stats['total_chunks']}")
    
    # Maintenant faire la recherche
    search_engine = SemanticSearchEngine(similarity_threshold=0.5)
    results = search_engine.search("parc informatique", top_k=3)
    
    print(f"Requête: '{results['query']}'")
    print(f"Temps de traitement: {results['processing_time_seconds']} secondes")
    print(f"Nombre de résultats: {results['total_results']}")
    
    # Afficher les résultats regroupés par document
    for i, doc in enumerate(results['results']):
        print(f"\nDocument {i+1}: {doc['document_id']}")
        print(f"  Similarité max: {doc['max_similarity']:.4f}")
        print(f"  Nombre de chunks: {len(doc['chunks'])}")
        print(f"  Premier chunk: {doc['chunks'][0]['text'][:100]}...")