from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import time
from datetime import datetime
import json
import logging
import traceback
import concurrent.futures
import queue
import threading

# Import des composants du système
from enhanced_document_processor import EnhancedDocumentProcessor
from vector_indexer import VectorIndexer
from semantic_search import SemanticSearchEngine
from hybrid_search import HybridSearchEngine
from contextual_reranker import ContextualReranker
from DocumentChat import DocumentChat

# Importation de la configuration et des utilitaires
from config import (
    PROCESSED_DIR, VECTOR_DB_DIR, DATA_DIR, 
    MAX_CONCURRENT_TASKS, logger,
    get_config, save_config
)
from utils import (
    load_json, save_json, timed, log_exceptions
)

class RAGPipelineOrchestrator:
    """
    Orchestrateur pour le pipeline RAG complet.
    Coordonne tous les composants du système et gère leur interaction.
    """
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
        vector_db_dir: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialise l'orchestrateur du pipeline RAG.
        
        Args:
            data_dir: Répertoire des données brutes
            processed_dir: Répertoire des données traitées
            vector_db_dir: Répertoire de la base vectorielle
            config: Configuration personnalisée
        """
        # Charger la configuration
        self.config = config or get_config()
        
        # Définir les chemins
        self.data_dir = data_dir or Path(self.config["paths"]["data_dir"])
        self.processed_dir = processed_dir or Path(self.config["paths"]["processed_dir"])
        self.vector_db_dir = vector_db_dir or Path(self.config["paths"]["vector_db_dir"])
        
        # Créer les répertoires nécessaires
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.processed_dir.mkdir(exist_ok=True, parents=True)
        self.vector_db_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialiser les composants (création différée)
        self._document_processor = None
        self._vector_indexer = None
        self._semantic_search = None
        self._hybrid_search = None
        self._contextual_reranker = None
        self._document_chat = None
        
        # État du pipeline
        self.pipeline_state = {
            "status": "initialized",
            "last_update": datetime.now().isoformat(),
            "components": {
                "document_processor": {"status": "not_initialized"},
                "vector_indexer": {"status": "not_initialized"},
                "search_engine": {"status": "not_initialized"},
                "reranker": {"status": "not_initialized"},
                "document_chat": {"status": "not_initialized"}
            },
            "statistics": {
                "total_documents": 0,
                "total_chunks": 0,
                "total_queries": 0,
                "last_processing_time": None,
                "last_indexing_time": None
            }
        }
        
        # File d'attente pour les tâches asynchrones
        self.task_queue = queue.Queue()
        self.worker_threads = []
        self.max_workers = MAX_CONCURRENT_TASKS
        
        # Démarrer les threads de travail
        self._start_worker_threads()
        
        logger.info(f"RAGPipelineOrchestrator initialisé")
    
    def _start_worker_threads(self):
        """Démarre les threads de travail pour les tâches asynchrones"""
        for i in range(self.max_workers):
            thread = threading.Thread(target=self._worker_loop, daemon=True)
            thread.start()
            self.worker_threads.append(thread)
            logger.debug(f"Thread de travail {i+1} démarré")
    
    def _worker_loop(self):
        """Boucle d'exécution pour les threads de travail"""
        while True:
            try:
                task, args, kwargs, result_callback = self.task_queue.get()
                
                try:
                    result = task(*args, **kwargs)
                    if result_callback:
                        result_callback(result)
                except Exception as e:
                    logger.error(f"Erreur dans la tâche asynchrone: {str(e)}")
                    traceback.print_exc()
                
                self.task_queue.task_done()
            except Exception as e:
                logger.error(f"Erreur dans le thread de travail: {str(e)}")
                traceback.print_exc()
    
    def _update_pipeline_state(self, component: str, status: str, **kwargs):
        """Met à jour l'état du pipeline"""
        self.pipeline_state["components"][component]["status"] = status
        self.pipeline_state["components"][component].update(kwargs)
        self.pipeline_state["last_update"] = datetime.now().isoformat()
        
        # Mettre à jour l'état global
        any_failed = any(comp["status"] == "failed" for comp in self.pipeline_state["components"].values())
        all_ready = all(comp["status"] == "ready" for comp in self.pipeline_state["components"].values())
        
        if any_failed:
            self.pipeline_state["status"] = "partially_failed"
        elif all_ready:
            self.pipeline_state["status"] = "ready"
        else:
            self.pipeline_state["status"] = "initializing"
    
    @property
    def document_processor(self) -> EnhancedDocumentProcessor:
        """Initialise et retourne le processeur de documents"""
        if self._document_processor is None:
            try:
                logger.info("Initialisation du processeur de documents amélioré")
                self._document_processor = EnhancedDocumentProcessor(
                    data_dir=self.data_dir,
                    processed_dir=self.processed_dir
                )
                self._update_pipeline_state("document_processor", "ready")
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation du processeur de documents: {str(e)}")
                self._update_pipeline_state("document_processor", "failed", error=str(e))
                raise
        
        return self._document_processor
    
    @property
    def vector_indexer(self) -> VectorIndexer:
        """Initialise et retourne l'indexeur vectoriel"""
        if self._vector_indexer is None:
            try:
                logger.info("Initialisation de l'indexeur vectoriel")
                self._vector_indexer = VectorIndexer(
                    processed_dir=self.processed_dir,
                    vector_db_dir=self.vector_db_dir,
                    embedding_model_name=self.config["embedding"]["model"],
                    vector_db_type=self.config["vector_db"]["type"]
                )
                self._update_pipeline_state("vector_indexer", "ready")
                
                # Mettre à jour les statistiques
                stats = self._vector_indexer.get_stats()
                self.pipeline_state["statistics"]["total_documents"] = stats["total_documents"]
                self.pipeline_state["statistics"]["total_chunks"] = stats["total_chunks"]
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation de l'indexeur vectoriel: {str(e)}")
                self._update_pipeline_state("vector_indexer", "failed", error=str(e))
                raise
        
        return self._vector_indexer
    
    @property
    def semantic_search(self) -> SemanticSearchEngine:
        """Initialise et retourne le moteur de recherche sémantique"""
        if self._semantic_search is None:
            try:
                # S'assurer que l'indexeur est initialisé
                _ = self.vector_indexer
                
                logger.info("Initialisation du moteur de recherche sémantique")
                self._semantic_search = SemanticSearchEngine(
                    vector_indexer=self.vector_indexer,
                    top_k=self.config["search"]["default_top_k"],
                    similarity_threshold=self.config["search"]["similarity_threshold"]
                )
                self._update_pipeline_state("search_engine", "initializing")
                
                # Initialiser la recherche hybride par-dessus la recherche sémantique
                logger.info("Initialisation du moteur de recherche hybride")
                self._hybrid_search = HybridSearchEngine(
                    semantic_search_engine=self._semantic_search,
                    similarity_threshold=self.config["search"]["similarity_threshold"],
                    semantic_weight=0.7  # Paramètre par défaut
                )
                
                self._update_pipeline_state("search_engine", "ready", type="hybrid")
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation du moteur de recherche: {str(e)}")
                self._update_pipeline_state("search_engine", "failed", error=str(e))
                raise
        
        return self._semantic_search
    
    @property
    def hybrid_search(self) -> HybridSearchEngine:
        """Initialise et retourne le moteur de recherche hybride"""
        if self._hybrid_search is None:
            # Initialiser le moteur de recherche sémantique qui créera aussi le moteur hybride
            _ = self.semantic_search
        
        return self._hybrid_search
    
    @property
    def contextual_reranker(self) -> ContextualReranker:
        """Initialise et retourne le réordonnateur contextuel"""
        if self._contextual_reranker is None:
            try:
                logger.info("Initialisation du réordonnateur contextuel")
                self._contextual_reranker = ContextualReranker(
                    enabled=self.config["search"].get("reranking_enabled", False)
                )
                
                status = "ready" if self._contextual_reranker.enabled else "disabled"
                self._update_pipeline_state("reranker", status)
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation du réordonnateur: {str(e)}")
                self._update_pipeline_state("reranker", "failed", error=str(e))
                raise
        
        return self._contextual_reranker
    
    @property
    def document_chat(self) -> DocumentChat:
        """Initialise et retourne l'interface de dialogue avec les documents"""
        if self._document_chat is None:
            try:
                # S'assurer que le moteur de recherche est initialisé
                _ = self.hybrid_search
                
                logger.info("Initialisation de l'interface de dialogue documentaire")
                self._document_chat = DocumentChat(
                    search_engine=self.hybrid_search
                )
                self._update_pipeline_state("document_chat", "ready")
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation de l'interface de dialogue: {str(e)}")
                self._update_pipeline_state("document_chat", "failed", error=str(e))
                raise
        
        return self._document_chat
    
    @timed(logger=logger)
    @log_exceptions(logger=logger)
    def process_documents(
        self, 
        subdirectory: Optional[str] = None, 
        force_reprocess: bool = False,
        async_mode: bool = True
    ) -> Union[Dict[str, Any], str]:
        """
        Traite les documents dans le répertoire de données.
        
        Args:
            subdirectory: Sous-répertoire optionnel à traiter
            force_reprocess: Force le retraitement même si les documents sont déjà traités
            async_mode: Si True, exécute la tâche en arrière-plan
            
        Returns:
            Résultats du traitement ou ID de la tâche si async_mode=True
        """
        if async_mode:
            task_id = f"process_{int(time.time())}"
            
            def callback(result):
                # Mettre à jour les statistiques
                self.pipeline_state["statistics"]["last_processing_time"] = datetime.now().isoformat()
                self.pipeline_state["statistics"]["total_documents"] = len(self.document_processor.document_registry["documents"])
                logger.info(f"Traitement des documents terminé: {result['processed_documents']} documents traités")
            
            self.task_queue.put((
                self.document_processor.process_directory,
                (subdirectory,),
                {"force_reprocess": force_reprocess},
                callback
            ))
            
            return {"task_id": task_id, "status": "processing", "message": "Traitement des documents en cours"}
        else:
            result = self.document_processor.process_directory(
                subdirectory=subdirectory,
                force_reprocess=force_reprocess
            )
            
            # Mettre à jour les statistiques
            self.pipeline_state["statistics"]["last_processing_time"] = datetime.now().isoformat()
            self.pipeline_state["statistics"]["total_documents"] = len(self.document_processor.document_registry["documents"])
            
            return result
    
    @timed(logger=logger)
    @log_exceptions(logger=logger)
    def index_documents(
        self, 
        force_reindex: bool = False,
        async_mode: bool = True
    ) -> Union[Dict[str, Any], str]:
        """
        Indexe les documents traités dans la base vectorielle.
        
        Args:
            force_reindex: Force la réindexation même si les documents sont déjà indexés
            async_mode: Si True, exécute la tâche en arrière-plan
            
        Returns:
            Résultats de l'indexation ou ID de la tâche si async_mode=True
        """
        if async_mode:
            task_id = f"index_{int(time.time())}"
            
            def callback(result):
                # Mettre à jour les statistiques
                stats = self.vector_indexer.get_stats()
                self.pipeline_state["statistics"]["last_indexing_time"] = datetime.now().isoformat()
                self.pipeline_state["statistics"]["total_documents"] = stats["total_documents"]
                self.pipeline_state["statistics"]["total_chunks"] = stats["total_chunks"]
                logger.info(f"Indexation terminée: {result['indexed_documents']} documents indexés")
                
                # Reconstruire les index lexicaux pour la recherche hybride
                if self._hybrid_search:
                    self._hybrid_search.rebuild_indices()
            
            self.task_queue.put((
                self.vector_indexer.index_all_documents,
                (),
                {"force_reindex": force_reindex},
                callback
            ))
            
            return {"task_id": task_id, "status": "indexing", "message": "Indexation des documents en cours"}
        else:
            result = self.vector_indexer.index_all_documents(force_reindex=force_reindex)
            
            # Mettre à jour les statistiques
            stats = self.vector_indexer.get_stats()
            self.pipeline_state["statistics"]["last_indexing_time"] = datetime.now().isoformat()
            self.pipeline_state["statistics"]["total_documents"] = stats["total_documents"]
            self.pipeline_state["statistics"]["total_chunks"] = stats["total_chunks"]
            
            # Reconstruire les index lexicaux pour la recherche hybride
            if self._hybrid_search:
                self._hybrid_search.rebuild_indices()
            
            return result
    
    @timed(logger=logger)
    @log_exceptions(logger=logger)
    def search(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        search_type: str = "hybrid",
        semantic_weight: Optional[float] = None,
        rerank_results: Optional[bool] = None,
        return_all_chunks: bool = False
    ) -> Dict[str, Any]:
        """
        Effectue une recherche dans la base de connaissances.
        
        Args:
            query: Requête de recherche
            top_k: Nombre de résultats à retourner (si None, utilise la valeur par défaut)
            filters: Filtres de métadonnées
            search_type: Type de recherche ("semantic", "hybrid")
            semantic_weight: Poids pour la partie sémantique (0-1)
            rerank_results: Si True, réordonne les résultats
            return_all_chunks: Si True, retourne tous les chunks individuels
            
        Returns:
            Résultats de la recherche
        """
        # Initialiser les moteurs de recherche si nécessaire
        _ = self.hybrid_search
        _ = self.contextual_reranker
        
        # Paramètres par défaut
        if top_k is None:
            top_k = self.config["search"]["default_top_k"]
        
        if rerank_results is None:
            rerank_results = self.config["search"].get("reranking_enabled", False)
        
        # Effectuer la recherche appropriée
        if search_type == "semantic":
            results = self.semantic_search.search(
                query=query,
                top_k=top_k,
                filters=filters,
                return_all_chunks=True  # Toujours retourner les chunks pour le réordonnancement
            )
        else:  # hybrid
            results = self.hybrid_search.search(
                query=query,
                top_k=top_k,
                filters=filters,
                return_all_chunks=True,  # Toujours retourner les chunks pour le réordonnancement
                semantic_weight=semantic_weight
            )
        
        # Réordonner les résultats si nécessaire
        if rerank_results and self.contextual_reranker.enabled:
            reranked_results = self.contextual_reranker.rerank(
                query=query,
                results=results["results"],
                top_k=top_k
            )
            results["results"] = reranked_results
            results["reranked"] = True
        else:
            results["reranked"] = False
        
        # Regrouper par document si nécessaire
        if not return_all_chunks:
            if search_type == "semantic":
                grouped_results = self.semantic_search._group_results_by_document(
                    [(doc, doc["similarity"]) for doc in results["results"]]
                )
            else:  # hybrid
                grouped_results = self.hybrid_search._group_results_by_document(results["results"])
            
            results["results"] = grouped_results
        
        # Mettre à jour les statistiques
        self.pipeline_state["statistics"]["total_queries"] += 1
        
        return results
    
    @timed(logger=logger)
    @log_exceptions(logger=logger)
    def ask(
        self, 
        question: str, 
        conversation_id: Optional[str] = None,
        search_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Pose une question au système et obtient une réponse basée sur les documents.
        
        Args:
            question: Question à poser
            conversation_id: Identifiant de la conversation
            search_params: Paramètres de recherche personnalisés
            
        Returns:
            Réponse générée avec les sources
        """
        # Initialiser l'interface de dialogue
        _ = self.document_chat
        
        # Paramètres de recherche par défaut
        default_search_params = {
            "top_k": self.config["search"]["default_top_k"],
            "search_type": "hybrid",
            "rerank_results": self.config["search"].get("reranking_enabled", False)
        }
        
        # Fusionner avec les paramètres personnalisés
        if search_params:
            default_search_params.update(search_params)
        
        # Poser la question
        response = self.document_chat.ask(
            question=question,
            conversation_id=conversation_id
        )
        
        return response
    
    def get_state(self) -> Dict[str, Any]:
        """Retourne l'état actuel du pipeline"""
        # Mettre à jour les statistiques si les composants sont initialisés
        if self._document_processor:
            self.pipeline_state["statistics"]["total_documents"] = len(self.document_processor.document_registry["documents"])
        
        if self._vector_indexer:
            stats = self.vector_indexer.get_stats()
            self.pipeline_state["statistics"]["total_chunks"] = stats["total_chunks"]
        
        return self.pipeline_state
    
    def configure(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Met à jour la configuration du pipeline.
        
        Args:
            config_updates: Mises à jour de la configuration
            
        Returns:
            Configuration complète mise à jour
        """
        # Fonction récursive pour mettre à jour la configuration
        def update_config(current_config, updates):
            for key, value in updates.items():
                if isinstance(value, dict) and key in current_config and isinstance(current_config[key], dict):
                    update_config(current_config[key], value)
                else:
                    current_config[key] = value
        
        # Appliquer les mises à jour
        update_config(self.config, config_updates)
        
        # Sauvegarder la configuration
        save_config(self.config)
        
        logger.info("Configuration du pipeline mise à jour")
        return self.config
    
    def cleanup(self):
        """Nettoie les ressources du pipeline"""
        # Arrêter les threads de travail
        for _ in self.worker_threads:
            self.task_queue.put((lambda: None, (), {}, None))
        
        for thread in self.worker_threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        
        logger.info("Ressources du pipeline nettoyées")