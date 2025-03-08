from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import numpy as np
from sentence_transformers import CrossEncoder

# Importation de la configuration et des utilitaires
from config import (
    SEARCH_CONFIG, logger,CROSS_ENCODER_MODELS
)
from utils import timed, log_exceptions

class ContextualReranker:
    """
    Réordonnateur contextuel pour améliorer la pertinence des résultats de recherche.
    Utilise un modèle Cross-Encoder pour évaluer la pertinence requête-document.
    """
    
    def __init__(
        self, 
        model_name: str = CROSS_ENCODER_MODELS.get("default"),
        batch_size: int = 32,
        enabled: bool = True
    ):
        """
        Initialise le réordonnateur contextuel.
        
        Args:
            model_name: Nom du modèle Cross-Encoder
            batch_size: Taille des lots pour l'inférence
            enabled: Si True, active le réordonancement
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.enabled = enabled
        self.model = None
        
        # Charger le modèle si activé
        if enabled:
            self._load_model()
            
        logger.info(f"ContextualReranker initialisé: model={model_name}, enabled={enabled}")
    
    def _load_model(self):
        """Charge le modèle Cross-Encoder"""
        try:
            logger.info(f"Chargement du modèle Cross-Encoder: {self.model_name}")
            self.model = CrossEncoder(self.model_name, max_length=512)
            logger.info("Modèle Cross-Encoder chargé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle Cross-Encoder: {str(e)}")
            self.enabled = False
    
    @timed(logger=logger)
    @log_exceptions(logger=logger)
    def rerank(
        self, 
        query: str, 
        results: List[Dict[str, Any]], 
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Réordonne les résultats de recherche.
        
        Args:
            query: Requête de recherche
            results: Résultats à réordonner
            top_k: Nombre de résultats à retourner (si None, retourne tous les résultats)
            
        Returns:
            Résultats réordonnés
        """
        if not self.enabled or not self.model or not results:
            return results
        
        start_time = time.time()
        
        # Préparer les paires requête-document pour le modèle
        pairs = [(query, result["text"]) for result in results]
        
        # Calculer les scores de pertinence
        try:
            scores = self.model.predict(pairs, batch_size=self.batch_size)
        except Exception as e:
            logger.error(f"Erreur lors du réordonnancement: {str(e)}")
            return results
        
        # Ajouter les scores de réordonnancement aux résultats
        for i, result in enumerate(results):
            result["rerank_score"] = float(scores[i])
        
        # Réordonner les résultats
        reranked_results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        
        # Mettre à jour les rangs
        for i, result in enumerate(reranked_results):
            result["original_rank"] = result.get("rank", i + 1)
            result["rank"] = i + 1
        
        # Limiter les résultats si nécessaire
        if top_k and top_k < len(reranked_results):
            reranked_results = reranked_results[:top_k]
        
        logger.info(f"Réordonnancement terminé: {len(reranked_results)} résultats en {time.time() - start_time:.2f}s")
        return reranked_results
    
    def toggle(self, enabled: bool) -> bool:
        """
        Active ou désactive le réordonnancement.
        
        Args:
            enabled: Si True, active le réordonnancement
            
        Returns:
            État actuel (True si activé)
        """
        if enabled and not self.model:
            self._load_model()
        
        self.enabled = enabled and self.model is not None
        return self.enabled