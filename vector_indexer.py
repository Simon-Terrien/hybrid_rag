from pathlib import Path
import time
import shutil
from typing import List, Dict, Any, Optional, Union

# Composants LangChain pour les embeddings et la base vectorielle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS, Chroma

# Importation de la configuration et des utilitaires
from config import (
    PROCESSED_DIR, VECTOR_DB_DIR, EMBEDDING_MODEL, USE_GPU,
    VECTOR_DB_TYPE, VECTOR_DB_CONFIG, DOCUMENT_REGISTRY_PATH,
    VECTOR_REGISTRY_PATH, BATCH_SIZE, logger
)
from utils import (
    load_json, save_json, timed, log_exceptions, retry,
    filter_complex_metadata, process_with_progress
)

class VectorIndexer:
    """Système de vectorisation et d'indexation pour les documents traités"""
    
    def __init__(
        self,
        processed_dir: Optional[Path] = None,
        vector_db_dir: Optional[Path] = None,
        embedding_model_name: Optional[str] = None,
        vector_db_type: Optional[str] = None,
        batch_size: Optional[int] = None
    ):
        """
        Initialise l'indexeur vectoriel.
        
        Args:
            processed_dir: Répertoire des données traitées (défaut: depuis config)
            vector_db_dir: Répertoire de la base vectorielle (défaut: depuis config)
            embedding_model_name: Nom du modèle d'embedding (défaut: depuis config)
            vector_db_type: Type de base vectorielle ('faiss' ou 'chroma') (défaut: depuis config)
            batch_size: Taille des lots pour le traitement par lots (défaut: depuis config)
        """
        self.processed_dir = processed_dir or PROCESSED_DIR
        self.vector_db_dir = vector_db_dir or VECTOR_DB_DIR
        self.embedding_model_name = embedding_model_name or EMBEDDING_MODEL
        self.vector_db_type = (vector_db_type or VECTOR_DB_TYPE).lower()
        self.batch_size = batch_size or BATCH_SIZE
        
        # Créer les répertoires nécessaires
        self.vector_db_dir.mkdir(exist_ok=True, parents=True)
        (self.vector_db_dir / "backups").mkdir(exist_ok=True)
        
        # Chargement du registre des documents
        self.document_registry_path = DOCUMENT_REGISTRY_PATH
        self.document_registry = self._load_document_registry()
        
        # Registre de vectorisation
        self.vector_registry_path = VECTOR_REGISTRY_PATH
        self.vector_registry = self._load_vector_registry()
        
        # Initialisation du modèle d'embedding
        logger.info(f"Initialisation du modèle d'embedding {self.embedding_model_name}...")
        self.embeddings = self._initialize_embeddings()
        logger.info(f"Modèle d'embedding initialisé sur {self.embeddings._client.device}")
        
        # Initialisation ou chargement de la base vectorielle
        self._init_vector_db()
        
        logger.info(f"VectorIndexer initialisé: vector_db_type={self.vector_db_type}, "
                  f"embedding_model={self.embedding_model_name}")
    
    def _initialize_embeddings(self) -> HuggingFaceEmbeddings:
        """Initialise le modèle d'embedding"""
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': 'cuda' if USE_GPU and self._is_cuda_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def _is_cuda_available(self) -> bool:
        """Vérifie si CUDA est disponible"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            logger.warning("PyTorch non installé, utilisation du CPU pour les embeddings")
            return False
    
    def _load_document_registry(self) -> Dict[str, Any]:
        """Charge le registre des documents"""
        if not self.document_registry_path.exists():
            raise FileNotFoundError(f"Le registre des documents n'existe pas: {self.document_registry_path}")
        
        return load_json(self.document_registry_path)
    
    def _load_vector_registry(self) -> Dict[str, Any]:
        """Charge le registre des vecteurs ou en crée un nouveau"""
        if self.vector_registry_path.exists():
            logger.debug(f"Chargement du registre vectoriel existant: {self.vector_registry_path}")
            return load_json(self.vector_registry_path)
        
        logger.info(f"Création d'un nouveau registre vectoriel")
        registry = {
            "vector_db_type": self.vector_db_type,
            "embedding_model": self.embedding_model_name,
            "indexed_documents": {},
            "total_chunks": 0,
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        save_json(registry, self.vector_registry_path)
        return registry
    
    def _save_vector_registry(self):
        """Sauvegarde le registre des vecteurs"""
        self.vector_registry["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        save_json(self.vector_registry, self.vector_registry_path, backup=True)
    def _init_vector_db(self):
        """Initialise ou charge la base vectorielle"""
        # Déterminer si la base vectorielle existe
        if self.vector_db_type == "faiss":
            index_file = self.vector_db_dir / "index.faiss"
            vector_db_exists = index_file.exists()
        else:  # chroma
            chroma_dir = self.vector_db_dir / "chroma"
            vector_db_exists = chroma_dir.exists() and any(chroma_dir.iterdir())
        
        if vector_db_exists:
            logger.info(f"Chargement de la base vectorielle existante ({self.vector_db_type})...")
            if self.vector_db_type == "faiss":
                self.vector_db = FAISS.load_local(
                    folder_path=str(self.vector_db_dir),
                    embeddings=self.embeddings,
                    index_name="index",
                    allow_dangerous_deserialization=True
                )
            else:  # chroma
                self.vector_db = Chroma(
                    persist_directory=str(self.vector_db_dir / "chroma"),
                    embedding_function=self.embeddings,
                    collection_name=VECTOR_DB_CONFIG.get("chroma", {}).get("collection_name", "documents")
                )
            logger.info(f"Base vectorielle chargée avec succès")
        else:
            logger.info(f"Aucune base vectorielle existante trouvée, elle sera créée lors de la première indexation")
            self.vector_db = None

    @retry(max_attempts=3, exceptions=(IOError,), logger=logger)
    def _prepare_chunks_for_indexing(self, document_id: str) -> List[Document]:
        """
        Prépare les chunks d'un document pour l'indexation.
        
        Args:
            document_id: Identifiant du document
            
        Returns:
            Liste des chunks sous forme d'objets Document
        """
        # Trouver le fichier de chunks correspondant
        document_entry = None
        for doc_path, doc_info in self.document_registry["documents"].items():
            if doc_info["document_id"] == document_id:  # Correction ici
                document_entry = doc_info
                break
        
        if not document_entry:
            raise ValueError(f"Document non trouvé avec l'ID: {document_id}")  # Correction ici
        
        chunk_file_path = self.processed_dir / document_entry["chunk_file"]
        if not chunk_file_path.exists():
            raise FileNotFoundError(f"Fichier de chunks non trouvé: {chunk_file_path}")
        
        # Charger les chunks
        chunks_data = load_json(chunk_file_path)
        
        # Convertir en objets Document
        documents = []
        for chunk_data in chunks_data:
            # Filtrer les métadonnées complexes qui ne peuvent pas être sérialisées
            filtered_metadata = filter_complex_metadata(chunk_data["metadata"])
            documents.append(Document(
                page_content=chunk_data["text"],
                metadata=filtered_metadata
            ))
        
        logger.debug(f"Préparé {len(documents)} chunks pour l'indexation du document {document_id}")  # Correction ici
        return documents
    def _index_chunks_batch(self, chunks: List[Document]) -> Union[FAISS, Chroma]:
        """
        Indexe un lot de chunks et retourne la base vectorielle mise à jour.
        
        Args:
            chunks: Liste des chunks à indexer
            
        Returns:
            Base vectorielle mise à jour
        """
        # Si c'est la première indexation, créer la base vectorielle
        if self.vector_db is None:
            logger.info(f"Création d'une nouvelle base vectorielle ({self.vector_db_type})...")
            if self.vector_db_type == "faiss":
                self.vector_db = FAISS.from_documents(
                    chunks, self.embeddings
                )
                # Sauvegarder immédiatement
                self.vector_db.save_local(str(self.vector_db_dir), index_name="index")
            else:  # chroma
                chroma_settings = VECTOR_DB_CONFIG.get("chroma", {})
                self.vector_db = Chroma.from_documents(
                    chunks,
                    self.embeddings,
                    persist_directory=str(self.vector_db_dir / "chroma"),
                    collection_name=chroma_settings.get("collection_name", "documents")
                )
                self.vector_db.persist()
        else:
            # Ajouter à la base existante
            if self.vector_db_type == "faiss":
                self.vector_db.add_documents(chunks)
                self.vector_db.save_local(str(self.vector_db_dir), index_name="index")
            else:  # chroma
                self.vector_db.add_documents(chunks)
                self.vector_db.persist()
        
        return self.vector_db
    
    @timed(logger=logger)
    @log_exceptions(logger=logger)
    def index_document(self, document_id: str, force_reindex: bool = False) -> Dict[str, Any]:
        """
        Indexe un document spécifique dans la base vectorielle.
        
        Args:
            document_id: Identifiant du document à indexer
            force_reindex: Si True, réindexe même si déjà indexé
            
        Returns:
            Information sur le document indexé
        """
        start_time = time.time()
        
        # Trouver l'entrée du document
        doc_path = None
        for path, info in self.document_registry["documents"].items():
            if info["document_id"] == document_id:
                doc_path = path
                # Si déjà indexé et hash identique, ignorer (sauf si force_reindex)
                if not force_reindex and document_id in self.vector_registry["indexed_documents"] and \
                   info["hash"] == self.vector_registry["indexed_documents"][document_id]["hash"]:
                    logger.info(f"Document {document_id} déjà indexé et à jour")
                    return self.vector_registry["indexed_documents"][document_id]
                break
        
        if not doc_path:
            raise ValueError(f"Document non trouvé avec l'ID: {document_id}")
        
        # Préparer les chunks pour l'indexation
        chunks = self._prepare_chunks_for_indexing(document_id)
        total_chunks = len(chunks)
        logger.info(f"Indexation de {total_chunks} chunks pour le document {document_id}...")
        
        # Supprimer les anciennes entrées si elles existent
        if document_id in self.vector_registry["indexed_documents"]:
            logger.info(f"Document {document_id} déjà indexé, mise à jour...")
            # Note: Suppression logique uniquement, les chunks sont remplacés lors de l'indexation
        
        # Indexer par lots pour optimiser la mémoire
        for i in range(0, total_chunks, self.batch_size):
            batch = chunks[i:i+self.batch_size]
            logger.info(f"Indexation du lot {i//self.batch_size + 1}/{(total_chunks-1)//self.batch_size + 1} ({len(batch)} chunks)")
            self._index_chunks_batch(batch)
        
        # Mettre à jour le registre de vectorisation
        self.vector_registry["indexed_documents"][document_id] = {
            "document_id": document_id,
            "hash": self.document_registry["documents"][doc_path]["hash"],
            "chunk_count": total_chunks,
            "indexing_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "processing_time_seconds": round(time.time() - start_time, 2)
        }
        
        # Mettre à jour le total des chunks
        self.vector_registry["total_chunks"] = sum(
            info["chunk_count"] for info in self.vector_registry["indexed_documents"].values()
        )
        
        self._save_vector_registry()
        
        logger.info(f"Document {document_id} indexé avec succès en {round(time.time() - start_time, 2)} secondes")
        return self.vector_registry["indexed_documents"][document_id]
    
    @timed(logger=logger)
    def index_all_documents(self, force_reindex: bool = False) -> Dict[str, Any]:
        """
        Indexe tous les documents traités dans la base vectorielle.
        
        Args:
            force_reindex: Si True, réindexe tous les documents même s'ils sont déjà indexés
            
        Returns:
            Résultats de l'indexation
        """
        start_time = time.time()
        
        # Collecter tous les documents à indexer
        documents_to_index = []
        for doc_path, doc_info in self.document_registry["documents"].items():
            document_id = doc_info["document_id"]
            
            # Vérifier si on doit réindexer
            if not force_reindex and document_id in self.vector_registry["indexed_documents"] and \
               doc_info["hash"] == self.vector_registry["indexed_documents"][document_id]["hash"]:
                logger.debug(f"Document {document_id} déjà indexé et à jour, ignoré")
                continue
                
            documents_to_index.append((document_id, doc_info))
        
        logger.info(f"Indexation de {len(documents_to_index)} documents sur {len(self.document_registry['documents'])}")
        
        results = {
            "total_documents": len(self.document_registry["documents"]),
            "indexed_documents": 0,
            "skipped_documents": len(self.document_registry["documents"]) - len(documents_to_index),
            "failed_documents": 0,
            "documents": []
        }
        
        # Fonction de traitement pour un document
        def process_document(doc_tuple):
            document_id, doc_info = doc_tuple
            try:
                indexing_result = self.index_document(document_id, force_reindex)
                results["indexed_documents"] += 1
                return {
                    "document_id": document_id,
                    "status": "indexed",
                    "chunk_count": indexing_result["chunk_count"],
                    "processing_time": indexing_result["processing_time_seconds"]
                }
            except Exception as e:
                logger.error(f"Erreur lors de l'indexation de {document_id}: {str(e)}")
                results["failed_documents"] += 1
                return {
                    "document_id": document_id,
                    "status": "failed",
                    "error": str(e)
                }
        
        # Traiter tous les documents avec suivi de progression
        if documents_to_index:
            document_results = process_with_progress(
                documents_to_index,
                process_document,
                description="indexation des documents",
                logger=logger
            )
            results["documents"] = document_results
        
        # Ajouter les documents ignorés
        for doc_path, doc_info in self.document_registry["documents"].items():
            document_id = doc_info["document_id"]
            if not any(d.get("document_id") == document_id for d in results["documents"]):
                results["documents"].append({
                    "document_id": document_id,
                    "status": "skipped",
                    "reason": "already_indexed"
                })
        
        results["total_processing_time"] = round(time.time() - start_time, 2)
        
        # Enregistrer les résultats
        save_json(results, self.vector_db_dir / "indexing_results.json")
        
        logger.info(f"Indexation terminée en {results['total_processing_time']} secondes:")
        logger.info(f"  Documents indexés: {results['indexed_documents']}")
        logger.info(f"  Documents ignorés: {results['skipped_documents']}")
        logger.info(f"  Documents en échec: {results['failed_documents']}")
        logger.info(f"  Total des chunks: {self.vector_registry['total_chunks']}")
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Renvoie des statistiques sur la base vectorielle.
        
        Returns:
            Statistiques de la base vectorielle
        """
        return {
            "vector_db_type": self.vector_db_type,
            "embedding_model": self.embedding_model_name,
            "total_documents": len(self.vector_registry["indexed_documents"]),
            "total_chunks": self.vector_registry["total_chunks"],
            "last_updated": self.vector_registry["last_updated"],
            "device": self.embeddings._client.device,
            "documents": [
                {
                    "document_id": doc_id,
                    "chunk_count": info["chunk_count"],
                    "indexing_date": info["indexing_date"]
                } for doc_id, info in self.vector_registry["indexed_documents"].items()
            ]
        }
    
    @log_exceptions(logger=logger)
    def clear_vector_db(self) -> None:
        """
        Supprime complètement la base vectorielle mais conserve une sauvegarde.
        """
        # Créer une sauvegarde avant suppression
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_dir = self.vector_db_dir / "backups" / timestamp
        backup_dir.mkdir(exist_ok=True, parents=True)
        
        # Sauvegarder le registre
        save_json(self.vector_registry, backup_dir / "vector_registry.json")
        
        # Fermer les connexions avant suppression
        self.vector_db = None
        
        # Supprimer les fichiers physiques avec sauvegarde
        if self.vector_db_type == "faiss":
            index_file = self.vector_db_dir / "index.faiss"
            metadata_file = self.vector_db_dir / "index.pkl"
            
            if index_file.exists():
                shutil.copy2(index_file, backup_dir / "index.faiss")
                index_file.unlink()
            
            if metadata_file.exists():
                shutil.copy2(metadata_file, backup_dir / "index.pkl")
                metadata_file.unlink()
                
        else:  # chroma
            chroma_dir = self.vector_db_dir / "chroma"
            if chroma_dir.exists():
                shutil.copytree(chroma_dir, backup_dir / "chroma")
                shutil.rmtree(chroma_dir)
        
        # Réinitialiser le registre
        self.vector_registry = {
            "vector_db_type": self.vector_db_type,
            "embedding_model": self.embedding_model_name,
            "indexed_documents": {},
            "total_chunks": 0,
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        self._save_vector_registry()
        
        logger.info(f"Base vectorielle supprimée avec succès (sauvegarde dans {backup_dir})")


# Exemple d'utilisation
if __name__ == "__main__":
    indexer = VectorIndexer()
    
    # Indexer un document spécifique
    # result = indexer.index_document("a1b2c3d4e5")
    # print(f"Document indexé: {result['document_id']}")
    
    # Indexer tous les documents
    results = indexer.index_all_documents(force_reindex=False)
    print(f"Indexation terminée: {results['indexed_documents']} documents indexés")
    
    # Afficher les statistiques
    stats = indexer.get_stats()
    print(f"Total des documents: {stats['total_documents']}")
    print(f"Total des chunks: {stats['total_chunks']}")