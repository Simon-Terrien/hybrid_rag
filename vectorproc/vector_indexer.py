from pathlib import Path
import time
import shutil
from typing import List, Dict, Any, Optional, Union
import numpy as np

# Hugging Face transformers library for embeddings
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer

# FAISS for vector search
import faiss
import pickle

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

class Document:
    """Simple document class to replace LangChain Document"""
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

class CustomEmbeddings:
    """Custom embeddings class to replace LangChain HuggingFaceEmbeddings"""
    
    def __init__(self, model_name: str, device: str = None):
        """
        Initialize with a Hugging Face model
        
        Args:
            model_name: Name of the Hugging Face model
            device: Device to use ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self._device = device or ('cuda' if torch.cuda.is_available() and USE_GPU else 'cpu')
        logger.info(f"Initializing embeddings model {model_name} on {self._device}")
        
        self._model = SentenceTransformer(model_name, device=self._device)
        self._client = self._model
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding
        """
        embedding = self._model.encode([text], normalize_embeddings=True)
        return embedding[0].tolist()

class FAISSVectorStore:
    """Custom FAISS vector store to replace LangChain FAISS"""
    
    def __init__(self, embedding_function: CustomEmbeddings):
        """
        Initialize FAISS vector store
        
        Args:
            embedding_function: Function to embed texts
        """
        self.embedding_function = embedding_function
        self.index = None
        self.docstore = {}
        self.index_to_docstore_id = []
    
    @classmethod
    def from_documents(cls, documents: List[Document], embedding: CustomEmbeddings):
        """
        Create a new FAISSVectorStore from documents
        
        Args:
            documents: List of documents
            embedding: Embedding function
            
        Returns:
            FAISSVectorStore instance
        """
        instance = cls(embedding)
        if not documents:
            # Initialize empty index
            dimension = 768  # Default dimension, adjust based on your model
            instance.index = faiss.IndexFlatIP(dimension)
            return instance
            
        instance.add_documents(documents)
        return instance
    
    def add_documents(self, documents: List[Document]):
        """
        Add documents to the vector store
        
        Args:
            documents: List of documents to add
        """
        if not documents:
            logger.warning("No documents to add")
            return
            
        # Extract texts
        texts = [doc.page_content for doc in documents]
        
        # Generate embeddings
        embeddings = self.embedding_function.embed_documents(texts)
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Initialize index if needed
        if self.index is None:
            dimension = embeddings_array.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
        
        # Add to docstore
        starting_index = len(self.index_to_docstore_id)
        for i, doc in enumerate(documents):
            doc_id = f"doc_{starting_index + i}"
            self.docstore[doc_id] = doc
            self.index_to_docstore_id.append(doc_id)
        
        # Add to index
        self.index.add(embeddings_array)
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """
        Search for documents similar to query
        
        Args:
            query: Query text
            k: Number of results
            
        Returns:
            List of (document, score) tuples
        """
        # Convert query to embedding
        query_embedding = self.embedding_function.embed_query(query)
        query_embedding_array = np.array([query_embedding]).astype('float32')
        
        # Search
        scores, indices = self.index.search(query_embedding_array, k=min(k, len(self.index_to_docstore_id)))
        
        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.index_to_docstore_id):
                continue
                
            doc_id = self.index_to_docstore_id[idx]
            if doc_id in self.docstore:
                results.append((self.docstore[doc_id], float(score)))
        
        return results
    
    def save_local(self, folder_path: str, index_name: str = "index"):
        """
        Save the vector store to disk
        
        Args:
            folder_path: Path to save to
            index_name: Name of the index
        """
        path = Path(folder_path)
        path.mkdir(exist_ok=True, parents=True)
        
        # Save the index
        faiss.write_index(self.index, str(path / f"{index_name}.faiss"))
        
        # Save the docstore and mapping
        metadata = {
            "docstore": self.docstore,
            "index_to_docstore_id": self.index_to_docstore_id
        }
        with open(path / f"{index_name}.pkl", 'wb') as f:
            pickle.dump(metadata, f)
    
    @classmethod
    def load_local(cls, folder_path: str, embedding_function: CustomEmbeddings, index_name: str = "index", allow_dangerous_deserialization: bool = False):
        """
        Load a vector store from disk
        
        Args:
            folder_path: Path to load from
            embedding_function: Embedding function
            index_name: Name of the index
            allow_dangerous_deserialization: Allow pickle deserialization
            
        Returns:
            FAISSVectorStore instance
        """
        path = Path(folder_path)
        
        # Create instance
        instance = cls(embedding_function)
        
        # Load the index
        instance.index = faiss.read_index(str(path / f"{index_name}.faiss"))
        
        # Load the docstore and mapping
        with open(path / f"{index_name}.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        instance.docstore = metadata["docstore"]
        instance.index_to_docstore_id = metadata["index_to_docstore_id"]
        
        return instance

class ChromaVectorStore:
    """Placeholder for Chroma implementation"""
    
    def __init__(self, embedding_function, persist_directory, collection_name="documents"):
        logger.warning("Chroma support is currently a placeholder. Using in-memory implementation.")
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Use FAISS as a fallback
        self.backend = FAISSVectorStore(embedding_function)
    
    @classmethod
    def from_documents(cls, documents, embedding_function, persist_directory, collection_name="documents"):
        instance = cls(embedding_function, persist_directory, collection_name)
        instance.backend = FAISSVectorStore.from_documents(documents, embedding_function)
        return instance
    
    def add_documents(self, documents):
        self.backend.add_documents(documents)
    
    def similarity_search_with_score(self, query, k=5, filter=None):
        results = self.backend.similarity_search_with_score(query, k)
        # Apply filters if any
        if filter and results:
            filtered_results = []
            for doc, score in results:
                if all(self._check_filter(doc.metadata, key, value) for key, value in filter.items()):
                    filtered_results.append((doc, score))
            return filtered_results
        return results
    
    def _check_filter(self, metadata, key, filter_value):
        if key not in metadata:
            return False
            
        if isinstance(filter_value, dict):
            # Handle operators
            for op, value in filter_value.items():
                if op == "$eq":
                    return metadata[key] == value
                elif op == "$in":
                    return metadata[key] in value
                # Add more operators as needed
        else:
            # Simple equality
            return metadata[key] == filter_value
        
        return False
    
    def persist(self):
        # In a real implementation, this would save to Chroma
        path = Path(self.persist_directory)
        path.mkdir(exist_ok=True, parents=True)
        
        self.backend.save_local(str(path), "chroma_fallback")

class VectorIndexer:
    """Système de vectorisation et d'indexation pour les documents traités"""
    
    def __init__(
        self,
        processed_dir: Optional[Path] = PROCESSED_DIR,
        vector_db_dir: Optional[Path] = VECTOR_DB_DIR,
        embedding_model_name: Optional[str] = EMBEDDING_MODEL,
        vector_db_type: Optional[str] = VECTOR_DB_TYPE,
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
        logger.info(f"Modèle d'embedding initialisé sur {self.embeddings._device}")
        
        # Initialisation ou chargement de la base vectorielle
        self._init_vector_db()
        
        logger.info(f"VectorIndexer initialisé: vector_db_type={self.vector_db_type}, "
                  f"embedding_model={self.embedding_model_name}")
    
    def _initialize_embeddings(self) -> CustomEmbeddings:
        """Initialise le modèle d'embedding"""
        return CustomEmbeddings(
            model_name=self.embedding_model_name,
            device='cuda' if USE_GPU and self._is_cuda_available() else 'cpu'
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
                self.vector_db = FAISSVectorStore.load_local(
                    folder_path=str(self.vector_db_dir),
                    embedding_function=self.embeddings,
                    index_name="index",
                    allow_dangerous_deserialization=True
                )
            else:  # chroma
                # For Chroma, load our custom implementation
                chroma_settings = VECTOR_DB_CONFIG.get("chroma", {})
                self.vector_db = ChromaVectorStore(
                    embedding_function=self.embeddings,
                    persist_directory=str(self.vector_db_dir / "chroma"),
                    collection_name=chroma_settings.get("collection_name", "documents")
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
        for doc_info in self.document_registry["documents"].values():
            if doc_info.get("document_id") == document_id:
                document_entry = doc_info
                break
        
        if not document_entry:
            raise ValueError(f"Document non trouvé avec l'ID: {document_id}")
        
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
        
        logger.debug(f"Préparé {len(documents)} chunks pour l'indexation du document {document_id}")
        return documents
        
    def _index_chunks_batch(self, chunks: List[Document]) -> Union[FAISSVectorStore, ChromaVectorStore]:
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
                self.vector_db = FAISSVectorStore.from_documents(
                    chunks, self.embeddings
                )
                # Sauvegarder immédiatement
                self.vector_db.save_local(str(self.vector_db_dir), index_name="index")
            else:  # chroma
                chroma_settings = VECTOR_DB_CONFIG.get("chroma", {})
                self.vector_db = ChromaVectorStore.from_documents(
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
            if info.get("document_id") == document_id:
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
        Index all processed documents in the vector database.
        
        Args:
            force_reindex: If True, reindex all documents even if already indexed
            
        Returns:
            Indexing results
        """
        start_time = time.time()
        
        # Check if the document registry is empty or has an unexpected structure
        if not self.document_registry.get("documents"):
            logger.warning("Document registry is empty or missing 'documents' key")
            return {
                "status": "failed",
                "error": "Document registry is empty or has an invalid structure",
                "total_documents": 0,
                "indexed_documents": 0,
                "skipped_documents": 0,
                "failed_documents": 0,
                "documents": []
            }
        
        # Collect all documents to index
        documents_to_index = []
        invalid_docs = 0
        
        logger.info(f"Scanning document registry with {len(self.document_registry['documents'])} entries")
        
        for doc_path, doc_info in self.document_registry["documents"].items():
            # Check if the document info contains the required fields
            if "document_id" not in doc_info:
                logger.warning(f"Document entry for path '{doc_path}' is missing 'document_id' field")
                invalid_docs += 1
                continue
            
            document_id = doc_info["document_id"]
            
            # Also check for hash field which is needed for comparison
            if "hash" not in doc_info:
                logger.warning(f"Document {document_id} is missing 'hash' field, will force reindex")
                documents_to_index.append((document_id, doc_info))
                continue
                
            # Check if we should reindex
            if not force_reindex and document_id in self.vector_registry["indexed_documents"] and \
            doc_info["hash"] == self.vector_registry["indexed_documents"][document_id]["hash"]:
                logger.debug(f"Document {document_id} already indexed and up to date, skipping")
                continue
                    
            documents_to_index.append((document_id, doc_info))
        
        valid_docs = len(self.document_registry["documents"]) - invalid_docs
        logger.info(f"Found {valid_docs} valid documents, {invalid_docs} invalid entries")
        logger.info(f"Indexing {len(documents_to_index)} documents out of {valid_docs} valid documents")
        
        results = {
            "total_documents": valid_docs,
            "indexed_documents": 0,
            "skipped_documents": valid_docs - len(documents_to_index),
            "failed_documents": 0,
            "invalid_documents": invalid_docs,
            "documents": []
        }
        
        # Document processing function
        def process_document(doc_tuple):
            document_id, doc_info = doc_tuple
            try:
                logger.info(f"Indexing document: {document_id}")
                indexing_result = self.index_document(document_id, force_reindex)
                results["indexed_documents"] += 1
                return {
                    "document_id": document_id,
                    "status": "indexed",
                    "chunk_count": indexing_result["chunk_count"],
                    "processing_time": indexing_result["processing_time_seconds"]
                }
            except Exception as e:
                logger.error(f"Error indexing document {document_id}: {str(e)}")
                logger.error(f"Exception type: {type(e).__name__}")
                results["failed_documents"] += 1
                return {
                    "document_id": document_id,
                    "status": "failed",
                    "error": str(e)
                }
        
        # Process all documents with progress tracking
        if documents_to_index:
            document_results = process_with_progress(
                documents_to_index,
                process_document,
                description="indexing documents",
                logger=logger
            )
            results["documents"] = document_results
        
        # Add skipped documents
        for doc_path, doc_info in self.document_registry["documents"].items():
            if "document_id" not in doc_info:
                continue  # Skip invalid entries
                
            document_id = doc_info["document_id"]
            if not any(d.get("document_id") == document_id for d in results["documents"]):
                results["documents"].append({
                    "document_id": document_id,
                    "status": "skipped",
                    "reason": "already_indexed"
                })
        
        results["total_processing_time"] = round(time.time() - start_time, 2)
        
        # Save the results
        save_json(results, self.vector_db_dir / "indexing_results.json")
        
        logger.info(f"Indexing completed in {results['total_processing_time']} seconds:")
        logger.info(f"  Documents indexed: {results['indexed_documents']}")
        logger.info(f"  Documents skipped: {results['skipped_documents']}")
        logger.info(f"  Documents failed: {results['failed_documents']}")
        logger.info(f"  Invalid documents: {results['invalid_documents']}")
        logger.info(f"  Total chunks: {self.vector_registry['total_chunks']}")
        
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
            "device": self.embeddings._device,
            "documents": [
                {
                    "document_id": doc_id,
                    "chunk_count": info["chunk_count"],
                    "indexing_date": info["indexing_date"]
                } for doc_id, info in self.vector_registry["indexed_documents"].items()
            ]
        }
    
    @log_exceptions(logger=logger)   


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