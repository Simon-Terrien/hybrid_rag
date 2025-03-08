from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Composants LangChain pour l'extraction et le traitement des documents
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Importation de la configuration et des utilitaires
from config import (
    DATA_DIR, PROCESSED_DIR, CHUNKING_CONFIG, 
    DOCUMENT_REGISTRY_PATH, SUPPORTED_FORMATS, logger
)
from utils import (
    compute_file_hash, extract_metadata, adapt_chunking_parameters,
    load_json, save_json, generate_document_id, generate_chunk_id,
    filter_complex_metadata, timed, log_exceptions, retry,
    process_with_progress
)

class DocumentProcessor:
    """Système d'extraction et de chunking pour documents sécurisés"""
    
    def __init__(self, data_dir: Optional[Path] = None, processed_dir: Optional[Path] = None):
        """
        Initialise le processeur de documents.
        
        Args:
            data_dir: Répertoire des données brutes (défaut: depuis config)
            processed_dir: Répertoire des données traitées (défaut: depuis config)
        """
        self.data_dir = data_dir or DATA_DIR
        self.processed_dir = processed_dir or PROCESSED_DIR
        
        # Créer les répertoires nécessaires
        self.processed_dir.mkdir(exist_ok=True, parents=True)
        (self.processed_dir / "chunks").mkdir(exist_ok=True)
        (self.processed_dir / "metadata").mkdir(exist_ok=True)
        
        # Registre des documents traités
        self.document_registry_path = DOCUMENT_REGISTRY_PATH
        self.document_registry = self._load_document_registry()
        
        logger.info(f"DocumentProcessor initialisé: data_dir={self.data_dir}, processed_dir={self.processed_dir}")
    
    def _load_document_registry(self) -> Dict[str, Any]:
        """Charge le registre des documents ou en crée un nouveau"""
        if self.document_registry_path.exists():
            logger.debug(f"Chargement du registre existant: {self.document_registry_path}")
            return load_json(self.document_registry_path)
        
        logger.info(f"Création d'un nouveau registre de documents")
        registry = {"documents": {}, "last_updated": datetime.now().isoformat()}
        save_json(registry, self.document_registry_path)
        return registry
    
    def _save_document_registry(self):
        """Sauvegarde le registre des documents"""
        self.document_registry["last_updated"] = datetime.now().isoformat()
        save_json(self.document_registry, self.document_registry_path, backup=True)
    
    def _get_document_loader(self, file_path: Path):
        """
        Sélectionne le bon loader pour le type de document.
        
        Args:
            file_path: Chemin du fichier à charger
            
        Returns:
            Loader LangChain approprié
        """
        extension = file_path.suffix.lower()
        
        if extension not in SUPPORTED_FORMATS:
            raise ValueError(f"Format de fichier non supporté: {extension}")
        
        loader_name = SUPPORTED_FORMATS[extension]["loader"]
        
        if loader_name == "PyPDFLoader":
            return PyPDFLoader(str(file_path))
        elif loader_name == "TextLoader":
            return TextLoader(str(file_path), encoding="utf-8")
        elif loader_name == "Docx2txtLoader":
            return Docx2txtLoader(str(file_path))
        else:
            raise ValueError(f"Loader non implémenté: {loader_name}")
    
    @timed(logger=logger)
    @log_exceptions(logger=logger)
    def process_document(self, file_path: Path) -> Dict[str, Any]:
        """
        Traite un document, extrait le texte et le découpe en chunks.
        
        Args:
            file_path: Chemin du fichier à traiter
            
        Returns:
            Information sur le document traité
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Le fichier {file_path} n'existe pas")
        
        # Calculer le hash du document pour vérifier les modifications
        doc_hash = compute_file_hash(file_path)
        document_id = generate_document_id(file_path, doc_hash)
        rel_path = str(file_path.relative_to(self.data_dir))
        
        # Vérifier si le document est déjà traité et n'a pas changé
        if rel_path in self.document_registry["documents"] and self.document_registry["documents"][rel_path]["hash"] == doc_hash:
            logger.info(f"Document {rel_path} déjà traité et inchangé")
            return self.document_registry["documents"][rel_path]
        
        logger.info(f"Traitement du document: {rel_path}")
        
        # Charger le document avec le loader approprié
        loader = self._get_document_loader(file_path)
        pages = loader.load()
        
        # Extraire les métadonnées
        doc_metadata = extract_metadata(file_path, len(pages))
        
        # Adapter les paramètres de chunking
        chunking_params = adapt_chunking_parameters(len(pages), CHUNKING_CONFIG)
        
        # Créer le text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(chunking_params["chunk_size"]),
            chunk_overlap=int(chunking_params["chunk_overlap"]),
            separators=CHUNKING_CONFIG.get("separators", ["\n\n", "\n", " ", ""]),
            keep_separator=False
        )
        
        # Découper le document en chunks
        chunks = text_splitter.split_documents(pages)
        
        # Enrichir les métadonnées de chaque chunk
        for i, chunk in enumerate(chunks):
            chunk_id = generate_chunk_id(document_id, i)
            chunk_metadata = chunk.metadata.copy()
            chunk_metadata.update({
                "document_id": document_id,
                "chunk_id": chunk_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                **doc_metadata
            })
            chunk.metadata = filter_complex_metadata(chunk_metadata)
        
        # Sauvegarder les chunks
        chunk_file = self.processed_dir / "chunks" / f"{document_id}.json"
        chunk_data = [{
            "text": chunk.page_content,
            "metadata": chunk.metadata
        } for chunk in chunks]
        save_json(chunk_data, chunk_file)
        
        # Sauvegarder les métadonnées
        metadata_file = self.processed_dir / "metadata" / f"{document_id}.json"
        metadata_info = {
            "metadata": doc_metadata,
            "hash": doc_hash,
            "chunk_count": len(chunks),
            "chunk_file": str(chunk_file.relative_to(self.processed_dir))
        }
        save_json(metadata_info, metadata_file)
        
        # Mettre à jour le registre
        self.document_registry["documents"][rel_path] = {
            "document_id": document_id,
            "hash": doc_hash,
            "metadata": doc_metadata,
            "chunk_count": len(chunks),
            "chunk_file": str(chunk_file.relative_to(self.processed_dir)),
            "metadata_file": str(metadata_file.relative_to(self.processed_dir))
        }
        self._save_document_registry()
        
        logger.info(f"Document {rel_path} traité avec succès: {len(chunks)} chunks générés")
        return self.document_registry["documents"][rel_path]
    
    @timed(logger=logger)
    def process_directory(self, subdirectory: Optional[str] = None) -> Dict[str, Any]:
        """
        Traite tous les documents dans le répertoire de données.
        
        Args:
            subdirectory: Sous-répertoire optionnel à traiter
            
        Returns:
            Résultats du traitement
        """
        base_dir = self.data_dir
        if subdirectory:
            base_dir = base_dir / subdirectory
            if not base_dir.exists():
                raise FileNotFoundError(f"Le sous-répertoire {subdirectory} n'existe pas")
        
        # Collecter tous les fichiers supportés
        all_files = []
        for ext in SUPPORTED_FORMATS.keys():
            all_files.extend(list(base_dir.glob(f"**/*{ext}")))
        
        logger.info(f"Traitement de {len(all_files)} fichiers dans {base_dir}")
        
        results = {
            "total_documents": len(all_files),
            "processed_documents": 0,
            "skipped_documents": 0,
            "failed_documents": 0,
            "documents": []
        }
        
        # Traiter chaque fichier avec suivi de progression
        def process_file(file_path):
            try:
                rel_path = str(file_path.relative_to(self.data_dir))
                doc_hash = compute_file_hash(file_path)
                
                # Vérifier si déjà traité et inchangé
                if rel_path in self.document_registry["documents"] and \
                   self.document_registry["documents"][rel_path]["hash"] == doc_hash:
                    results["skipped_documents"] += 1
                    return {
                        "path": rel_path,
                        "status": "skipped",
                        "reason": "unchanged"
                    }
                
                # Traiter le document
                doc_result = self.process_document(file_path)
                results["processed_documents"] += 1
                return {
                    "path": rel_path,
                    "document_id": doc_result["document_id"],
                    "chunk_count": doc_result["chunk_count"],
                    "status": "processed"
                }
            except Exception as e:
                logger.error(f"Erreur lors du traitement de {file_path}: {str(e)}")
                results["failed_documents"] += 1
                return {
                    "path": str(file_path.relative_to(self.data_dir)),
                    "error": str(e),
                    "status": "failed"
                }
        
        document_results = process_with_progress(
            all_files, 
            process_file, 
            description="traitement des documents",
            logger=logger
        )
        
        results["documents"] = document_results
        
        # Enregistrer les résultats du traitement
        save_json(results, self.processed_dir / "processing_results.json")
        
        logger.info(f"Traitement terminé: {results['processed_documents']} traités, "
                  f"{results['skipped_documents']} ignorés, "
                  f"{results['failed_documents']} en échec")
        
        return results


# Exemple d'utilisation
if __name__ == "__main__":
    processor = DocumentProcessor()
    
    # Traiter un seul document
    # result = processor.process_document(Path(DATA_DIR) / "exemple_document.pdf")
    # print(f"Document traité: {result['document_id']}")
    
    # Traiter tous les documents
    results = processor.process_directory()
    print(f"Documents traités: {results['processed_documents']}")
    print(f"Documents ignorés: {results['skipped_documents']}")
    print(f"Documents en échec: {results['failed_documents']}")