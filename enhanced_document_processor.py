from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

import pandas as pd
import unidecode
import re
import string

# Composants LangChain pour l'extraction et le traitement des documents
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader, Docx2txtLoader
from langchain.schema import Document

# Import the new semantic chunker
from semantic_chunker import SemanticChunker

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

class EnhancedDocumentProcessor:
    """Système amélioré d'extraction et de chunking pour documents sécurisés"""
    
    def __init__(self, data_dir: Optional[Path] = None, processed_dir: Optional[Path] = None):
        """
        Initialise le processeur de documents amélioré.
        
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
        (self.processed_dir / "cleaned").mkdir(exist_ok=True)  # Nouveau répertoire pour les fichiers nettoyés
        
        # Registre des documents traités
        self.document_registry_path = DOCUMENT_REGISTRY_PATH
        self.document_registry = self._load_document_registry()
        
        # Initialize the semantic chunker
        self.semantic_chunker = SemanticChunker(
            base_chunk_size=CHUNKING_CONFIG.get("default_chunk_size", 1000),
            base_chunk_overlap=CHUNKING_CONFIG.get("default_chunk_overlap", 200),
            use_spacy=True,
            use_markdown_headers=True
        )
        
        logger.info(f"EnhancedDocumentProcessor initialisé: data_dir={self.data_dir}, processed_dir={self.processed_dir}")
    
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
    
    def _clean_text(self, text: str) -> str:
        """
        Nettoie le texte en supprimant les caractères indésirables et en normalisant le contenu.
        
        Args:
            text: Texte à nettoyer
            
        Returns:
            Texte nettoyé
        """
        # Remplacer les caractères spéciaux par leurs équivalents ASCII
        text = unidecode.unidecode(text)
        
        # Suppression des caractères de contrôle
        text = ''.join(ch for ch in text if ch >= ' ' or ch in ['\n', '\t'])
        
        # Normalisation des espaces multiples
        text = re.sub(r'\s+', ' ', text)
        
        # Normalisation des sauts de ligne multiples
        text = re.sub(r'\n+', '\n\n', text)
        
        # Normalisation des tabulations
        text = re.sub(r'\t+', '\t', text)
        
        # Suppression des lignes vides consécutives
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        # Suppression des espaces en début et fin de texte
        text = text.strip()
        
        return text
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Nettoie les métadonnées du document.
        
        Args:
            metadata: Métadonnées à nettoyer
            
        Returns:
            Métadonnées nettoyées
        """
        cleaned_metadata = {}
        
        for key, value in metadata.items():
            # Normalisation des clés
            clean_key = key.strip().lower().replace(' ', '_')
            
            # Traitement des valeurs
            if isinstance(value, str):
                # Nettoyage et normalisation des chaînes
                clean_value = value.strip()
                # Conversion des dates au format ISO si possible
                if any(date_keyword in clean_key for date_keyword in ['date', 'time', 'created', 'modified']):
                    try:
                        # Attempt to parse and standardize date format
                        clean_value = pd.to_datetime(clean_value).isoformat()
                    except:
                        pass
                cleaned_metadata[clean_key] = clean_value
            elif isinstance(value, (int, float, bool, type(None))):
                # Conserver les valeurs numériques, booléennes et None telles quelles
                cleaned_metadata[clean_key] = value
            elif isinstance(value, dict):
                # Nettoyage récursif des dictionnaires
                cleaned_metadata[clean_key] = self._clean_metadata(value)
            elif isinstance(value, list):
                # Nettoyage des listes (premier niveau)
                cleaned_metadata[clean_key] = [
                    self._clean_metadata(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                # Conversion en chaîne pour les autres types
                cleaned_metadata[clean_key] = str(value)
        
        return cleaned_metadata
    
    def _extract_additional_metadata(self, text: str, file_path: Path) -> Dict[str, Any]:
        """
        Extrait des métadonnées supplémentaires à partir du contenu du document.
        
        Args:
            text: Contenu textuel du document
            file_path: Chemin du fichier
            
        Returns:
            Métadonnées supplémentaires
        """
        additional_metadata = {}
        
        # Détection de la langue (utilisation basique, remplacer par un modèle plus robuste en production)
        french_indicators = ['et', 'ou', 'le', 'la', 'un', 'une', 'des', 'les', 'ce', 'cette']
        english_indicators = ['and', 'or', 'the', 'a', 'an', 'of', 'for', 'to', 'in', 'this']
        
        french_count = sum(1 for word in text.lower().split() if word in french_indicators)
        english_count = sum(1 for word in text.lower().split() if word in english_indicators)
        
        if french_count > english_count:
            additional_metadata['language'] = 'french'
        else:
            additional_metadata['language'] = 'english'
        
        # Statistiques basiques sur le document
        words = text.split()
        additional_metadata['word_count'] = len(words)
        additional_metadata['character_count'] = len(text)
        additional_metadata['line_count'] = text.count('\n') + 1
        
        # Estimation du temps de lecture (mots par minute)
        reading_speed = 200  # Mots par minute (moyenne)
        additional_metadata['estimated_reading_time_minutes'] = round(len(words) / reading_speed, 1)
        
        # Extraction basique de mots-clés (fréquence des mots)
        stop_words = set(french_indicators + english_indicators + ['est', 'sont', 'a', 'ont', 'is', 'are', 'has', 'have'])
        words_filtered = [word.lower().strip(string.punctuation) for word in words if word.lower().strip(string.punctuation) not in stop_words and len(word) > 3]
        
        # Compter la fréquence des mots
        word_freq = {}
        for word in words_filtered:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
        
        # Extraire les mots les plus fréquents comme mots-clés
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        additional_metadata['keywords'] = [keyword[0] for keyword in keywords]
        
        return additional_metadata
    
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
    def process_document(self, file_path: Path, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Traite un document, extrait le texte, le nettoie et le découpe en chunks.
        
        Args:
            file_path: Chemin du fichier à traiter
            force_reprocess: Force le retraitement même si le document existe déjà
            
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
        if not force_reprocess and rel_path in self.document_registry["documents"] and \
           self.document_registry["documents"][rel_path]["hash"] == doc_hash:
            logger.info(f"Document {rel_path} déjà traité et inchangé")
            return self.document_registry["documents"][rel_path]
        
        logger.info(f"Traitement du document: {rel_path}")
        
        # Charger le document avec le loader approprié
        loader = self._get_document_loader(file_path)
        pages = loader.load()
        
        # Extraire le texte brut
        raw_text = "\n\n".join([page.page_content for page in pages])
        
        # Nettoyer le texte
        cleaned_text = self._clean_text(raw_text)
        
        # Sauvegarder le texte nettoyé
        cleaned_file = self.processed_dir / "cleaned" / f"{document_id}.txt"
        with open(cleaned_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        # Extraire les métadonnées
        doc_metadata = extract_metadata(file_path, len(pages))
        
        # Extraire des métadonnées supplémentaires à partir du contenu
        additional_metadata = self._extract_additional_metadata(cleaned_text, file_path)
        doc_metadata.update(additional_metadata)
        
        # Nettoyer les métadonnées
        doc_metadata = self._clean_metadata(doc_metadata)
        
        # Utiliser le chunking sémantique
        chunks_data = self.semantic_chunker.split_text(cleaned_text)
        
        # Enrichir les métadonnées de chaque chunk
        enriched_chunks = []
        for i, chunk in enumerate(chunks_data):
            chunk_id = generate_chunk_id(document_id, i)
            
            # Fusionner les métadonnées du document avec celles du chunk
            chunk_metadata = chunk["metadata"].copy()
            chunk_metadata.update({
                "document_id": document_id,
                "chunk_id": chunk_id,
                "chunk_index": i,
                "total_chunks": len(chunks_data),
                **doc_metadata
            })
            
            # Créer un objet chunk enrichi
            enriched_chunks.append({
                "text": chunk["text"],
                "metadata": filter_complex_metadata(chunk_metadata)
            })
        
        # Sauvegarder les chunks
        chunk_file = self.processed_dir / "chunks" / f"{document_id}.json"
        save_json(enriched_chunks, chunk_file)
        
        # Sauvegarder les métadonnées
        metadata_file = self.processed_dir / "metadata" / f"{document_id}.json"
        metadata_info = {
            "metadata": doc_metadata,
            "hash": doc_hash,
            "chunk_count": len(enriched_chunks),
            "chunk_file": str(chunk_file.relative_to(self.processed_dir)),
            "cleaned_file": str(cleaned_file.relative_to(self.processed_dir))
        }
        save_json(metadata_info, metadata_file)
        
        # Mettre à jour le registre
        self.document_registry["documents"][rel_path] = {
            "document_id": document_id,
            "hash": doc_hash,
            "metadata": doc_metadata,
            "chunk_count": len(enriched_chunks),
            "chunk_file": str(chunk_file.relative_to(self.processed_dir)),
            "metadata_file": str(metadata_file.relative_to(self.processed_dir)),
            "cleaned_file": str(cleaned_file.relative_to(self.processed_dir))
        }
        self._save_document_registry()
        
        logger.info(f"Document {rel_path} traité avec succès: {len(enriched_chunks)} chunks générés")
        return self.document_registry["documents"][rel_path]
    
    @timed(logger=logger)
    def process_directory(self, subdirectory: Optional[str] = None, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Traite tous les documents dans le répertoire de données.
        
        Args:
            subdirectory: Sous-répertoire optionnel à traiter
            force_reprocess: Force le retraitement de tous les documents
            
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
                if not force_reprocess and rel_path in self.document_registry["documents"] and \
                   self.document_registry["documents"][rel_path]["hash"] == doc_hash:
                    results["skipped_documents"] += 1
                    return {
                        "path": rel_path,
                        "status": "skipped",
                        "reason": "unchanged"
                    }
                
                # Traiter le document
                doc_result = self.process_document(file_path, force_reprocess)
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