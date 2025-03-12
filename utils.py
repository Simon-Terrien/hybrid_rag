import os
import json
import hashlib
import time
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime
import logging
import functools
import traceback
from cryptography.fernet import Fernet
import base64
import spacy
from spacy.cli import download as spacy_download
# Import de la configuration
from config import logger, SECRET_KEY, ENCRYPT_METADATA, ENCRYPT_CONTENT
def load_spacy_model(model_name: str):
    """
    Attempt to load a spaCy model. If not found, download it and then load.
    """
    try:
        return spacy.load(model_name)
    except OSError:
        logging.warning(f"Model {model_name} not found. Downloading...")
        spacy_download(model_name)
        return spacy.load(model_name)
# Fonctions de décoration pour la répétition et le timing
def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0, 
          exceptions: tuple = (Exception,), logger: Optional[logging.Logger] = None):
    """
    Décorateur pour réessayer une fonction en cas d'exception.
    
    Args:
        max_attempts: Nombre maximum de tentatives
        delay: Délai initial entre les tentatives (secondes)
        backoff: Facteur multiplicatif pour le délai entre chaque tentative
        exceptions: Tuple d'exceptions qui déclenchent une nouvelle tentative
        logger: Logger pour enregistrer les exceptions
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            local_logger = logger or logging.getLogger(func.__module__)
            attempt = 0
            current_delay = delay
            
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        local_logger.error(f"Échec définitif de {func.__name__} après {max_attempts} tentatives: {str(e)}")
                        raise
                    
                    local_logger.warning(f"Tentative {attempt}/{max_attempts} de {func.__name__} a échoué: {str(e)}. Nouvelle tentative dans {current_delay:.1f}s")
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            return func(*args, **kwargs)  # Dernière tentative
        return wrapper
    return decorator

def timed(logger: Optional[logging.Logger] = None):
    """
    Décorateur pour mesurer et enregistrer le temps d'exécution d'une fonction.
    
    Args:
        logger: Logger pour enregistrer le temps d'exécution
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            local_logger = logger or logging.getLogger(func.__module__)
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            local_logger.info(f"{func.__name__} exécuté en {execution_time:.2f} secondes")
            return result
        return wrapper
    return decorator

def log_exceptions(logger: Optional[logging.Logger] = None, reraise: bool = True):
    """
    Décorateur pour capturer et enregistrer les exceptions.
    
    Args:
        logger: Logger pour enregistrer l'exception
        reraise: Si True, relève l'exception après l'enregistrement
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            local_logger = logger or logging.getLogger(func.__module__)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Capturer la trace compète
                tb = traceback.format_exc()
                local_logger.error(f"Exception dans {func.__name__}: {str(e)}\n{tb}")
                if reraise:
                    raise
        return wrapper
    return decorator

# Fonctions de hachage et de cryptage
def compute_file_hash(file_path: Union[str, Path]) -> str:
    """
    Calcule un hash SHA-256 pour un fichier.
    
    Args:
        file_path: Chemin du fichier
    
    Returns:
        Hash SHA-256 du fichier sous forme de chaîne hexadécimale
    """
    hash_sha256 = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    
    return hash_sha256.hexdigest()

def compute_string_hash(content: str) -> str:
    """
    Calcule un hash SHA-256 pour une chaîne de caractères.
    
    Args:
        content: Contenu à hacher
    
    Returns:
        Hash SHA-256 du contenu sous forme de chaîne hexadécimale
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def get_encryption_key() -> bytes:
    """
    Génère une clé de chiffrement à partir de la clé secrète.
    
    Returns:
        Clé de chiffrement pour Fernet
    """
    digest = hashlib.sha256(SECRET_KEY.encode()).digest()
    return base64.urlsafe_b64encode(digest)

def encrypt_data(data: str) -> str:
    """
    Chiffre des données en utilisant Fernet.
    
    Args:
        data: Données à chiffrer
    
    Returns:
        Données chiffrées en base64
    """
    key = get_encryption_key()
    cipher = Fernet(key)
    return cipher.encrypt(data.encode('utf-8')).decode('utf-8')

def decrypt_data(encrypted_data: str) -> str:
    """
    Déchiffre des données chiffrées avec Fernet.
    
    Args:
        encrypted_data: Données chiffrées en base64
    
    Returns:
        Données déchiffrées
    """
    key = get_encryption_key()
    cipher = Fernet(key)
    return cipher.decrypt(encrypted_data.encode('utf-8')).decode('utf-8')

# Fonctions de gestion de fichiers JSON
@retry(max_attempts=3, delay=0.5, exceptions=(IOError,), logger=logger)
def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Charge un fichier JSON avec gestion d'erreur et retry.
    
    Args:
        file_path: Chemin du fichier JSON
    
    Returns:
        Contenu du fichier JSON sous forme de dictionnaire
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

@retry(max_attempts=3, delay=0.5, exceptions=(IOError,), logger=logger)
def save_json(data: Dict[str, Any], file_path: Union[str, Path], 
              backup: bool = True, encrypt: bool = False) -> None:
    """
    Sauvegarde des données dans un fichier JSON avec backup optionnel.
    
    Args:
        data: Données à sauvegarder
        file_path: Chemin du fichier JSON
        backup: Si True, crée une sauvegarde du fichier existant avant l'écrasement
        encrypt: Si True, chiffre les données avant de les sauvegarder
    """
    file_path = Path(file_path)
    
    # Créer une sauvegarde si le fichier existe
    if backup and file_path.exists():
        backup_dir = file_path.parent / "backups"
        backup_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"{file_path.stem}_{timestamp}{file_path.suffix}"
        shutil.copy2(file_path, backup_file)
        logger.debug(f"Backup créé: {backup_file}")
    
    # Préparer les données
    content = json.dumps(data, indent=2, ensure_ascii=False)
    
    # Chiffrer si nécessaire
    if encrypt:
        content = encrypt_data(content)
    
    # Sauvegarder
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.debug(f"Fichier sauvegardé: {file_path}")

# Fonctions de gestion des métadonnées
def extract_metadata(file_path: Union[str, Path], 
                     page_count: Optional[int] = None) -> Dict[str, Any]:
    """
    Extrait les métadonnées d'un fichier.
    
    Args:
        file_path: Chemin du fichier
        page_count: Nombre de pages (pour les documents paginés)
    
    Returns:
        Dictionnaire de métadonnées
    """
    file_path = Path(file_path)
    
    metadata = {
        "filename": file_path.name,
        "extension": file_path.suffix.lower(),
        "file_size_bytes": os.path.getsize(file_path),
        "creation_date": datetime.fromtimestamp(os.path.getctime(file_path)).isoformat(),
        "modification_date": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
        "processing_date": datetime.now().isoformat(),
        "hash": compute_file_hash(file_path)
    }
    
    if page_count is not None:
        metadata["page_count"] = page_count
        metadata["document_type"] = "long" if page_count > 50 else "short"
    
    if ENCRYPT_METADATA:
        metadata = {k: encrypt_data(str(v)) for k, v in metadata.items()}
    
    return metadata

def filter_complex_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filtre les métadonnées complexes qui ne peuvent pas être sérialisées en JSON.
    
    Args:
        metadata: Dictionnaire de métadonnées
    
    Returns:
        Dictionnaire de métadonnées filtrées
    """
    filtered = {}
    
    for key, value in metadata.items():
        # Conserver uniquement les types sérialisables
        if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
            # Pour les conteneurs, vérifier récursivement
            if isinstance(value, dict):
                filtered[key] = filter_complex_metadata(value)
            elif isinstance(value, list):
                # Filtrer les éléments de liste qui sont des dictionnaires
                filtered[key] = [
                    filter_complex_metadata(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                filtered[key] = value
    
    return filtered

# Fonctions de chunking adaptatif
def adapt_chunking_parameters(doc_length: int, config: Dict[str, Any]) -> Dict[str, int]:
    """
    Adapte les paramètres de chunking en fonction de la longueur du document.
    
    Args:
        doc_length: Longueur du document (en pages)
        config: Configuration de base pour le chunking
    
    Returns:
        Paramètres de chunking adaptés
    """
    threshold = config.get("long_document_threshold", 50)
    
    if doc_length > threshold:
        # Documents longs: chunks plus grands, plus de chevauchement
        chunk_size = int(config.get("default_chunk_size", 1000) * 1.5)
        chunk_overlap = int(config.get("default_chunk_overlap", 200) * 2)
    else:
        # Documents courts: paramètres standards
        chunk_size = config.get("default_chunk_size", 1000)
        chunk_overlap = config.get("default_chunk_overlap", 200)
    
    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap
    }

# Fonctions de gestion des dates
def parse_iso_date(date_string: str) -> datetime:
    """
    Convertit une chaîne de date ISO en objet datetime.
    
    Args:
        date_string: Date au format ISO
    
    Returns:
        Objet datetime
    """
    try:
        return datetime.fromisoformat(date_string)
    except (ValueError, TypeError):
        logger.warning(f"Format de date invalide: {date_string}")
        return datetime.now()

def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Formate un objet datetime en chaîne.
    
    Args:
        dt: Objet datetime
        format_str: Format de date
    
    Returns:
        Date formatée
    """
    return dt.strftime(format_str)

def get_time_delta(start_time: Union[datetime, str], 
                  end_time: Optional[Union[datetime, str]] = None) -> float:
    """
    Calcule la différence de temps en secondes.
    
    Args:
        start_time: Horodatage de début
        end_time: Horodatage de fin (défaut: maintenant)
    
    Returns:
        Différence en secondes
    """
    if isinstance(start_time, str):
        start_time = parse_iso_date(start_time)
    
    if end_time is None:
        end_time = datetime.now()
    elif isinstance(end_time, str):
        end_time = parse_iso_date(end_time)
    
    return (end_time - start_time).total_seconds()

# Génération d'identifiants uniques
def generate_document_id(file_path: Union[str, Path], content_hash: Optional[str] = None) -> str:
    """
    Génère un identifiant unique pour un document.
    
    Args:
        file_path: Chemin du fichier
        content_hash: Hash du contenu (si déjà calculé)
    
    Returns:
        Identifiant unique du document
    """
    if content_hash is None:
        content_hash = compute_file_hash(file_path)
    
    return content_hash[:10]

def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    """
    Génère un identifiant unique pour un chunk.
    
    Args:
        document_id: Identifiant du document
        chunk_index: Index du chunk
    
    Returns:
        Identifiant unique du chunk
    """
    return f"{document_id}-{chunk_index:04d}"

# Utilitaires de conversion
def to_bytes(text: str) -> bytes:
    """Convertit une chaîne en bytes"""
    return text.encode('utf-8')

def from_bytes(data: bytes) -> str:
    """Convertit des bytes en chaîne"""
    return data.decode('utf-8')

# Fonctions avec feedback sur la progression
def process_with_progress(items: List[Any], 
                          process_func: Callable[[Any], Any], 
                          description: str = "Traitement", 
                          logger: Optional[logging.Logger] = None) -> List[Any]:
    """
    Traite une liste d'éléments avec suivi de progression.
    
    Args:
        items: Liste d'éléments à traiter
        process_func: Fonction à appliquer à chaque élément
        description: Description du traitement
        logger: Logger pour le suivi
    
    Returns:
        Liste des résultats
    """
    local_logger = logger or logging.getLogger(__name__)
    total = len(items)
    results = []
    
    local_logger.info(f"Début du {description}: 0/{total} (0.0%)")
    start_time = time.time()
    
    for i, item in enumerate(items):
        try:
            result = process_func(item)
            results.append(result)
        except Exception as e:
            local_logger.error(f"Erreur lors du traitement de l'élément {i}: {str(e)}")
            results.append(None)
        
        # Afficher la progression tous les 5% ou au moins toutes les 10 itérations
        if (i + 1) % max(1, total // 20) == 0 or (i + 1) % 10 == 0 or i + 1 == total:
            elapsed_time = time.time() - start_time
            eta = (elapsed_time / (i + 1)) * (total - i - 1) if i < total - 1 else 0
            percentage = ((i + 1) / total) * 100
            local_logger.info(
                f"Progression du {description}: {i+1}/{total} "
                f"({percentage:.1f}%) - Temps écoulé: {elapsed_time:.1f}s - "
                f"ETA: {eta:.1f}s"
            )
    
    total_time = time.time() - start_time
    local_logger.info(f"{description} terminé en {total_time:.2f} secondes")
    
    return results

# Tests unitaires si exécuté directement
if __name__ == "__main__":
    print("Test des fonctions utilitaires...")
    
    # Test de hachage
    test_string = "Ceci est un test"
    hash_result = compute_string_hash(test_string)
    print(f"Hash de '{test_string}': {hash_result}")
    
    # Test de chiffrement
    encrypted = encrypt_data(test_string)
    decrypted = decrypt_data(encrypted)
    print(f"Original: '{test_string}'")
    print(f"Chiffré: '{encrypted}'")
    print(f"Déchiffré: '{decrypted}'")
    
    # Test des décorateurs
    @timed()
    @retry(max_attempts=2)
    def test_function():
        print("Fonction de test exécutée")
        # Simuler une erreur la première fois
        if not hasattr(test_function, 'called'):
            test_function.called = True
            raise ValueError("Erreur simulée")
        return "Succès"
    
    result = test_function()
    print(f"Résultat: {result}")
    
    print("Tests terminés")