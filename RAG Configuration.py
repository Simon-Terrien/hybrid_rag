import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dotenv import load_dotenv
import logging
import json

# Charger les variables d'environnement
load_dotenv()

# Définir le dossier racine du projet
ROOT_DIR = Path(__file__).parent.absolute()

# Configuration des chemins
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "processed_data"))
VECTOR_DB_DIR = Path(os.getenv("VECTOR_DB_DIR", "vector_db"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", "cache"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))

# S'assurer que les chemins sont absolus
if not DATA_DIR.is_absolute():
    DATA_DIR = ROOT_DIR / DATA_DIR
if not PROCESSED_DIR.is_absolute():
    PROCESSED_DIR = ROOT_DIR / PROCESSED_DIR
if not VECTOR_DB_DIR.is_absolute():
    VECTOR_DB_DIR = ROOT_DIR / VECTOR_DB_DIR
if not CACHE_DIR.is_absolute():
    CACHE_DIR = ROOT_DIR / CACHE_DIR
if not MODEL_DIR.is_absolute():
    MODEL_DIR = ROOT_DIR / MODEL_DIR

# Configuration des embeddings
DEFAULT_EMBEDDING_MODELS = {
    "default": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v1",
    "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "high_quality": "BAAI/bge-large-en-v1.5"
}
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODELS["default"])
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"

# Configuration des cross-encoders pour le reranking
CROSS_ENCODER_MODELS = {
    "default": "cross-encoder/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "multilingual": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
}
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", CROSS_ENCODER_MODELS["default"])

# Configuration de la base vectorielle
VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "faiss").lower()
VECTOR_DB_CONFIG = {
    "faiss": {
        "distance_strategy": "cosine",
        "normalize_embeddings": True
    },
    "chroma": {
        "collection_name": "documents",
        "persist_directory": str(VECTOR_DB_DIR / "chroma"),
        "collection_metadata": {"hnsw:space": "cosine"}
    }
}

# Paramètres de chunking
CHUNKING_CONFIG = {
    "default_chunk_size": int(os.getenv("DEFAULT_CHUNK_SIZE", "1000")),
    "default_chunk_overlap": int(os.getenv("DEFAULT_CHUNK_OVERLAP", "200")),
    "long_document_threshold": int(os.getenv("LONG_DOCUMENT_THRESHOLD", "50")),
    "separators": ["\n\n", "\n", ".", " ", ""],
    "semantic_chunking": os.getenv("SEMANTIC_CHUNKING", "true").lower() == "true",
    "chunk_by_markdown_headers": os.getenv("CHUNK_BY_MARKDOWN_HEADERS", "true").lower() == "true",
    "use_spacy_ner": os.getenv("USE_SPACY_NER", "false").lower() == "true",
    "document_types": {
        "code": {
            "chunk_size": 500,
            "chunk_overlap": 150
        },
        "markdown": {
            "chunk_size": 1200,
            "chunk_overlap": 300
        },
        "pdf": {
            "chunk_size": 1000,
            "chunk_overlap": 200
        }
    }
}

# Paramètres de recherche et récupération
SEARCH_CONFIG = {
    "default_top_k": int(os.getenv("DEFAULT_TOP_K", "5")),
    "similarity_threshold": float(os.getenv("SIMILARITY_THRESHOLD", "0.7")),
    "max_tokens_per_chunk": 1000,
    "reranking_enabled": os.getenv("RERANKING_ENABLED", "false").lower() == "true",
    "hybrid_search_enabled": os.getenv("HYBRID_SEARCH_ENABLED", "true").lower() == "true",
    "semantic_weight": float(os.getenv("SEMANTIC_WEIGHT", "0.7")),
    "lexical_weight": float(os.getenv("LEXICAL_WEIGHT", "0.3")),
    "metadata_fields_boost": {
        "title": 1.5,
        "description": 1.2,
        "source": 1.0
    },
    "context_window": int(os.getenv("CONTEXT_WINDOW", "3"))
}

# Paramètres du modèle LLM
LLM_CONFIG = {
    "model_name": os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2"),
    "local_model": os.getenv("LOCAL_MODEL", "true").lower() == "true",
    "api_base": os.getenv("LLM_API_BASE", ""),
    "api_key": os.getenv("LLM_API_KEY", ""),
    "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", "512")),
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
    "top_p": float(os.getenv("LLM_TOP_P", "0.95")),
    "top_k": int(os.getenv("LLM_TOP_K", "50")),
    "repetition_penalty": float(os.getenv("LLM_REPETITION_PENALTY", "1.1")),
    "use_8bit_quantization": os.getenv("USE_8BIT_QUANTIZATION", "true").lower() == "true",
    "custom_prompt_template": os.getenv("CUSTOM_PROMPT_TEMPLATE", "")
}

# Paramètres de performance
PERFORMANCE_CONFIG = {
    "batch_size": int(os.getenv("BATCH_SIZE", "64")),
    "max_concurrent_tasks": int(os.getenv("MAX_CONCURRENT_TASKS", "4")),
    "cache_embeddings": os.getenv("CACHE_EMBEDDINGS", "true").lower() == "true",
    "cache_dir": str(CACHE_DIR),
    "use_pre_computed_embeddings": os.getenv("USE_PRE_COMPUTED_EMBEDDINGS", "true").lower() == "true",
    "enable_dynamic_batching": os.getenv("ENABLE_DYNAMIC_BATCHING", "true").lower() == "true"
}

# Configuration du logging
LOG_LEVEL = getattr(logging, os.getenv("LOG_LEVEL", "INFO"))
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = PROCESSED_DIR / "system.log"

# Sécurité
SECRET_KEY = os.getenv("SECRET_KEY", "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
ENCRYPT_METADATA = os.getenv("ENCRYPT_METADATA", "false").lower() == "true"
ENCRYPT_CONTENT = os.getenv("ENCRYPT_CONTENT", "false").lower() == "true"

# Chemins des fichiers de registre
DOCUMENT_REGISTRY_PATH = PROCESSED_DIR / "document_registry.json"
VECTOR_REGISTRY_PATH = VECTOR_DB_DIR / "vector_registry.json"

# Configuration des formats de document supportés
SUPPORTED_FORMATS = {
    ".pdf": {
        "loader": "PyPDFLoader",
        "mime_type": "application/pdf",
        "description": "Document PDF"
    },
    ".txt": {
        "loader": "TextLoader",
        "mime_type": "text/plain",
        "description": "Document texte"
    },
    ".docx": {
        "loader": "Docx2txtLoader",
        "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "description": "Document Word"
    },
    ".md": {
        "loader": "TextLoader",
        "mime_type": "text/markdown",
        "description": "Document Markdown"
    },
    ".csv": {
        "loader": "CSVLoader",
        "mime_type": "text/csv",
        "description": "Fichier CSV"
    },
    ".json": {
        "loader": "JSONLoader",
        "mime_type": "application/json",
        "description": "Fichier JSON"
    },
    ".html": {
        "loader": "BSHTMLLoader",
        "mime_type": "text/html",
        "description": "Document HTML"
    },
    ".py": {
        "loader": "TextLoader",
        "mime_type": "text/x-python",
        "description": "Script Python"
    }
}

# Configuration des modèles de prompts
PROMPT_TEMPLATES = {
    "qa_default": """Tu es un assistant IA expert chargé de fournir des réponses précises basées uniquement sur les documents fournis.

Historique de la conversation:
{chat_history}

Contexte des documents:
{context}

Question de l'utilisateur: {question}

Instructions:
1. Réponds uniquement à partir des informations fournies dans le contexte ci-dessus.
2. Si tu ne trouves pas la réponse dans le contexte, dis simplement que tu ne peux pas répondre.
3. Ne fabrique pas d'informations ou de connaissances qui ne sont pas présentes dans le contexte.
4. Cite la source exacte (numéro de document, identifiant) lorsque tu fournis des informations.
5. Présente ta réponse de manière claire et structurée.

Réponds de façon concise à la question suivante en te basant uniquement sur le contexte fourni: {question}""",

    "qa_conversational": """Tu es un assistant IA amical chargé d'aider l'utilisateur à comprendre les documents fournis.

Historique de la conversation:
{chat_history}

Contexte des documents:
{context}

Question de l'utilisateur: {question}

Instructions:
1. Réponds uniquement à partir des informations fournies dans le contexte ci-dessus.
2. Essaie de comprendre l'intention de l'utilisateur dans le contexte de la conversation.
3. Ne fabrique pas d'informations qui ne sont pas présentes dans le contexte.
4. Sois conversationnel et amical, mais reste précis.
5. Si tu n'es pas sûr, précise les limites de ta compréhension.

Réponds à la question de l'utilisateur de manière conversationnelle en te basant sur le contexte fourni:"""
}

def get_config() -> Dict[str, Any]:
    """Retourne la configuration complète sous forme de dictionnaire"""
    return {
        "paths": {
            "root_dir": str(ROOT_DIR),
            "data_dir": str(DATA_DIR),
            "processed_dir": str(PROCESSED_DIR),
            "vector_db_dir": str(VECTOR_DB_DIR),
            "cache_dir": str(CACHE_DIR),
            "model_dir": str(MODEL_DIR)
        },
        "embedding": {
            "model": EMBEDDING_MODEL,
            "models": DEFAULT_EMBEDDING_MODELS,
            "use_gpu": USE_GPU
        },
        "cross_encoder": {
            "model": CROSS_ENCODER_MODEL,
            "models": CROSS_ENCODER_MODELS
        },
        "vector_db": {
            "type": VECTOR_DB_TYPE,
            "config": VECTOR_DB_CONFIG
        },
        "chunking": CHUNKING_CONFIG,
        "search": SEARCH_CONFIG,
        "llm": LLM_CONFIG,
        "performance": PERFORMANCE_CONFIG,
        "logging": {
            "level": logging.getLevelName(LOG_LEVEL),
            "file": str(LOG_FILE)
        },
        "security": {
            "encrypt_metadata": ENCRYPT_METADATA,
            "encrypt_content": ENCRYPT_CONTENT
        },
        "supported_formats": SUPPORTED_FORMATS,
        "prompt_templates": PROMPT_TEMPLATES
    }

def save_config(config: Optional[Dict[str, Any]] = None, path: Optional[Union[str, Path]] = None) -> None:
    """Sauvegarde la configuration actuelle ou spécifiée dans un fichier JSON"""
    if config is None:
        config = get_config()
    
    if path is None:
        path = PROCESSED_DIR / "config.json"
    
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Charge une configuration depuis un fichier JSON"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier de configuration non trouvé: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_required_directories() -> None:
    """Crée les répertoires nécessaires s'ils n'existent pas"""
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
    (PROCESSED_DIR / "chunks").mkdir(exist_ok=True, parents=True)
    (PROCESSED_DIR / "metadata").mkdir(exist_ok=True, parents=True)
    (PROCESSED_DIR / "cleaned").mkdir(exist_ok=True, parents=True)
    VECTOR_DB_DIR.mkdir(exist_ok=True, parents=True)
    (VECTOR_DB_DIR / "backups").mkdir(exist_ok=True, parents=True)
    CACHE_DIR.mkdir(exist_ok=True, parents=True)
    MODEL_DIR.mkdir(exist_ok=True, parents=True)

def setup_logging() -> logging.Logger:
    """Configure et retourne le logger"""
    PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
    
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("document_system")

# Initialiser les répertoires et le logging au chargement du module
create_required_directories()
logger = setup_logging()

if __name__ == "__main__":
    # Afficher et sauvegarder la configuration lors de l'exécution directe
    print("Configuration actuelle:")
    print(json.dumps(get_config(), indent=2))
    
    save_config()
    print(f"Configuration sauvegardée dans {PROCESSED_DIR / 'config.json'}")