import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dotenv import load_dotenv
import logging
import json

# Charger les variables d'environnement depuis le fichier .env
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
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v1")
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"

# Configuration des models d'embedding disponibles
EMBEDDING_MODELS = {
    "default": "paraphrase-multilingual-MiniLM-L12-v1",
    "high_quality": "all-mpnet-base-v2",
    "fast": "paraphrase-multilingual-MiniLM-L12-v1",
    "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",
    "multilingual_high_quality": "paraphrase-multilingual-mpnet-base-v2",
    "semantic_search": "multi-qa-MiniLM-L6-cos-v1"
}

# Configuration des cross-encoders pour le reranking
CROSS_ENCODER_MODELS = {
    "default": "cross-encoder/ms-marco-MiniLM-L-6-v2",
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

# Paramètres de performance
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "4"))

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
    }
}

# Configuration de la recherche
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

# Configuration du modèle LLM
LLM_CONFIG = {
    "provider": os.getenv("LLM_PROVIDER", "ollama"),
    "ollama": {
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "model": os.getenv("OLLAMA_MODEL", "mistral"),
        "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.7")),
        "top_p": float(os.getenv("OLLAMA_TOP_P", "0.95")),
        "max_tokens": int(os.getenv("OLLAMA_MAX_TOKENS", "512")),
        "system_prompt": os.getenv("OLLAMA_SYSTEM_PROMPT", "You are an AI assistant that provides accurate information based solely on the provided context. If the information is not in the context, acknowledge that you don't know.")
    },
    "huggingface": {
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
}

# Configuration des performances
PERFORMANCE_CONFIG = {
    "batch_size": BATCH_SIZE,
    "max_concurrent_tasks": MAX_CONCURRENT_TASKS,
    "cache_embeddings": os.getenv("CACHE_EMBEDDINGS", "true").lower() == "true",
    "cache_dir": str(CACHE_DIR),
    "use_pre_computed_embeddings": os.getenv("USE_PRE_COMPUTED_EMBEDDINGS", "true").lower() == "true",
    "enable_dynamic_batching": os.getenv("ENABLE_DYNAMIC_BATCHING", "true").lower() == "true"
}

# Configuration de l'intégration Docling
DOCLING_CONFIG = {
    "use_docling": os.getenv("USE_DOCLING", "true").lower() == "true",
    "use_langchain": os.getenv("USE_LANGCHAIN", "true").lower() == "true",
    "use_hybrid_search": os.getenv("USE_HYBRID_SEARCH", "true").lower() == "true",
    "use_reranker": os.getenv("USE_RERANKER", "true").lower() == "true",
    "docling": {
        "enable_enrichments": os.getenv("DOCLING_ENABLE_ENRICHMENTS", "true").lower() == "true",
        "use_docling_chunking": os.getenv("DOCLING_USE_CHUNKING", "true").lower() == "true",
        "enrichments": {
            "do_code_enrichment": os.getenv("DOCLING_CODE_ENRICHMENT", "true").lower() == "true",
            "do_formula_enrichment": os.getenv("DOCLING_FORMULA_ENRICHMENT", "true").lower() == "true",
            "do_picture_classification": os.getenv("DOCLING_PICTURE_CLASSIFICATION", "true").lower() == "true",
            "do_picture_description": os.getenv("DOCLING_PICTURE_DESCRIPTION", "false").lower() == "true",
            "do_table_structure": os.getenv("DOCLING_TABLE_STRUCTURE", "true").lower() == "true"
        }
    },
    "langchain": {
        "embedding_model": EMBEDDING_MODEL,
        "vector_store_type": VECTOR_DB_TYPE,
        "default_vector_store_id": "default",
        "prompt_templates": {
            "default": "Tu es un assistant IA expert chargé de fournir des réponses précises basées uniquement sur les documents fournis.\n\nContexte des documents:\n{context}\n\nQuestion de l'utilisateur: {question}\n\nInstructions:\n1. Réponds uniquement à partir des informations fournies dans le contexte ci-dessus.\n2. Si tu ne trouves pas la réponse dans le contexte, dis simplement que tu ne peux pas répondre.\n3. Ne fabrique pas d'informations ou de connaissances qui ne sont pas présentes dans le contexte.\n4. Cite la source exacte lorsque tu fournis des informations.\n5. Présente ta réponse de manière claire et structurée.\n\nRéponds de façon concise à la question suivante en te basant uniquement sur le contexte fourni:"
        }
    }
}

# Configuration de l'intégration Ollama
OLLAMA_CONFIG = {
    "enabled": os.getenv("OLLAMA_ENABLED", "true").lower() == "true",
    "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    "models": {
        "default": os.getenv("OLLAMA_DEFAULT_MODEL", "mistral"),
        "high_quality": os.getenv("OLLAMA_HQ_MODEL", "llama3"),
        "fast": os.getenv("OLLAMA_FAST_MODEL", "mistral:7b-instruct-v0.2-q4_0"),
        "large": os.getenv("OLLAMA_LARGE_MODEL", "llama3:70b-instruct-q4_K_M")
    },
    "parameters": {
        "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.7")),
        "top_p": float(os.getenv("OLLAMA_TOP_P", "0.95")),
        "max_tokens": int(os.getenv("OLLAMA_MAX_TOKENS", "512")),
        "response_format": {
            "type": "text"
        }
    }
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
            "models": EMBEDDING_MODELS,
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
        "docling_integration": DOCLING_CONFIG,
        "ollama_integration": OLLAMA_CONFIG
    }

def save_config(path: Optional[Union[str, Path]] = None) -> None:
    """Sauvegarde la configuration actuelle dans un fichier JSON"""
    if path is None:
        path = PROCESSED_DIR / "config.json"
    
    # Créer le répertoire parent s'il n'existe pas
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    config = get_config()
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

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
    # Créer le répertoire du journal s'il n'existe pas
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    
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
    print(json.dumps(get_config(), indent=2, ensure_ascii=False))
    
    save_config()
    print(f"Configuration sauvegardée dans {PROCESSED_DIR / 'config.json'}")