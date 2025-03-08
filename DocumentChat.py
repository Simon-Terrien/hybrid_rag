from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import time
from datetime import datetime
import json
import os

# Check if Ollama integration is available
try:
    from langchain_ollama import OllamaLLM
    OLLAMA_AVAILABLE = True
except ImportError:
    try:
        # Try to import from our custom module
        from backupollama.langchain_ollama import OllamaLLM
        OLLAMA_AVAILABLE = True
    except ImportError:
        OLLAMA_AVAILABLE = False

# Try to import LLM components from LangChain
try:
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chains.conversational_retrieval.base import BaseConversationalRetrievalChain
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Importation de la configuration et des utilitaires
from config import (
    PROCESSED_DIR, VECTOR_DB_DIR, logger,
    ROOT_DIR, LLM_CONFIG, OLLAMA_CONFIG
)
from utils import (
    load_json, save_json, timed, log_exceptions
)

class DocumentSearchRetriever:
    """
    Adaptateur du moteur de recherche pour l'interface LangChain Retriever.
    Cette classe permet d'utiliser notre moteur de recherche avec les chaînes LangChain.
    """
    
    def __init__(self, search_engine: Any, k: int = 3):
        """
        Initialise le retriever.
        
        Args:
            search_engine: Moteur de recherche (sémantique ou hybride)
            k: Nombre de documents à récupérer
        """
        self.search_engine = search_engine
        self.k = k
    
    def get_relevant_documents(self, query: str) -> List[Any]:
        """
        Récupère les documents pertinents pour une requête.
        
        Args:
            query: Requête de recherche
            
        Returns:
            Liste de documents pertinents
        """
        # Effectuer la recherche
        search_results = self.search_engine.search(
            query, 
            top_k=self.k, 
            return_all_chunks=True
        )
        
        # Convertir les résultats en Documents
        documents = []
        
        if "results" in search_results:
            for result in search_results["results"]:
                # Create a document-like object that matches LangChain's expected format
                doc = type('Document', (), {})()
                doc.page_content = result["text"]
                doc.metadata = result["metadata"]
                documents.append(doc)
        
        return documents

class DocumentChat:
    """Interface de dialogue avec les documents"""
    
    def __init__(
        self,
        search_engine: Optional[Any] = None,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        conversation_memory_limit: int = 5,
        max_new_tokens: int = 512,
        context_window: int = 3,
        use_ollama: bool = True,
        use_langchain: bool = True
    ):
        """
        Initialise l'interface de dialogue avec les documents.
        
        Args:
            search_engine: Moteur de recherche sémantique ou hybride
            model_name: Nom du modèle de langage à utiliser
            conversation_memory_limit: Nombre de tours de conversation à conserver en mémoire
            max_new_tokens: Nombre maximum de tokens pour la génération de réponses
            context_window: Nombre de chunks contextuels à inclure autour d'un résultat
            use_ollama: Utiliser Ollama si disponible
            use_langchain: Utiliser LangChain si disponible
        """
        # Initialiser le moteur de recherche s'il n'est pas fourni
        self.search_engine = search_engine
        
        # Paramètres de configuration
        self.model_name = model_name
        self.conversation_memory_limit = conversation_memory_limit
        self.max_new_tokens = max_new_tokens
        self.context_window = context_window
        self.use_ollama = use_ollama and OLLAMA_AVAILABLE
        self.use_langchain = use_langchain and LANGCHAIN_AVAILABLE
        
        # Initialiser le LLM et la mémoire de conversation
        self.llm = self._initialize_llm()
        
        if self.use_langchain:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        
        # Historique des conversations
        self.conversations = {}
        self.current_conversation_id = None
        
        logger.info(f"DocumentChat initialisé: model={model_name}, use_ollama={self.use_ollama}, use_langchain={self.use_langchain}")
        
    def _initialize_llm(self):
        """
        Initialise le modèle de langage.
        
        Returns:
            Instance du LLM ou None
        """
        # Si Ollama est activé et disponible, l'utiliser en priorité
        if self.use_ollama and OLLAMA_AVAILABLE:
            try:
                ollama_config = OLLAMA_CONFIG
                logger.info(f"Initialisation d'Ollama LLM: {ollama_config['models']['default']}")
                
                return OllamaLLM(
                    base_url=ollama_config.get("base_url", "http://localhost:11434"),
                    model=ollama_config.get("models", {}).get("default", "mistral"),
                    temperature=ollama_config.get("parameters", {}).get("temperature", 0.7),
                    top_p=ollama_config.get("parameters", {}).get("top_p", 0.95),
                    max_tokens=ollama_config.get("parameters", {}).get("max_tokens", 512)
                )
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation d'Ollama: {str(e)}")
        
        # Si LangChain est disponible, essayer d'utiliser HuggingFace
        if self.use_langchain and LANGCHAIN_AVAILABLE:
            try:
                # Import here to avoid dependencies issues
                from langchain_huggingface import HuggingFacePipeline
                import torch
                from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
                
                # Check if CUDA is available
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Initialisation du LLM {self.model_name} sur {device}")
                
                # Load model and tokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="auto" if device == "cuda" else None,
                    load_in_8bit=device == "cuda",  # 8-bit quantization if GPU available
                )
                
                # Create pipeline
                text_generation_pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                # Create LangChain LLM
                return HuggingFacePipeline(pipeline=text_generation_pipeline)
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation du LLM: {str(e)}")
        
        logger.warning("Fonctionnement en mode dégradé: génération de réponses simple sans LLM")
        return None
    
    def _build_retrieval_chain(self):
        """
        Construit la chaîne de récupération conversationnelle.
        
        Returns:
            Chaîne de récupération LangChain ou None
        """
        if not self.llm or not self.use_langchain:
            return None
        
        # Créer un wrapper pour le moteur de recherche
        retriever = DocumentSearchRetriever(self.search_engine, k=3)
        
        # Template pour la synthèse des documents
        document_prompt = PromptTemplate(
            input_variables=["page_content"],
            template="""Contenu: {page_content}"""
        )
        
        # Template pour la question-réponse
        qa_prompt = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template="""Tu es un assistant IA expert chargé de fournir des réponses précises basées uniquement sur les documents fournis.

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

Réponds de façon concise à la question suivante en te basant uniquement sur le contexte fourni: {question}"""
        )
        
        try:
            # Création de la chaîne de récupération
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": qa_prompt},
                document_prompt=document_prompt
            )
            
            return chain
        except Exception as e:
            logger.error(f"Erreur lors de la création de la chaîne de récupération: {str(e)}")
            return None
    
    @timed(logger=logger)
    @log_exceptions(logger=logger)
    def ask(self, question: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Pose une question au système et obtient une réponse basée sur les documents.
        
        Args:
            question: Question à poser
            conversation_id: Identifiant de la conversation (si None, utilise la conversation courante)
            
        Returns:
            Réponse générée avec les sources
        """
        start_time = time.time()
        
        # Gérer l'identifiant de conversation
        if conversation_id is None:
            if self.current_conversation_id is None:
                # Créer une nouvelle conversation
                self.current_conversation_id = f"conv_{int(time.time())}"
                self.conversations[self.current_conversation_id] = {
                    "id": self.current_conversation_id,
                    "created_at": datetime.now().isoformat(),
                    "last_update": datetime.now().isoformat(),
                    "messages": []
                }
            conversation_id = self.current_conversation_id
        else:
            # Vérifier si la conversation existe
            if conversation_id not in self.conversations:
                raise ValueError(f"Conversation non trouvée: {conversation_id}")
            self.current_conversation_id = conversation_id
        
        # Préparer la réponse
        response = {
            "conversation_id": conversation_id,
            "question": question,
            "timestamp": datetime.now().isoformat()
        }
        
        # Mode de fonctionnement: avec LLM ou simplifié
        if self.llm is not None and self.use_langchain:
            try:
                # Construire la chaîne de récupération si nécessaire
                chain = self._build_retrieval_chain()
                
                if chain:
                    # Exécuter la chaîne pour obtenir une réponse
                    chain_response = chain({"question": question})
                    
                    # Extraire la réponse et les documents sources
                    answer = chain_response["answer"]
                    source_documents = chain_response.get("source_documents", [])
                    
                    # Préparer les sources pour la réponse
                    sources = []
                    for doc in source_documents:
                        sources.append({
                            "document_id": doc.metadata.get("document_id"),
                            "chunk_id": doc.metadata.get("chunk_id"),
                            "source": doc.metadata.get("source", ""),
                            "page": doc.metadata.get("page", 0),
                            "text_preview": doc.page_content[:100] + "..."
                        })
                    
                    response["answer"] = answer
                    response["sources"] = sources
                else:
                    # En cas d'échec de la chaîne, utiliser le mode simplifié
                    simplified_response = self._simplified_response(question, conversation_id, start_time)
                    response.update(simplified_response)
            except Exception as e:
                logger.error(f"Erreur lors de la génération de réponse avec LLM: {str(e)}")
                # Fallback au mode simplifié
                simplified_response = self._simplified_response(question, conversation_id, start_time)
                response.update(simplified_response)
        else:
            # Mode simplifié (sans LLM)
            simplified_response = self._simplified_response(question, conversation_id, start_time)
            response.update(simplified_response)
        
        # Mettre à jour l'historique de la conversation
        self.conversations[conversation_id]["messages"].append({
            "role": "user",
            "content": question,
            "timestamp": response["timestamp"]
        })
        
        self.conversations[conversation_id]["messages"].append({
            "role": "assistant",
            "content": response["answer"],
            "sources": response.get("sources", []),
            "timestamp": datetime.now().isoformat()
        })
        
        self.conversations[conversation_id]["last_update"] = datetime.now().isoformat()
        
        # Limiter la taille de l'historique
        if len(self.conversations[conversation_id]["messages"]) > self.conversation_memory_limit * 2:
            # Conserver les n derniers échanges (question + réponse)
            self.conversations[conversation_id]["messages"] = \
                self.conversations[conversation_id]["messages"][-self.conversation_memory_limit * 2:]
        
        response["processing_time_seconds"] = round(time.time() - start_time, 3)
        return response
    
    def _simplified_response(self, question: str, conversation_id: str, start_time: float) -> Dict[str, Any]:
        """
        Génère une réponse simplifiée basée uniquement sur la recherche (sans LLM).
        
        Args:
            question: Question posée
            conversation_id: Identifiant de la conversation
            start_time: Temps de début du traitement
            
        Returns:
            Réponse simplifiée avec les sources
        """
        # Effectuer une recherche
        search_results = self.search_engine.search(question, top_k=3)
        
        # Extraire les chunks les plus pertinents
        if search_results.get("total_results", 0) > 0:
            # Récupérer les contextes pour les meilleurs chunks
            contexts = []
            sources = []
            
            for i, result in enumerate(search_results["results"]):
                # Pour les résultats regroupés par document
                if "chunks" in result:
                    for chunk in result["chunks"][:2]:  # Prendre les 2 meilleurs chunks par document
                        # Récupérer le contexte autour du chunk
                        if hasattr(self.search_engine, 'get_document_context'):
                            context = self.search_engine.get_document_context(
                                result["document_id"], 
                                chunk["chunk_id"],
                                window_size=self.context_window
                            )
                            
                            contexts.append({
                                "document_id": result["document_id"],
                                "chunk_id": chunk["chunk_id"],
                                "text": chunk["text"],
                                "context": [c["text"] for c in context["context"]]
                            })
                        else:
                            contexts.append({
                                "document_id": result["document_id"],
                                "chunk_id": chunk["chunk_id"],
                                "text": chunk["text"],
                                "context": []
                            })
                        
                        sources.append({
                            "document_id": result["document_id"],
                            "chunk_id": chunk["chunk_id"],
                            "source": result["metadata"].get("source", ""),
                            "page": chunk["metadata"].get("page", 0),
                            "text_preview": chunk["text"][:100] + "..."
                        })
                # Pour les résultats de chunks individuels
                elif "text" in result:
                    contexts.append({
                        "document_id": result["document_id"],
                        "chunk_id": result["chunk_id"],
                        "text": result["text"],
                        "context": []
                    })
                    
                    sources.append({
                        "document_id": result["document_id"],
                        "chunk_id": result["chunk_id"],
                        "source": result["metadata"].get("source", ""),
                        "page": result["metadata"].get("page", 0),
                        "text_preview": result["text"][:100] + "..."
                    })
            
            # Construire une réponse simplifiée
            if contexts:
                answer = "Voici les passages les plus pertinents trouvés dans les documents :\n\n"
                for i, ctx in enumerate(contexts[:3], 1):  # Limiter à 3 contextes
                    answer += f"**Extrait {i}** (Document {ctx['document_id']}):\n"
                    answer += ctx["text"] + "\n\n"
                
                answer += "\nPour une réponse plus élaborée, veuillez activer le mode LLM."
            else:
                answer = "Aucune information pertinente n'a été trouvée dans les documents."
                sources = []
        else:
            answer = "Aucune information pertinente n'a été trouvée dans les documents."
            sources = []
        
        return {
            "conversation_id": conversation_id,
            "question": question,
            "answer": answer,
            "sources": sources,
            "timestamp": datetime.now().isoformat(),
            "processing_time_seconds": round(time.time() - start_time, 3),
            "mode": "simplified"
        }
    
    def new_conversation(self) -> str:
        """
        Crée une nouvelle conversation.
        
        Returns:
            Identifiant de la nouvelle conversation
        """
        conversation_id = f"conv_{int(time.time())}"
        self.conversations[conversation_id] = {
            "id": conversation_id,
            "created_at": datetime.now().isoformat(),
            "last_update": datetime.now().isoformat(),
            "messages": []
        }
        
        self.current_conversation_id = conversation_id
        
        if self.use_langchain:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        
        logger.info(f"Nouvelle conversation créée: {conversation_id}")
        return conversation_id
    
    def get_conversation(self, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Récupère une conversation.
        
        Args:
            conversation_id: Identifiant de la conversation (si None, retourne la conversation courante)
            
        Returns:
            Détails de la conversation
        """
        if conversation_id is None:
            conversation_id = self.current_conversation_id
            
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation non trouvée: {conversation_id}")
            
        return self.conversations[conversation_id]
    
    def list_conversations(self) -> List[Dict[str, Any]]:
        """
        Liste toutes les conversations.
        
        Returns:
            Liste des résumés de conversations
        """
        return [
            {
                "id": conv_id,
                "created_at": conv["created_at"],
                "last_update": conv["last_update"],
                "message_count": len(conv["messages"]),
                "preview": conv["messages"][0]["content"][:50] + "..." if conv["messages"] else ""
            }
            for conv_id, conv in self.conversations.items()
        ]
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Supprime une conversation.
        
        Args:
            conversation_id: Identifiant de la conversation
            
        Returns:
            True si la suppression a réussi, False sinon
        """
        if conversation_id not in self.conversations:
            return False
            
        del self.conversations[conversation_id]
        
        if self.current_conversation_id == conversation_id:
            self.current_conversation_id = None
            
        logger.info(f"Conversation supprimée: {conversation_id}")
        return True
    
    def save_conversations(self, file_path: Optional[Union[str, Path]] = None) -> str:
        """
        Sauvegarde toutes les conversations dans un fichier.
        
        Args:
            file_path: Chemin du fichier de sauvegarde (si None, utilise un nom par défaut)
            
        Returns:
            Chemin du fichier de sauvegarde
        """
        if file_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_path = PROCESSED_DIR / "conversations" / f"conversations_{timestamp}.json"
            
        file_path = Path(file_path)
        file_path.parent.mkdir(exist_ok=True, parents=True)
        
        save_json(self.conversations, file_path)
        logger.info(f"Conversations sauvegardées dans {file_path}")
        
        return str(file_path)
    
    def load_conversations(self, file_path: Union[str, Path]) -> int:
        """
        Charge les conversations depuis un fichier.
        
        Args:
            file_path: Chemin du fichier de conversations
            
        Returns:
            Nombre de conversations chargées
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier de conversations non trouvé: {file_path}")
            
        loaded_conversations = load_json(file_path)
        self.conversations.update(loaded_conversations)
        
        logger.info(f"{len(loaded_conversations)} conversations chargées depuis {file_path}")
        return len(loaded_conversations)


# Exemple d'utilisation
if __name__ == "__main__":
    # Import the search engine
    from semantic_search import SemanticSearchEngine
    from vector_indexer import VectorIndexer
    
    # Initialize components
    vector_indexer = VectorIndexer()
    search_engine = SemanticSearchEngine(vector_indexer=vector_indexer)
    
    # Initialize document chat
    document_chat = DocumentChat(search_engine=search_engine)
    
    # Create a new conversation
    conversation_id = document_chat.new_conversation()
    print(f"Nouvelle conversation: {conversation_id}")
    
    # Ask a question
    question = "Quelles sont les principales fonctionnalités du système?"
    response = document_chat.ask(question)
    
    print(f"Question: {question}")
    print(f"Réponse: {response['answer']}")
    print(f"Temps de traitement: {response['processing_time_seconds']} secondes")
    
    if "sources" in response and response["sources"]:
        print("\nSources:")
        for source in response["sources"]:
            print(f"  - Document: {source['document_id']}, Chunk: {source['chunk_id']}")
            print(f"    Aperçu: {source['text_preview']}")
    
    # Ask a follow-up question
    follow_up = "Peux-tu me donner plus de détails sur la phase 2?"
    response = document_chat.ask(follow_up)
    
    print(f"\nQuestion de suivi: {follow_up}")
    print(f"Réponse: {response['answer']}")