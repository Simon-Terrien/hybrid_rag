import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import time
from datetime import datetime

# LangGraph imports for modern memory management
try:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.prebuilt import create_react_agent
    from langgraph.store.memory import InMemoryStore
    from langgraph.utils.config import get_store
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

# LangMem for explicit memory management
try:
    from langmem import create_manage_memory_tool
    LANGMEM_AVAILABLE = True
except ImportError:
    LANGMEM_AVAILABLE = False

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
    from pydantic import Field
    from typing import Any, List
    from langchain.schema.retriever import BaseRetriever
    from langchain.schema import Document
    from langchain.prompts import PromptTemplate
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chains.conversational_retrieval.base import BaseConversationalRetrievalChain
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    from langchain.schema.retriever import BaseRetriever  # For proper type inheritance
    from langchain.schema import Document  # For proper document handling
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Import configuration and utilities
from config import (
    PROCESSED_DIR, logger,
    OLLAMA_CONFIG
)
from utils import (
    load_json, save_json, timed, log_exceptions
)

class DocumentSearchRetriever(BaseRetriever):
    """
    Adapter for the search engine to implement LangChain's BaseRetriever interface.
    This allows our search engine to work with LangChain's chains.
    """
    search_engine: Any = Field(description="The search engine to use for retrieval")
    k: int = Field(default=3, description="Number of documents to retrieve")
    # For Pydantic compatibility, we need to provide model configuration
    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "forbid",
    }
    
    # Note: We remove the field declaration here since it's causing issues
    # with Pydantic's initialization. Instead, we'll use standard Python attributes.
    
    def __init__(self, search_engine: Any, k: int = 3):
        """
        Initialize the retriever.
        
        Args:
            search_engine: Search engine (semantic or hybrid)
            k: Number of documents to retrieve
        """
        # Initialize the parent class with any kwargs
        super().__init__()
        
        # Set attributes directly
        self.search_engine = search_engine
        self.k = k
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            
        Returns:
            List of relevant documents
        """
        # Execute the search
        search_results = self.search_engine.search(
            query, 
            top_k=self.k, 
            return_all_chunks=True
        )
        
        # Convert the results to LangChain Documents
        documents = []
        
        if "results" in search_results:
            for result in search_results["results"]:
                # Create a proper LangChain Document
                doc = Document(
                    page_content=result["text"],
                    metadata=result["metadata"]
                )
                documents.append(doc)
        
        return documents

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """
        Asynchronous version of get_relevant_documents (required by BaseRetriever interface)
        """
        # For simplicity, we call the sync version
        return self.get_relevant_documents(query)

class DocumentChat:
    """Interface for dialogue with documents"""
    
    def __init__(
        self,
        search_engine: Optional[Any] = None,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        conversation_memory_limit: int = 5,
        max_new_tokens: int = 512,
        context_window: int = 3,
        use_ollama: bool = True,
        use_langchain: bool = True,
        use_langgraph: bool = True
    ):
        """
        Initialize the document chat interface.
        
        Args:
            search_engine: Semantic or hybrid search engine
            model_name: Language model name
            conversation_memory_limit: Number of conversation turns to keep in memory
            max_new_tokens: Maximum number of tokens for response generation
            context_window: Number of contextual chunks to include around a result
            use_ollama: Use Ollama if available
            use_langchain: Use LangChain if available
            use_langgraph: Use LangGraph if available
        """
        # Initialize search engine
        self.search_engine = search_engine
        
        # Configuration parameters
        self.model_name = model_name
        self.conversation_memory_limit = conversation_memory_limit
        self.max_new_tokens = max_new_tokens
        self.context_window = context_window
        self.use_ollama = use_ollama and OLLAMA_AVAILABLE
        self.use_langchain = use_langchain and LANGCHAIN_AVAILABLE
        self.use_langgraph = use_langgraph and LANGGRAPH_AVAILABLE
        
        # Initialize LLM and conversation memory based on available libraries
        if self.use_langgraph:
            # Initialize LangGraph components
            self._initialize_langgraph()
        else:
            # Initialize traditional LLM
            self.llm = self._initialize_llm()
            
            if self.use_langchain:
                self.memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="answer"
                )
        
        # Conversation history
        self.conversations = {}
        self.current_conversation_id = None
        
        logger.info(f"DocumentChat initialized: model={model_name}, "
                   f"use_ollama={self.use_ollama}, "
                   f"use_langchain={self.use_langchain}, "
                   f"use_langgraph={self.use_langgraph}")
    
    def _initialize_langgraph(self):
        """
        Initialize LangGraph components for memory management
        """
        if not LANGGRAPH_AVAILABLE:
            logger.error("LangGraph not available. Using traditional LLM.")
            self.use_langgraph = False
            self.llm = self._initialize_llm()
            return
        
        try:
            local_embed_model ="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            # Initialize memory store
            self.store = InMemoryStore(
                index={
                    "dims": 1536,  # Vector dimensions
                    "embed":local_embed_model ,  # Could be configured in config.py
                }
            )
            
            # Initialize checkpointer
            self.checkpointer = MemorySaver()
            
            # Create document retrieval tool
            self.search_tool = self._create_document_retrieval_tool()
            
            # Define prompt function
            def prompt_fn(state):
                """Prepare messages for the LLM including memory."""
                # Get store from configured contextvar
                store = get_store()  # Same as that provided to create_react_agent
                
                # Search for relevant memories
                memories = ""
                try:
                    if len(state["messages"]) > 0 and state["messages"][-1].content:
                        memories = store.search(
                            # Search within the memories namespace
                            ("memories",),
                            query=state["messages"][-1].content,
                        )
                except Exception as e:
                    logger.error(f"Error searching memories: {str(e)}")
                
                system_msg = f"""You are a helpful assistant that answers questions based on provided documents.

## Retrieved Documents
<documents>
{memories}
</documents>

When answering:
1. Use ONLY information from the retrieved documents above
2. If you don't find the answer in the documents, say you don't know
3. Don't make up information that isn't in the documents
4. Cite document IDs when providing information
5. Be clear and concise
"""
                return [{"role": "system", "content": system_msg}, *state["messages"]]
            
            # Create tools list
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "search_documents",
                        "description": "Search for relevant documents based on a query",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query"
                                }
                            },
                            "required": ["query"]
                        }
                    }
                }
            ]
            
            # Add memory management tool if available
            if LANGMEM_AVAILABLE:
                memory_tool = create_manage_memory_tool(namespace=("memories",))
                tools.append(memory_tool)
            
            # Define tool implementations
            tool_implementations = {
                "search_documents": lambda params: self.search_tool(params["query"])
            }
            
            # Create the model name based on available providers
            # Create the model name based on available providers
            if self.use_ollama:
                # Use Ollama with the configured model
                ollama_config = OLLAMA_CONFIG
                model_name = f"ollama:{ollama_config.get('models', {}).get('default', 'mistral')}"
            else:
                # Fallback to a local model if Ollama not available
                model_name = "huggingface:mistralai/Mistral-7B-Instruct-v0.2"
            
            # Create the agent
            self.agent = create_react_agent(
                model_name,  # Using Ollama model
                prompt=prompt_fn,
                tools=tools,
                tool_implementations=tool_implementations,
                store=self.store,  # Using in-memory store
                checkpointer=self.checkpointer,
            )
            
            logger.info(f"LangGraph agent initialized with model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error initializing LangGraph components: {str(e)}")
            self.use_langgraph = False
            self.llm = self._initialize_llm()
    def save_state_to_disk(self, conversation_id, state):
        """Save conversation state to disk"""
        file_path = Path(PROCESSED_DIR) / "langgraph_states" / f"{conversation_id}.json"
        file_path.parent.mkdir(exist_ok=True, parents=True)
        with open(file_path, 'w') as f:
            json.dump(state, f)
    
    def load_state_from_disk(self, conversation_id):
        """Load conversation state from disk"""
        file_path = Path(PROCESSED_DIR) / "langgraph_states" / f"{conversation_id}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return None
    def _create_document_retrieval_tool(self):
        """
        Create a document retrieval tool function for LangGraph
        """
        def search_documents(query: str) -> List[Dict[str, Any]]:
            """
            Search for relevant documents.
            
            Args:
                query: Search query
                
            Returns:
                List of relevant documents
            """
            # Execute the search
            search_results = self.search_engine.search(
                query, 
                top_k=3,  # Could be configurable 
                return_all_chunks=True
            )
            
            # Convert the results to a format suitable for the agent
            documents = []
            
            if "results" in search_results:
                for result in search_results["results"]:
                    document = {
                        "content": result["text"],
                        "metadata": result["metadata"],
                        "document_id": result["document_id"] if "document_id" in result else result["metadata"].get("document_id", "unknown"),
                        "chunk_id": result["chunk_id"] if "chunk_id" in result else result["metadata"].get("chunk_id", "unknown"),
                    }
                    documents.append(document)
            
            return documents
        
        return search_documents
    
    def _initialize_llm(self):
        """
        Initialize the language model.
        
        Returns:
            LLM instance or None
        """
        # If Ollama is enabled and available, use it as priority
        if self.use_ollama and OLLAMA_AVAILABLE:
            try:
                ollama_config = OLLAMA_CONFIG
                logger.info(f"Initializing Ollama LLM: {ollama_config['models']['default']}")
                
                return OllamaLLM(
                    base_url=ollama_config.get("base_url", "http://localhost:11434"),
                    model=ollama_config.get("models", {}).get("default", "mistral"),
                    temperature=ollama_config.get("parameters", {}).get("temperature", 0.7),
                    top_p=ollama_config.get("parameters", {}).get("top_p", 0.95),
                    max_tokens=ollama_config.get("parameters", {}).get("max_tokens", 512)
                )
            except Exception as e:
                logger.error(f"Error initializing Ollama: {str(e)}")
        
        # If LangChain is available, try to use HuggingFace
        if self.use_langchain and LANGCHAIN_AVAILABLE:
            try:
                # Import here to avoid dependencies issues
                from langchain_huggingface import HuggingFacePipeline
                import torch
                from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
                
                # Check if CUDA is available
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Initializing LLM {self.model_name} on {device}")
                
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
                logger.error(f"Error initializing LLM: {str(e)}")
        
        logger.warning("Operating in degraded mode: simple response generation without LLM")
        return None
        
    def _build_retrieval_chain(self):
        """
        Build the conversational retrieval chain.
        
        Returns:
            LangChain retrieval chain or None
        """
        if not self.llm or not self.use_langchain:
            return None
        
        try:
            # Create a wrapper for the search engine
            retriever = DocumentSearchRetriever(self.search_engine, k=3)
            
            # Template for question-answering
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
            
            # Adapt to newer LangChain versions
            try:
                # Try the newer approach first (LangChain 0.1.0+)
                from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
                
                chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=retriever,
                    memory=self.memory,
                    return_source_documents=True,
                    combine_docs_chain_kwargs={"prompt": qa_prompt}
                )
            except Exception as e1:
                # Try alternative construction methods
                logger.warning(f"Using alternative chain construction due to: {str(e1)}")
                
                try:
                    # Try direct instantiation
                    from langchain.chains import ConversationalRetrievalChain
                    from langchain.chains.question_answering import load_qa_chain
                    
                    qa_chain = load_qa_chain(
                        llm=self.llm,
                        chain_type="stuff",
                        prompt=qa_prompt
                    )
                    
                    chain = ConversationalRetrievalChain(
                        retriever=retriever,
                        combine_docs_chain=qa_chain,
                        memory=self.memory,
                        return_source_documents=True
                    )
                except Exception as e2:
                    logger.error(f"Failed to create chain with alternative method: {str(e2)}")
                    return None
            
            return chain
        except Exception as e:
            logger.error(f"Error creating retrieval chain: {str(e)}")
            return None
        
    





    @timed(logger=logger)
    @log_exceptions(logger=logger)
    def ask(self, question: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Ask a question and get a response based on documents.
        
        Args:
            question: Question to ask
            conversation_id: Conversation ID (if None, use current conversation)
            
        Returns:
            Generated response with sources
        """
        start_time = time.time()
        
        # Handle conversation ID
        if conversation_id is None:
            if self.current_conversation_id is None:
                # Create a new conversation
                self.current_conversation_id = f"conv_{int(time.time())}"
                self.conversations[self.current_conversation_id] = {
                    "id": self.current_conversation_id,
                    "created_at": datetime.now().isoformat(),
                    "last_update": datetime.now().isoformat(),
                    "messages": []
                }
            conversation_id = self.current_conversation_id
        else:
            # Check if conversation exists
            if conversation_id not in self.conversations:
                raise ValueError(f"Conversation not found: {conversation_id}")
            self.current_conversation_id = conversation_id
        
        # Prepare the response
        response = {
            "conversation_id": conversation_id,
            "question": question,
            "timestamp": datetime.now().isoformat()
        }
        
        # Processing mode: LangGraph, LangChain, or simplified
        if self.use_langgraph:
            try:
                # Use LangGraph agent
                config = {"configurable": {"thread_id": conversation_id}}
                
                agent_response = self.agent.invoke(
                    {
                        "messages": [
                            {"role": "user", "content": question}
                        ]
                    },
                    config=config
                )
                
                # Extract the answer
                answer = agent_response["messages"][-1].content
                
                # Construct a response with sources (if available)
                response["answer"] = answer
                response["sources"] = []  # LangGraph doesn't provide sources in same format, needs custom handling
                
            except Exception as e:
                logger.error(f"Error generating response with LangGraph: {str(e)}")
                # Fallback to simplified mode
                simplified_response = self._simplified_response(question, conversation_id, start_time)
                response.update(simplified_response)
                
        elif self.llm is not None and self.use_langchain:
            try:
                # Build the retrieval chain if needed
                chain = self._build_retrieval_chain()
                
                if chain:
                    # Execute the chain to get a response
                    chain_response = chain({"question": question})
                    
                    # Extract the answer and source documents
                    answer = chain_response["answer"]
                    source_documents = chain_response.get("source_documents", [])
                    
                    # Prepare sources for the response
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
                    # If chain creation fails, use simplified mode
                    simplified_response = self._simplified_response(question, conversation_id, start_time)
                    response.update(simplified_response)
            except Exception as e:
                logger.error(f"Error generating response with LLM: {str(e)}")
                # Fallback to simplified mode
                simplified_response = self._simplified_response(question, conversation_id, start_time)
                response.update(simplified_response)
        else:
            # Simplified mode (without LLM)
            simplified_response = self._simplified_response(question, conversation_id, start_time)
            response.update(simplified_response)
        
        # Update conversation history
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
        
        # Limit history size
        if len(self.conversations[conversation_id]["messages"]) > self.conversation_memory_limit * 2:
            # Keep the last n exchanges (question + answer)
            self.conversations[conversation_id]["messages"] = \
                self.conversations[conversation_id]["messages"][-self.conversation_memory_limit * 2:]
        
        response["processing_time_seconds"] = round(time.time() - start_time, 3)
        return response
    
    def _simplified_response(self, question: str, conversation_id: str, start_time: float) -> Dict[str, Any]:
        """
        Generate a simplified response based only on search (without LLM).
        
        Args:
            question: Question asked
            conversation_id: Conversation ID
            start_time: Processing start time
            
        Returns:
            Simplified response with sources
        """
        # Perform a search
        search_results = self.search_engine.search(question, top_k=3)
        
        # Extract the most relevant chunks
        if search_results.get("total_results", 0) > 0:
            # Retrieve contexts for the best chunks
            contexts = []
            sources = []
            
            for i, result in enumerate(search_results["results"]):
                # For results grouped by document
                if "chunks" in result:
                    for chunk in result["chunks"][:2]:  # Take the 2 best chunks per document
                        # Retrieve context around the chunk
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
                # For individual chunk results
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
            
            # Build a simplified response
            if contexts:
                answer = "Voici les passages les plus pertinents trouvés dans les documents :\n\n"
                for i, ctx in enumerate(contexts[:3], 1):  # Limit to 3 contexts
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
        Create a new conversation.
        
        Returns:
            ID of the new conversation
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
        
        logger.info(f"New conversation created: {conversation_id}")
        return conversation_id
    
    def get_conversation(self, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve a conversation.
        
        Args:
            conversation_id: Conversation ID (if None, returns current conversation)
            
        Returns:
            Conversation details
        """
        if conversation_id is None:
            conversation_id = self.current_conversation_id
            
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation not found: {conversation_id}")
            
        return self.conversations[conversation_id]
    
    def list_conversations(self) -> List[Dict[str, Any]]:
        """
        List all conversations.
        
        Returns:
            List of conversation summaries
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
        Delete a conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            True if deletion succeeded, False otherwise
        """
        if conversation_id not in self.conversations:
            return False
            
        del self.conversations[conversation_id]
        
        if self.current_conversation_id == conversation_id:
            self.current_conversation_id = None
            
        logger.info(f"Conversation deleted: {conversation_id}")
        return True
    
    def save_conversations(self, file_path: Optional[Union[str, Path]] = None) -> str:
        """
        Save all conversations to a file.
        
        Args:
            file_path: Path to save file (if None, uses a default name)
            
        Returns:
            Path to the save file
        """
        if file_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_path = PROCESSED_DIR / "conversations" / f"conversations_{timestamp}.json"
            
        file_path = Path(file_path)
        file_path.parent.mkdir(exist_ok=True, parents=True)
        
        save_json(self.conversations, file_path)
        logger.info(f"Conversations saved to {file_path}")
        
        return str(file_path)
    
    def load_conversations(self, file_path: Union[str, Path]) -> int:
        """
        Load conversations from a file.
        
        Args:
            file_path: Path to conversations file
            
        Returns:
            Number of conversations loaded
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Conversations file not found: {file_path}")
            
        loaded_conversations = load_json(file_path)
        self.conversations.update(loaded_conversations)
        
        logger.info(f"{len(loaded_conversations)} conversations loaded from {file_path}")
        return len(loaded_conversations)
    