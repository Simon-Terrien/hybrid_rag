import traceback
import logging
from typing import Dict, Any, Optional, List, Union
from enum import Enum
import asyncio
import time

logger = logging.getLogger(__name__)

class ModelType(Enum):
    DEFAULT = "default"
    OLLAMA = "ollama"

class AgenticRAGService:
    def __init__(self, orchestrator, ollama_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the QA service with support for Ollama models and agentic RAG
        
        Args:
            orchestrator: The orchestrator for retrieval and default LLM processing
            ollama_config: Optional configuration for Ollama (host, headers, etc.)
        """
        self.orchestrator = orchestrator
        self.ollama_config = ollama_config or {}
        
    async def _query_ollama(
        self, 
        model_name: str, 
        messages: List[Dict[str, str]],
        stream: bool = False,
        system: Optional[str] = None,
        context: Optional[List[str]] = None
    ) -> Union[Dict[str, Any], Any]:
        """
        Query Ollama API using the official Python client
        
        Args:
            model_name: Name of the Ollama model (e.g., "mistral", "llama3.2")
            messages: List of message objects with role and content
            stream: Whether to stream the response
            system: Optional system prompt
            context: Optional context passages to include
            
        Returns:
            Response from Ollama API
        """
        try:
            # Import here to make it optional
            from ollama import AsyncClient
            
            # Format context if provided
            if context:
                # Prepare context by adding it to the user's message
                context_text = "\n\n".join([f"Passage {i+1}:\n{passage}" for i, passage in enumerate(context)])
                
                # Add context to the last user message
                for i in reversed(range(len(messages))):
                    if messages[i]["role"] == "user":
                        messages[i]["content"] = f"Context information:\n{context_text}\n\nQuestion: {messages[i]['content']}"
                        break
            
            # Create client with optional config
            client = AsyncClient(**self.ollama_config)
            
            # Prepare parameters
            params = {
                "model": model_name,
                "messages": messages,
                "stream": stream
            }
            
            if system:
                params["system"] = system
                
            # Execute the request
            if stream:
                return await client.chat(**params)
            else:
                return await client.chat(**params)
                
        except ImportError:
            logger.error("Ollama Python library not installed. Install with 'pip install ollama'")
            raise Exception("Ollama Python library not installed. Install with 'pip install ollama'")
        except Exception as e:
            logger.error(f"Error querying Ollama: {str(e)}")
            raise Exception(f"Failed to query Ollama API: {str(e)}")
    
    async def ask(
        self,
        question: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        search_type: str = "hybrid",
        conversation_id: Optional[str] = None,
        model_type: ModelType = ModelType.DEFAULT,
        model_name: str = "mistral",
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
        agentic: bool = False
    ) -> Dict[str, Any]:
        """
        Ask a question using RAG with optional Ollama integration and agentic capabilities
        
        Args:
            question: Question to ask
            top_k: Number of context passages to retrieve
            filter_dict: Metadata filters
            search_type: Search type ("hybrid", "semantic")
            conversation_id: Optional conversation ID for context
            model_type: Model type to use (default or ollama)
            model_name: Model name for Ollama (e.g., "mistral", "llama3.2")
            system_prompt: Optional system prompt
            messages: Optional conversation history (for chat mode)
            stream: Whether to stream the response
            agentic: Whether to use agentic RAG workflow
            
        Returns:
            Answer and source passages
        """
        start_time = time.time()
        logger.info(f"API: Answering question '{question}' using {model_type.value} model")
        
        try:
            # For agentic RAG, use the orchestrator's agentic workflow
            if agentic and hasattr(self.orchestrator, "run_agentic_workflow"):
                result = await self.orchestrator.run_agentic_workflow(
                    question=question,
                    conversation_id=conversation_id
                )
                
                processing_time = time.time() - start_time
                
                return {
                    "status": "success",
                    "question": question,
                    "answer": result.get("answer", ""),
                    "sources": result.get("sources", []),
                    "processing_time": processing_time,
                    "conversation_id": conversation_id,
                    "model_used": "agentic_workflow"
                }
            
            # Standard RAG: Always retrieve context using the orchestrator
            retrieval_result = await self.orchestrator.retrieve(
                query=question,
                top_k=top_k,
                filter_dict=filter_dict,
                search_type=search_type
            )
            
            sources = retrieval_result.get("sources", [])
            context_passages = [source.get("content", "") for source in sources]
            
            # Prepare messages if not provided
            if not messages:
                messages = [{"role": "user", "content": question}]
            
            if model_type == ModelType.OLLAMA:
                # Use Ollama for generation
                if stream:
                    # For streaming, return an async generator
                    response_stream = await self._query_ollama(
                        model_name=model_name,
                        messages=messages,
                        stream=True,
                        system=system_prompt,
                        context=context_passages
                    )
                    
                    processing_time = time.time() - start_time
                    
                    return {
                        "status": "success",
                        "question": question,
                        "answer_stream": response_stream,  # This is an async generator
                        "sources": sources,
                        "processing_time": processing_time,
                        "conversation_id": conversation_id,
                        "model_used": f"ollama/{model_name}"
                    }
                else:
                    # Regular non-streaming response
                    response = await self._query_ollama(
                        model_name=model_name,
                        messages=messages,
                        stream=False,
                        system=system_prompt,
                        context=context_passages
                    )
                    
                    answer = response.message.content if hasattr(response, 'message') else response["message"]["content"]
                    
                    processing_time = time.time() - start_time
                    
                    return {
                        "status": "success",
                        "question": question,
                        "answer": answer,
                        "sources": sources,
                        "processing_time": processing_time,
                        "conversation_id": conversation_id,
                        "model_used": f"ollama/{model_name}"
                    }
            else:
                # Use default model from orchestrator
                result = await self.orchestrator.ask(
                    question=question,
                    top_k=top_k,
                    filter_dict=filter_dict,
                    search_type=search_type,
                    conversation_id=conversation_id
                )
                
                processing_time = time.time() - start_time
                
                return {
                    "status": "success",
                    "question": question,
                    "answer": result.get("answer", ""),
                    "sources": sources,
                    "processing_time": processing_time,
                    "conversation_id": conversation_id,
                    "model_used": "default"
                }
            
        except Exception as e:
            logger.error(f"Error answering question '{question}': {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error("Stack trace:")
            traceback.print_exc()
            
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
            