"""
Self-RAG Ollama Integration

This file contains the Self-RAG enhanced Ollama endpoints that you can add directly 
to your main.py file instead of importing them from a module.
"""

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import json
import asyncio
from config import (
    OLLAMA_BASE_URL, 
    OLLAMA_DEFAULT_MODEL, 
    OLLAMA_DEFAULT_TEMPERATURE, 
    OLLAMA_DEFAULT_TOP_K,
    logger
)
# Import from your new Self-RAG module
from agent.RAGagent import SelfRAGService, list_ollama_models, pull_ollama_model



# Create Self-RAG service once
selfrag_service = None

def setup_selfrag_endpoints(app: FastAPI, get_orchestrator):
    """
    Add Self-RAG enhanced Ollama endpoints directly to the FastAPI app
    
    Args:
        app: The FastAPI application
        get_orchestrator: Dependency function to get the orchestrator
    """
    
    # Define models within this function to avoid reference issues
    class OllamaModelEnum(str, Enum):
        MISTRAL = "mistral"
        LLAMA3 = "llama3.2"
        LLAMA3_8B = "llama3:8b"
        LLAMA3_70B = "llama3:70b"
        MIXTRAL = "mixtral"
        GEMMA = "gemma"
        GEMMA2 = "gemma2"
        NEURAL_CHAT = "neural-chat"
        PHI3 = "phi3"
    
    class SelfRAGRequest(BaseModel):
        question: str = Field(..., description="Question to ask")
        model_name: Optional[str] = Field("mistral", description="Ollama model to use")
        temperature: Optional[float] = Field(0.0, description="Temperature for generation (0.0-1.0)")
        top_k: Optional[int] = Field(5, description="Number of documents to retrieve")
        conversation_id: Optional[str] = Field(None, description="Conversation ID")
        ollama_host: Optional[str] = Field(None, description="Custom Ollama host (e.g., 'http://localhost:11434')")
    
    class StreamingRequest(BaseModel):
        question: str = Field(..., description="Question to ask")
        model_name: Optional[str] = Field("mistral", description="Ollama model to use")
        temperature: Optional[float] = Field(0.0, description="Temperature for generation (0.0-1.0)")
        top_k: Optional[int] = Field(5, description="Number of documents to retrieve")
        conversation_id: Optional[str] = Field(None, description="Conversation ID")
        ollama_host: Optional[str] = Field(None, description="Custom Ollama host (e.g., 'http://localhost:11434')")
    
    class OllamaHost(BaseModel):
        host: str = Field(..., description="Ollama host URL (e.g., 'http://localhost:11434')")
    
    class OllamaModelPull(BaseModel):
        model_name: str = Field(..., description="Model name to pull (e.g., 'mistral', 'llama3:8b')")
        host: Optional[str] = Field("http://localhost:11434", description="Ollama host URL")
    
    def get_selfrag_service(orchestrator):
        """Get or create the Self-RAG service singleton"""
        global selfrag_service
        if selfrag_service is None:
# Initialize the service
            selfrag_service = SelfRAGService(
                orchestrator=orchestrator,
                ollama_config={"base_url": OLLAMA_BASE_URL},
                default_model=OLLAMA_DEFAULT_MODEL,
                default_temperature=OLLAMA_DEFAULT_TEMPERATURE,
                default_top_k=OLLAMA_DEFAULT_TOP_K
            )
        return selfrag_service
    
    # Add the endpoints to the app
    @app.post("/selfrag/stream", tags=["Self-RAG"])
    async def stream_selfrag(
        query: StreamingRequest,
        orchestrator = Depends(get_orchestrator)
    ):
        """
        Stream a response using Self-RAG (shows step-by-step reasoning)
        
        - **question**: The question to ask
        - **model_name**: Ollama model to use (default: mistral)
        - **temperature**: Temperature for generation, 0-1 (default: 0.0)
        - **top_k**: Number of documents to retrieve (default: 5)
        - **conversation_id**: Optional conversation ID for tracking
        - **ollama_host**: Optional custom Ollama host URL
        """
        service = get_selfrag_service(orchestrator)
        logger.debug(f"Self-RAG streaming request: {query.model_dump()}")

        async def generate_stream():
            try:
                # Build the workflow
                workflow = await service.build_workflow()
                app = workflow.compile()
                
                # Set up initial state
                initial_state = {
                    "question": query.question,
                    "documents": [],
                    "generation": None,
                    "model_name": query.model_name or service.default_model,
                    "temperature": query.temperature or service.default_temperature,
                    "top_k": query.top_k or service.default_top_k
                }
                
                # Temporarily override Ollama host if specified
                original_host = None
                if query.ollama_host:
                    original_host = service.ollama_config.get("host")
                    service.ollama_config["host"] = query.ollama_host
                    logger.info(f"Temporarily overriding Ollama host to: {query.ollama_host}")
                
                try:
                    # Stream each state update as it happens
                    async for output in app.astream(initial_state):
                        for key, value in output.items():
                            # Convert node results to a streamable format
                            node_name = key
                            state = value
                            
                            # Send information about the current node
                            node_info = {
                                "node": node_name,
                                "stage": node_name,  # Can add more descriptive stages if needed
                            }
                            
                            # Add appropriate state information based on node
                            if node_name == "retrieve" and state.get("documents"):
                                node_info["document_count"] = len(state.get("documents", []))
                            
                            if node_name == "grade_documents" and state.get("documents"):
                                node_info["relevant_documents"] = len(state.get("documents", []))
                            
                            if node_name == "generate" and state.get("generation"):
                                node_info["partial_answer"] = state.get("generation")
                                
                            response_json = json.dumps(node_info)
                            yield f"data: {response_json}\n\n"
                    
                    # At the end, send the complete answer
                    if state.get("generation"):
                        final_result = {
                            "done": True,
                            "answer": state.get("generation"),
                            "documents": [
                                {"content": doc.get("content", ""), "metadata": doc.get("metadata", {})} 
                                for doc in state.get("documents", [])
                            ],
                            "model_used": f"ollama/{state.get('model_name')}",
                            "temperature": state.get("temperature"),
                            "top_k": state.get("top_k")
                        }
                        yield f"data: {json.dumps(final_result)}\n\n"
                    else:
                        yield f"data: {json.dumps({'error': 'No answer generated'})}\n\n"
                finally:
                    # Reset original Ollama host if it was overridden
                    if query.ollama_host and original_host:
                        service.ollama_config["host"] = original_host
                        logger.info(f"Restored original Ollama host: {original_host}")
                    
            except Exception as e:
                logger.error(f"Streaming error: {str(e)}")
                error_json = json.dumps({"error": f"Streaming error: {str(e)}"})
                yield f"data: {error_json}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream"
        )
    
    @app.post("/ollama/models", tags=["Ollama"])
    async def get_ollama_models(host_data: OllamaHost):
        """
        List available Ollama models from a specific host
        
        - **host**: Ollama host URL (e.g., 'http://localhost:11434')
        """
        try:
            result = await list_ollama_models(host=host_data.host)
            return result
            
        except Exception as e:
            logger.error(f"Error listing Ollama models: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/ollama/pull", tags=["Ollama"])
    async def pull_model(
        pull_request: OllamaModelPull,
        background_tasks: BackgroundTasks
    ):
        """
        Pull an Ollama model from a specific host
        
        - **model_name**: Model name to pull (e.g., 'mistral', 'llama3:8b')
        - **host**: Ollama host URL (default: 'http://localhost:11434')
        """
        try:
            # Helper function for background pull
            async def pull_in_background():
                try:
                    result = await pull_ollama_model(
                        model_name=pull_request.model_name,
                        host=pull_request.host
                    )
                    logger.info(f"Pull completed: {result}")
                except Exception as e:
                    logger.error(f"Error in background pull: {str(e)}")
            
            # Start pulling in background
            background_tasks.add_task(asyncio.create_task, pull_in_background())
            
            return {
                "status": "success",
                "message": f"Started pulling model {pull_request.model_name} from {pull_request.host} in background"
            }
            
        except Exception as e:
            logger.error(f"Error starting Ollama model pull: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
            
    @app.post("/agentic/ask", tags=["Agentic RAG"])
    async def ask_agentic(
        query: SelfRAGRequest,
        orchestrator = Depends(get_orchestrator)
    ):
        """
        Backward compatibility endpoint for agentic RAG workflow
        Now redirects to Self-RAG implementation
        
        - **question**: The question to ask
        - **model_name**: Ollama model to use (default: mistral)
        - **temperature**: Temperature for generation, 0-1 (default: 0.0)
        - **top_k**: Number of documents to retrieve (default: 5)
        - **conversation_id**: Optional conversation ID for tracking
        - **ollama_host**: Optional custom Ollama host URL
        """
        service = get_selfrag_service(orchestrator)
        logger.debug(f"Agentic ask (redirected to Self-RAG): {query.model_dump()}")
        
        try:
            # Call the Self-RAG implementation
            result = await service.ask_with_selfrag(
                question=query.question,
                model_name=query.model_name,
                temperature=query.temperature,
                top_k=query.top_k,
                conversation_id=query.conversation_id,
                ollama_host=query.ollama_host
            )
            
            # Debug the result before returning
            logger.debug(f"Raw result from selfrag_service.ask_with_selfrag: {result}")
            
            # Make sure we're not returning an error or empty answer
            if result.get("status") == "success" and result.get("answer"):
                logger.info(f"Answer generated successfully. Length: {len(result['answer'])}")
            else:
                logger.warning(f"No answer or error in result: {result}")
            
            return result
                
        except Exception as e:
            logger.error(f"Error using Self-RAG: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
            
    logger.info("Self-RAG endpoints set up successfully")
    
    return {
        "stream_selfrag": stream_selfrag,
        "get_ollama_models": get_ollama_models,
        "pull_model": pull_model,
        "ask_agentic": ask_agentic  # For backward compatibility
    }