"""
Direct Ollama Integration

This file contains the Ollama endpoints that you can add directly 
to your main.py file instead of importing them from a module.
"""

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import json
import logging
import asyncio

# Import from your existing modules
from agent.RAGagent import AgenticRAGService, ModelType

# Configure logging
logger = logging.getLogger(__name__)

# Create Ollama RAG service once
ollama_service = None

def setup_ollama_endpoints(app: FastAPI, get_orchestrator):
    """
    Add Ollama endpoints directly to the FastAPI app
    
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
    
    class OllamaQuestion(BaseModel):
        question: str = Field(..., description="Question to ask")
        top_k: int = Field(5, description="Number of context passages to retrieve")
        filter_dict: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
        search_type: str = Field("hybrid", description="Search type (hybrid or semantic)")
        conversation_id: Optional[str] = Field(None, description="Conversation ID")
        model_name: str = Field("mistral", description="Ollama model to use")
        system_prompt: Optional[str] = Field(None, description="System prompt for the model")
        messages: Optional[List[Dict[str, str]]] = Field(None, description="Chat history in message format")
        stream: bool = Field(False, description="Whether to stream the response")
    
    class AgenticQuestion(BaseModel):
        question: str = Field(..., description="Question to ask")
        top_k: int = Field(5, description="Number of context passages to retrieve")
        filter_dict: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
        search_type: str = Field("hybrid", description="Search type (hybrid or semantic)")
        conversation_id: Optional[str] = Field(None, description="Conversation ID")
    
    def get_ollama_service(orchestrator):
        """Get or create the Ollama service singleton"""
        global ollama_service
        if ollama_service is None:
            ollama_service = AgenticRAGService(
                orchestrator=orchestrator,
                ollama_config={"host": "http://localhost:11434"}
            )
        return ollama_service
    
    # Add the endpoints to the app
    @app.post("/ollama/ask", tags=["Ollama"])
    async def ask_ollama(
        query: OllamaQuestion,
        orchestrator = Depends(get_orchestrator)
    ):
        """
        Ask a question using Ollama models
        """
        service = get_ollama_service(orchestrator)
        logger.debug(f"Ollama ask: {query.dict()}")

        try:
            # If streaming is requested, return a streaming response
            if query.stream:
                async def generate_stream():
                    try:
                        result = await service.ask(
                            question=query.question,
                            top_k=query.top_k,
                            filter_dict=query.filter_dict,
                            search_type=query.search_type,
                            conversation_id=query.conversation_id,
                            model_type=ModelType.OLLAMA,
                            model_name=query.model_name,
                            system_prompt=query.system_prompt,
                            messages=query.messages,
                            stream=True,
                            agentic=False
                        )
                        
                        if result.get("status") == "error":
                            error_json = json.dumps({"error": result.get("error")})
                            yield f"data: {error_json}\n\n"
                            return
                        
                        # Stream the answer chunks
                        answer_stream = result.get("answer_stream")
                        if answer_stream:
                            async for chunk in answer_stream:
                                content = chunk["message"]["content"]
                                # Format as Server-Sent Events with proper JSON escaping
                                response_json = json.dumps({"content": content})
                                yield f"data: {response_json}\n\n"
                                
                        # Send the sources at the end
                        sources = result.get("sources", [])
                        sources_json = json.dumps({"sources": sources})
                        yield f"data: {sources_json}\n\n"
                        
                        # End of stream
                        yield f"data: {json.dumps({'done': True})}\n\n"
                    except Exception as e:
                        logger.error(f"Streaming error: {str(e)}")
                        error_json = json.dumps({"error": f"Streaming error: {str(e)}"})
                        yield f"data: {error_json}\n\n"
                
                return StreamingResponse(
                    generate_stream(),
                    media_type="text/event-stream"
                )
            else:
                # Regular non-streaming response
                result = await service.ask(
                    question=query.question,
                    top_k=query.top_k,
                    filter_dict=query.filter_dict,
                    search_type=query.search_type,
                    conversation_id=query.conversation_id,
                    model_type=ModelType.OLLAMA,
                    model_name=query.model_name,
                    system_prompt=query.system_prompt,
                    messages=query.messages,
                    stream=False,
                    agentic=False
                )
                
                if result.get("status") == "error":
                    raise HTTPException(status_code=500, detail=result.get("error"))
                
                return result
                
        except Exception as e:
            logger.error(f"Error using Ollama: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/ollama/models", tags=["Ollama"])
    async def list_ollama_models():
        """
        List available Ollama models
        """
        try:
            # Import here to make it optional
            from ollama import Client
            
            client = Client(host="http://localhost:11434")
            models = client.list()
            
            return {
                "status": "success",
                "models": models["models"]
            }
        except ImportError:
            logger.error("Ollama library not installed")
            return {
                "status": "error",
                "error": "Ollama Python library not installed. Install with 'pip install ollama'"
            }
        except Exception as e:
            logger.error(f"Error listing Ollama models: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    @app.post("/ollama/pull", tags=["Ollama"])
    async def pull_ollama_model(
        model_name: str,
        background_tasks: BackgroundTasks
    ):
        """
        Pull an Ollama model in the background
        """
        try:
            # Import here to make it optional
            from ollama import Client
            
            def pull_model():
                try:
                    client = Client(host="http://localhost:11434")
                    client.pull(model_name)
                    logger.info(f"Successfully pulled Ollama model: {model_name}")
                except Exception as e:
                    logger.error(f"Error pulling Ollama model {model_name}: {str(e)}")
            
            # Start pulling in background
            background_tasks.add_task(pull_model)
            
            return {
                "status": "success",
                "message": f"Started pulling model {model_name} in background"
            }
        except ImportError:
            logger.error("Ollama library not installed")
            return {
                "status": "error",
                "error": "Ollama Python library not installed. Install with 'pip install ollama'"
            }
        except Exception as e:
            logger.error(f"Error starting Ollama model pull: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
            
    @app.post("/agentic/ask", tags=["Agentic RAG"])
    async def ask_agentic(
        query: AgenticQuestion,
        orchestrator = Depends(get_orchestrator)
    ):
        """
        Ask a question using the agentic RAG workflow
        """
        service = get_ollama_service(orchestrator)
        logger.debug(f"Agentic ask: {query.dict()}")
        
        try:
            result = await service.ask(
                question=query.question,
                top_k=query.top_k,
                filter_dict=query.filter_dict,
                search_type=query.search_type,
                conversation_id=query.conversation_id,
                agentic=True
            )
            
            if result.get("status") == "error":
                raise HTTPException(status_code=500, detail=result.get("error"))
            
            return result
                
        except Exception as e:
            logger.error(f"Error using agentic RAG: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
            
    logger.info("Ollama endpoints set up successfully")
    
    return {
        "ask_ollama": ask_ollama,
        "list_ollama_models": list_ollama_models,
        "pull_ollama_model": pull_ollama_model,
        "ask_agentic": ask_agentic
    }