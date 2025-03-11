"""
Ollama Endpoints Module

This module provides FastAPI endpoint definitions for the Ollama integration.
"""

from fastapi import Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import json
from config import logger
# Import from your existing modules
from agent.RAGagent import AgenticRAGService, ModelType



class OllamaModelType(str, Enum):
    MISTRAL = "mistral"
    LLAMA3 = "llama3.2"
    LLAMA3_8B = "llama3:8b"
    LLAMA3_70B = "llama3:70b"
    MIXTRAL = "mixtral"
    GEMMA = "gemma"
    GEMMA2 = "gemma2"
    NEURAL_CHAT = "neural-chat"
    PHI3 = "phi3"
    
    # Allow custom model names as fallback
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if v in [member.value for member in cls]:
            return v
        # If the value isn't in our predefined list, just return it as is
        return v

class OllamaAskQuery(BaseModel):
    question: str = Field(..., description="Question to ask")
    top_k: int = Field(5, description="Number of context passages to retrieve")
    filter_dict: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    search_type: str = Field("hybrid", description="Search type (hybrid or semantic)")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    model_name: OllamaModelType = Field(OllamaModelType.MISTRAL, description="Ollama model to use")
    system_prompt: Optional[str] = Field(None, description="System prompt for the model")
    messages: Optional[List[Dict[str, str]]] = Field(None, description="Chat history in message format")
    stream: bool = Field(False, description="Whether to stream the response")
    agentic: bool = Field(False, description="Whether to use agentic RAG workflow")

# Create Ollama RAG service once
ollama_service = None

def get_ollama_service(orchestrator):
    """Get or create the Ollama service singleton"""
    global ollama_service
    if ollama_service is None:
        ollama_service = AgenticRAGService(
            orchestrator=orchestrator,
            ollama_config={"host": "http://localhost:11434"}
        )
    return ollama_service

def register_ollama_endpoints(app, get_orchestrator=None):
    """
    Register Ollama endpoints to the FastAPI app
    
    Args:
        app: The FastAPI application
        get_orchestrator: Dependency function to get the orchestrator
    """
    if get_orchestrator is None:
        logger.warning("No orchestrator dependency provided, using a placeholder")
        # Define a placeholder if none is provided
        def get_orchestrator():
            from orchastrator import RAGOrchestrator
            return RAGOrchestrator()
    
