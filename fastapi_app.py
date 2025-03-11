import json
from fastapi import BackgroundTasks, FastAPI, File, UploadFile, Form, Query, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import os
import uvicorn
import asyncio

# Import the RAG Orchestrator
from agent.ollama_endpoints import OllamaAskQuery, get_ollama_service
from orchastrator import RAGOrchestrator
from agent.RAGagent import AgenticRAGService, ModelType

# Import configuration
from config import logger

# Define API models
class SearchQuery(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(5, description="Number of results to return")
    filter_dict: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    search_type: str = Field("hybrid", description="Search type (hybrid or semantic)")
    rerank_results: bool = Field(True, description="Whether to rerank results")

class QuestionQuery(BaseModel):
    question: str = Field(..., description="Question to ask")
    top_k: int = Field(5, description="Number of context passages to retrieve")
    filter_dict: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    search_type: str = Field("hybrid", description="Search type (hybrid or semantic)")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")

class DirectoryProcessQuery(BaseModel):
    directory: Optional[str] = Field(None, description="Subdirectory to process")
    force_reprocess: bool = Field(False, description="Whether to force reprocessing")

class IndexingQuery(BaseModel):
    force_reindex: bool = Field(False, description="Whether to force reindexing")

# Create the orchestrator (singleton)
orchestrator = RAGOrchestrator()

# Create FastAPI app
app = FastAPI(
    title="RAG API",
    description="API for RAG (Retrieval Augmented Generation) system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get the orchestrator
def get_orchestrator():
    return orchestrator

#ollama setup
@app.post("/ollama/ask", tags=["Ollama"])
async def ask_ollama(
        query: OllamaAskQuery,
        orchestrator = Depends(get_orchestrator)
    ):
        """
        Ask a question using Ollama models
        """
        service = get_ollama_service(orchestrator)

        try:
            # If streaming is requested, return a streaming response
            if query.stream:
                async def generate_stream():
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
                        agentic=query.agentic
                    )
                    
                    if result.get("status") == "error":
                        error_json = json.dumps({"error": result.get("error")})
                        yield f"data: {error_json}\n\n"
                        return
                    
                    # Stream the answer chunks
                    answer_stream = result.get("answer_stream")
                    if answer_stream:
                        try:
                            async for chunk in answer_stream:
                                content = chunk["message"]["content"]
                                # Format as Server-Sent Events with proper JSON escaping
                                response_json = json.dumps({"content": content})
                                yield f"data: {response_json}\n\n"
                        except Exception as e:
                            error_json = json.dumps({"error": f"Streaming error: {str(e)}"})
                            yield f"data: {error_json}\n\n"
                            
                    # Send the sources at the end
                    sources = result.get("sources", [])
                    sources_json = json.dumps({"sources": sources})
                    yield f"data: {sources_json}\n\n"
                    
                    # End of stream
                    yield f"data: {json.dumps({'done': True})}\n\n"
                
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
                    agentic=query.agentic
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
            
            async def pull_model():
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
        query: OllamaAskQuery,
        orchestrator = Depends(get_orchestrator)
    ):
        """
        Ask a question using the agentic RAG workflow
        """
        service = get_ollama_service(orchestrator)

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
    
   
# Define API routes
@app.post("/documents/process", tags=["Documents"])
async def process_document(
    file: UploadFile = File(...), 
    force_reprocess: bool = Form(False),
    orchestrator: RAGOrchestrator = Depends(get_orchestrator)
):
    """
    Process a single document from an uploaded file
    """
    logger.info(f"API: Processing uploaded document: {file.filename}")
    
    try:
        # Save uploaded file
        uploads_dir = Path(orchestrator.data_dir) / "uploads"
        os.makedirs(uploads_dir, exist_ok=True)
        
        temp_file_path = uploads_dir / file.filename
        
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        logger.info(f"File saved to {temp_file_path}")
        
        # Process the document
        result = await orchestrator.process_document(temp_file_path, force_reprocess)
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error"))
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing uploaded document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/process-directory", tags=["Documents"])
async def process_directory(
    query: DirectoryProcessQuery,
    orchestrator: RAGOrchestrator = Depends(get_orchestrator)
):
    """
    Process all documents in a directory
    """
    result = await orchestrator.process_directory(query.directory, query.force_reprocess)
    
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error"))
    
    return result

@app.post("/documents/index", tags=["Documents"])
async def index_documents(
    query: IndexingQuery,
    orchestrator: RAGOrchestrator = Depends(get_orchestrator)
):
    """
    Index all processed documents
    """
    result = await orchestrator.index_documents(query.force_reindex)
    
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error"))
    
    return result

@app.post("/search", tags=["Search"])
async def search(
    query: SearchQuery,
    orchestrator: RAGOrchestrator = Depends(get_orchestrator)
):
    """
    Search for documents
    """
    result = await orchestrator.search(
        query=query.query,
        top_k=query.top_k,
        filter_dict=query.filter_dict,
        search_type=query.search_type,
        rerank_results=query.rerank_results
    )
    
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error"))
    
    return result

@app.post("/ask", tags=["QA"])
async def ask(
    query: QuestionQuery,
    orchestrator: RAGOrchestrator = Depends(get_orchestrator)
):
    """
    Ask a question using the RAG system
    """
    result = await orchestrator.ask(
        question=query.question,
        top_k=query.top_k,
        filter_dict=query.filter_dict,
        search_type=query.search_type,
        conversation_id=query.conversation_id
    )
    
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error"))
    
    return result

@app.get("/status", tags=["System"])
async def get_system_status(
    orchestrator: RAGOrchestrator = Depends(get_orchestrator)
):
    """
    Get the current system status
    """
    result = await orchestrator.get_system_status()
    
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error"))
    
    return result

@app.get("/documents/{document_id}", tags=["Documents"])
async def get_document_info(
    document_id: str,
    orchestrator: RAGOrchestrator = Depends(get_orchestrator)
):
    """
    Get information about a specific document
    """
    result = await orchestrator.get_document_info(document_id)
    
    if result.get("status") == "error":
        if "not found" in result.get("error", ""):
            raise HTTPException(status_code=404, detail=result.get("error"))
        else:
            raise HTTPException(status_code=500, detail=result.get("error"))
    
    return result

@app.get("/documents", tags=["Documents"])
async def list_documents(
    limit: int = Query(100, ge=1, le=1000), 
    offset: int = Query(0, ge=0),
    orchestrator: RAGOrchestrator = Depends(get_orchestrator)
):
    """
    List all documents in the system
    """
    result = await orchestrator.list_documents(limit, offset)
    
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error"))
    
    return result

@app.get("/health", tags=["System"])
async def health_check():
    """
    Simple health check endpoint
    """
    return {"status": "ok", "message": "RAG API is running"}

@app.get("/", tags=["System"])
async def root():
    """
    API documentation redirect
    """
    return {"message": "Welcome to RAG API. Visit /docs for API documentation."}

# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": str(exc), "error_type": type(exc).__name__},
    )

# Run the application
if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting RAG API server")
    uvicorn.run(app, host="0.0.0.0", port=8000)