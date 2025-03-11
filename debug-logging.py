import logging
import sys
import os
import inspect
from fastapi import FastAPI
from fastapi.routing import APIRoute
from pydantic import BaseModel
import importlib

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('api_debug.log')
    ]
)

logger = logging.getLogger("api_debug")

def debug_models():
    """Debug Pydantic models and FastAPI routes"""
    logger.debug("=" * 50)
    logger.debug("DEBUGGING PYDANTIC MODELS AND ROUTES")
    logger.debug("=" * 50)
    
    # Log Python path
    logger.debug("Python sys.path:")
    for p in sys.path:
        logger.debug(f"  - {p}")
    
    # Log current directory
    logger.debug(f"Current directory: {os.getcwd()}")
    
    # Log module information
    try:
        logger.debug("Attempting to import ollama_endpoints module")
        try:
            ollama_module = importlib.import_module("ollama_endpoints")
            logger.debug(f"Successfully imported ollama_endpoints from {ollama_module.__file__}")
            
            # Log classes in the module
            logger.debug("Classes in ollama_endpoints module:")
            for name, obj in inspect.getmembers(ollama_module):
                if inspect.isclass(obj):
                    logger.debug(f"  - {name}: {obj.__module__}.{obj.__name__}")
                    
                    # If it's a Pydantic model, log more details
                    if isinstance(obj, type) and issubclass(obj, BaseModel):
                        logger.debug(f"    Pydantic model fields: {obj.__fields__}")
                        
        except ImportError as e:
            logger.error(f"Error importing ollama_endpoints: {e}")
            
        # Log FastAPI app routes
        logger.debug("=" * 50)
        logger.debug("FASTAPI ROUTES")
        logger.debug("=" * 50)
        
        # This function should be called after the app is defined
        # and routes are registered
        from main import app  # Import your FastAPI app
        
        logger.debug(f"FastAPI app routes ({len(app.routes)}):")
        for route in app.routes:
            if isinstance(route, APIRoute):
                logger.debug(f"Route: {route.path} [{route.methods}]")
                logger.debug(f"  - Endpoint: {route.endpoint.__name__}")
                logger.debug(f"  - Response model: {route.response_model}")
                
                # Log the parameter details
                sig = inspect.signature(route.endpoint)
                logger.debug(f"  - Parameters:")
                for param_name, param in sig.parameters.items():
                    if param_name not in ['request', 'background_tasks']:
                        logger.debug(f"    - {param_name}: {param.annotation}")
                
                # Check for Pydantic models in parameters
                for param_name, param in sig.parameters.items():
                    if hasattr(param.annotation, '__origin__') and param.annotation.__origin__ is inspect._empty:
                        continue
                    
                    if isinstance(param.annotation, type) and issubclass(param.annotation, BaseModel):
                        logger.debug(f"    Pydantic model in parameter: {param_name}")
                        logger.debug(f"    Model details: {param.annotation.__module__}.{param.annotation.__name__}")
                
    except Exception as e:
        logger.error(f"Error in debug_models: {e}")
        import traceback
        logger.error(traceback.format_exc())

# Place this in main.py after registering all routes
# debug_models()