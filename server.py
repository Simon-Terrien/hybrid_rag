#!/usr/bin/env python
"""
Standalone server script for the RAG API.
This avoids asyncio issues by running the server directly.
"""

import argparse
import uvicorn
from config import logger

def main():
    parser = argparse.ArgumentParser(description="RAG API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    logger.info(f"Starting RAG API server on {args.host}:{args.port}")
    
    # Run the server
    uvicorn.run(
        "fastapi_app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()