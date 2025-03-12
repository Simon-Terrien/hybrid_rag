import argparse
import asyncio
import os
import sys
from pathlib import Path

# Import configuration
from config import logger, DATA_DIR

# Import the orchestrators
from orchastrator import RAGOrchestrator
from utils import load_spacy_model

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="RAG System Command Line Interface")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process documents")
    process_parser.add_argument("--file", "-f", help="Path to a specific file to process")
    process_parser.add_argument("--dir", "-d", help="Path to a directory to process")
    process_parser.add_argument("--force", action="store_true", help="Force reprocessing of already processed documents")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index documents")
    index_parser.add_argument("--force", action="store_true", help="Force reindexing of already indexed documents")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results to return")
    search_parser.add_argument("--type", "-t", choices=["hybrid", "semantic"], default="hybrid", help="Search type")
    search_parser.add_argument("--no-rerank", action="store_true", help="Disable reranking")
    
    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of context passages to retrieve")
    ask_parser.add_argument("--type", "-t", choices=["hybrid", "semantic"], default="hybrid", help="Search type")
    
    # System command
    system_parser = subparsers.add_parser("system", help="System operations")
    system_parser.add_argument("--status", action="store_true", help="Show system status")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start the API server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    server_parser.add_argument("--port", "-p", type=int, default=8000, help="Port to bind the server to")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List documents")
    list_parser.add_argument("--limit", "-l", type=int, default=10, help="Maximum number of documents to list")
    list_parser.add_argument("--offset", "-o", type=int, default=0, help="Offset for pagination")
    
    # Document info command
    docinfo_parser = subparsers.add_parser("docinfo", help="Get document info")
    docinfo_parser.add_argument("document_id", help="Document ID to get info for")
    
    return parser.parse_args()

async def handle_process(args):
    """Handle the process command"""
    orchestrator = RAGOrchestrator()
    load_spacy_model("fr_core_news_lg")
    load_spacy_model("en_core_web_lg")
    if args.file:
        # Process a specific file
        file_path = Path(args.file)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return
        
        logger.info(f"Processing file: {file_path}")
        result = await orchestrator.process_document(file_path, args.force)
        
        if result.get("status") == "success":
            logger.info(f"File processed successfully: {result.get('document_id')}")
        else:
            logger.error(f"Error processing file: {result.get('error')}")
    
    elif args.dir:
        # Process a specific directory
        dir_path = args.dir
        logger.info(f"Processing directory: {dir_path}")
        result = await orchestrator.process_directory(dir_path, args.force)
        
        if result.get("status") == "success":
            logger.info(f"Directory processed successfully: {result.get('processed_count')} documents processed, {result.get('failed_count')} failed")
        else:
            logger.error(f"Error processing directory: {result.get('error')}")
    
    else:
        # Process the default data directory
        logger.info(f"Processing default data directory")
        result = await orchestrator.process_directory(None, args.force)
        
        if result.get("status") == "success":
            logger.info(f"Data directory processed successfully: {result.get('processed_count')} documents processed, {result.get('failed_count')} failed")
        else:
            logger.error(f"Error processing data directory: {result.get('error')}")

async def handle_index(args):
    """Handle the index command"""
    orchestrator = RAGOrchestrator()
    
    logger.info(f"Indexing documents...")
    result = await orchestrator.index_documents(args.force)
    
    if result.get("status") == "success":
        logger.info(f"Documents indexed successfully: {result.get('indexed_count')} indexed, {result.get('skipped_count')} skipped, {result.get('failed_count')} failed")
    else:
        logger.error(f"Error indexing documents: {result.get('error')}")

async def handle_search(args):
    """Handle the search command"""
    orchestrator = RAGOrchestrator()
    
    logger.info(f"Searching for: {args.query}")
    result = await orchestrator.search(
        query=args.query,
        top_k=args.top_k,
        search_type=args.type,
        rerank_results=not args.no_rerank
    )
    
    if result.get("status") == "success":
        print(f"\nSearch results for: {args.query}")
        print(f"Found {result.get('result_count')} results in {result.get('processing_time', 0):.2f} seconds\n")
        
        for i, doc in enumerate(result.get("results", [])):
            print(f"{i+1}. Score: {doc.get('similarity', 0):.4f}")
            
            # If result has document metadata, show it
            if "metadata" in doc:
                print(f"   Document: {doc.get('metadata', {}).get('document_id', 'Unknown')}")
            
            text = doc.get("text", "")
            # Truncate text if too long
            if len(text) > 500:
                text = text[:500] + "..."
            print(f"   {text}\n")
    else:
        logger.error(f"Error searching: {result.get('error')}")

async def handle_ask(args):
    """Handle the ask command"""
    orchestrator = RAGOrchestrator()
    
    logger.info(f"Asking: {args.question}")
    result = await orchestrator.ask(
        question=args.question,
        top_k=args.top_k,
        search_type=args.type
    )
    
    if result.get("status") == "success":
        print(f"\nQuestion: {args.question}\n")
        print(f"Answer: {result.get('answer')}\n")
        
        if result.get("sources"):
            print("Sources:")
            for i, source in enumerate(result.get("sources", [])):
                print(f"{i+1}. {source.get('metadata', {}).get('document_id', 'Unknown')}")
                
                text = source.get("text", "")
                # Truncate text if too long
                if len(text) > 200:
                    text = text[:200] + "..."
                print(f"   {text}\n")
    else:
        logger.error(f"Error asking question: {result.get('error')}")

async def handle_system(args):
    """Handle the system command"""
    orchestrator = RAGOrchestrator()
    
    if args.status:
        logger.info("Getting system status")
        result = await orchestrator.get_system_status()
        
        if result.get("status") == "success":
            print("\nSystem Status:")
            print(f"Status: {result.get('system_status', 'unknown')}")
            
            components = result.get("components", {})
            print("\nComponents:")
            for component, info in components.items():
                print(f"  - {component}: {info.get('status', 'unknown')}")
            
            stats = result.get("statistics", {})
            print("\nStatistics:")
            print(f"  - Total documents: {stats.get('total_documents', 0)}")
            print(f"  - Total chunks: {stats.get('total_chunks', 0)}")
            print(f"  - Total queries: {stats.get('total_queries', 0)}")
            
            vector_stats = stats.get("vector_db", {})
            if vector_stats:
                print("\nVector Database:")
                print(f"  - Type: {vector_stats.get('vector_db_type', 'unknown')}")
                print(f"  - Model: {vector_stats.get('embedding_model', 'unknown')}")
                print(f"  - Total documents: {vector_stats.get('total_documents', 0)}")
                print(f"  - Total chunks: {vector_stats.get('total_chunks', 0)}")
        else:
            logger.error(f"Error getting system status: {result.get('error')}")
def handle_server(args):
    """Handle the server command"""
    import uvicorn
    from fastapi_app import app
    load_spacy_model("fr_core_news_lg")
    load_spacy_model("en_core_web_lg")
    logger.info(f"Starting API server on {args.host}:{args.port}")
    
    # Run the server directly without asyncio.run()
    uvicorn.run(
        "fastapi_app:app",  # Use module:app format instead of app object
        host=args.host,
        port=args.port,
        reload=False
    )


async def handle_list(args):
    """Handle the list command"""
    orchestrator = RAGOrchestrator()
    
    logger.info(f"Listing documents (limit={args.limit}, offset={args.offset})")
    result = await orchestrator.list_documents(args.limit, args.offset)
    
    if result.get("status") == "success":
        print("\nDocuments:")
        
        pagination = result.get("pagination", {})
        print(f"Showing {len(result.get('documents', []))} of {pagination.get('total', 0)} total documents")
        
        for i, doc in enumerate(result.get("documents", [])):
            print(f"{i+1}. ID: {doc.get('document_id', 'Unknown')}")
            print(f"   Path: {doc.get('file_path', 'Unknown')}")
            
            # Show some metadata if available
            metadata = doc.get("metadata", {})
            if metadata:
                if "title" in metadata:
                    print(f"   Title: {metadata.get('title')}")
                if "source" in metadata:
                    print(f"   Source: {metadata.get('source')}")
            
            print()
        
        # Show pagination info
        if pagination.get("has_more", False):
            next_offset = args.offset + args.limit
            print(f"Use --offset {next_offset} to see the next page")
    else:
        logger.error(f"Error listing documents: {result.get('error')}")

async def handle_docinfo(args):
    """Handle the document info command"""
    orchestrator = RAGOrchestrator()
    
    logger.info(f"Getting info for document: {args.document_id}")
    result = await orchestrator.get_document_info(args.document_id)
    
    if result.get("status") == "success":
        document = result.get("document", {})
        
        print("\nDocument Information:")
        print(f"ID: {document.get('document_id', 'Unknown')}")
        print(f"Path: {document.get('file_path', 'Unknown')}")
        print(f"Hash: {document.get('hash', 'Unknown')}")
        
        chunks = result.get("chunks", {})
        print(f"Chunks: {chunks.get('count', 0)}")
        print(f"Chunk path: {chunks.get('path', 'Unknown')}")
        
        metadata = document.get("metadata", {})
        if metadata:
            print("\nMetadata:")
            for key, value in metadata.items():
                print(f"  - {key}: {value}")
    else:
        logger.error(f"Error getting document info: {result.get('error')}")


async def main_async():
    """Main async function"""
    args = parse_args()
    
    if args.command == "process":
        await handle_process(args)
    elif args.command == "index":
        await handle_index(args)
    elif args.command == "search":
        await handle_search(args)
    elif args.command == "ask":
        await handle_ask(args)
    elif args.command == "system":
        await handle_system(args)
    elif args.command == "list":
        await handle_list(args)
    elif args.command == "docinfo":
        await handle_docinfo(args)
    elif args.command == "server":
        # Server command should exit the async context and run directly
        return "server", args
    else:
        print("Please specify a command. Use --help for more information.")
        return None

def main():
    """Main entry point"""
    try:
        result = asyncio.run(main_async())
        
        # Handle server command outside of asyncio context
        if result and isinstance(result, tuple) and result[0] == "server":
            _, args = result
            handle_server(args)
            
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()