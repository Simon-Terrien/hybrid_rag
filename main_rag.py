#!/usr/bin/env python3
"""
Main RAG Application with Docling Integration
This script provides a command-line interface to the enhanced RAG system with Docling integration.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Import configuration
from config import (
    DATA_DIR, PROCESSED_DIR, VECTOR_DB_DIR, logger,
    get_config, save_config, DOCLING_CONFIG, OLLAMA_CONFIG
)

# Import the original RAG orchestrator as a fallback
from rag_pipeline_orchestrator import RAGPipelineOrchestrator

# Flag to control Docling and Ollama usage
# Set these to False if you're experiencing issues
ENABLE_DOCLING = True
ENABLE_OLLAMA = True

# Only import Docling components if enabled
if ENABLE_DOCLING:
    try:
        # Try to import the Docling components
        from docling_rag_orchestrator import DoclingEnhancedRAGOrchestrator
    except ImportError:
        logger.warning("Could not import Docling components. Falling back to base RAG orchestrator.")
        ENABLE_DOCLING = False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Enhanced RAG Application with Docling Integration"
    )
    
    # Main commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Initialization command
    init_parser = subparsers.add_parser("init", help="Initialize the RAG system")
    init_parser.add_argument("--config", help="Path to configuration file")
    init_parser.add_argument("--no-docling", dest="use_docling", action="store_false", 
                           help="Don't use Docling integration")
    init_parser.add_argument("--no-ollama", dest="use_ollama", action="store_false", 
                           help="Don't use Ollama integration")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process documents")
    process_parser.add_argument("--files", nargs="+", help="Specific files to process")
    process_parser.add_argument("--dir", help="Directory to process (subdirectory of data_dir)")
    process_parser.add_argument("--force", action="store_true", help="Force reprocessing of documents")
    process_parser.add_argument("--no-docling", dest="use_docling", action="store_false", 
                              help="Don't use Docling for processing")
    process_parser.add_argument("--async", dest="async_mode", action="store_true", 
                              help="Process asynchronously")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index documents")
    index_parser.add_argument("--files", nargs="+", help="Specific files to index (LangChain only)")
    index_parser.add_argument("--force", action="store_true", help="Force reindexing of documents")
    index_parser.add_argument("--no-langchain", dest="use_langchain", action="store_false", 
                            help="Don't use LangChain for indexing")
    index_parser.add_argument("--vector-store", default="default", 
                            help="Vector store ID (LangChain only)")
    index_parser.add_argument("--async", dest="async_mode", action="store_true", 
                            help="Index asynchronously")
    
    # Process and index command
    process_index_parser = subparsers.add_parser("process-index", 
                                               help="Process and index documents in one step")
    process_index_parser.add_argument("--files", nargs="+", help="Specific files to process and index")
    process_index_parser.add_argument("--dir", help="Directory to process")
    process_index_parser.add_argument("--force-process", action="store_true", 
                                    help="Force reprocessing")
    process_index_parser.add_argument("--force-index", action="store_true", 
                                    help="Force reindexing")
    process_index_parser.add_argument("--no-docling", dest="use_docling", action="store_false", 
                                    help="Don't use Docling for processing")
    process_index_parser.add_argument("--no-langchain", dest="use_langchain", action="store_false", 
                                    help="Don't use LangChain for indexing")
    process_index_parser.add_argument("--vector-store", default="default", 
                                    help="Vector store ID")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--top-k", type=int, default=5, 
                             help="Number of results to return")
    search_parser.add_argument("--filter", action="append", 
                             help="Metadata filters (key:value format)")
    search_parser.add_argument("--type", choices=["hybrid", "semantic"], default="hybrid", 
                             help="Search type")
    search_parser.add_argument("--no-langchain", dest="use_langchain", action="store_false", 
                             help="Don't use LangChain for search")
    search_parser.add_argument("--vector-store", default="default", 
                             help="Vector store ID (LangChain only)")
    search_parser.add_argument("--no-rerank", dest="rerank", action="store_false", 
                             help="Don't use reranker")
    search_parser.add_argument("--output", choices=["text", "json"], default="text", 
                             help="Output format")
    
    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.add_argument("--top-k", type=int, default=5, 
                          help="Number of context documents")
    ask_parser.add_argument("--filter", action="append", 
                          help="Metadata filters (key:value format)")
    ask_parser.add_argument("--no-langchain", dest="use_langchain", action="store_false", 
                          help="Don't use LangChain for QA")
    ask_parser.add_argument("--vector-store", default="default", 
                          help="Vector store ID (LangChain only)")
    ask_parser.add_argument("--custom-prompt", 
                          help="Path to custom prompt template file")
    ask_parser.add_argument("--conversation", 
                          help="Conversation ID for base QA")
    ask_parser.add_argument("--stream", action="store_true", 
                          help="Stream the response (LangChain only)")
    ask_parser.add_argument("--output", choices=["text", "json"], default="text", 
                          help="Output format")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show system status")
    status_parser.add_argument("--output", choices=["text", "json"], default="text", 
                             help="Output format")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_subparsers = config_parser.add_subparsers(dest="config_command", 
                                                   help="Configuration command")
    
    # Config show command
    config_show_parser = config_subparsers.add_parser("show", help="Show configuration")
    config_show_parser.add_argument("--output", choices=["text", "json"], default="text", 
                                  help="Output format")
    
    # Config update command
    config_update_parser = config_subparsers.add_parser("update", help="Update configuration")
    config_update_parser.add_argument("--docling", action="store_true", 
                                    help="Use Docling for processing")
    config_update_parser.add_argument("--no-docling", dest="docling", action="store_false", 
                                    help="Don't use Docling for processing")
    config_update_parser.add_argument("--langchain", action="store_true", 
                                    help="Use LangChain for RAG")
    config_update_parser.add_argument("--no-langchain", dest="langchain", action="store_false", 
                                    help="Don't use LangChain for RAG")
    config_update_parser.add_argument("--hybrid-search", action="store_true", 
                                    help="Use hybrid search")
    config_update_parser.add_argument("--no-hybrid-search", dest="hybrid_search", 
                                    action="store_false", help="Don't use hybrid search")
    config_update_parser.add_argument("--reranker", action="store_true", 
                                    help="Use contextual reranker")
    config_update_parser.add_argument("--no-reranker", dest="reranker", action="store_false", 
                                    help="Don't use contextual reranker")
    
    return parser.parse_args()

def parse_filters(filter_strings):
    """Parse filter strings in key:value format"""
    if not filter_strings:
        return None
    
    filters = {}
    for filter_str in filter_strings:
        key, value = filter_str.split(":", 1)
        
        # Try to convert value to appropriate type
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif value.isdigit():
            value = int(value)
        elif value.replace(".", "", 1).isdigit() and value.count(".") == 1:
            value = float(value)
        
        filters[key] = value
    
    return filters

def format_output(data, format_type="text"):
    """Format output based on specified format"""
    if format_type == "json":
        return json.dumps(data, indent=2)
    
    # Text format (default)
    if isinstance(data, dict):
        output = []
        for key, value in data.items():
            if isinstance(value, dict) or isinstance(value, list):
                output.append(f"{key}:")
                output.append(format_output(value, format_type))
            else:
                output.append(f"{key}: {value}")
        return "\n".join(output)
    elif isinstance(data, list):
        output = []
        for i, item in enumerate(data):
            if isinstance(item, dict) or isinstance(item, list):
                output.append(f"[{i}]:")
                output.append(format_output(item, format_type))
            else:
                output.append(f"[{i}] {item}")
        return "\n".join(output)
    else:
        return str(data)

def load_custom_prompt(file_path):
    """Load custom prompt template from file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def get_orchestrator(use_docling=True, use_ollama=True):
    """Get appropriate orchestrator based on available components"""
    global ENABLE_DOCLING
    
    if ENABLE_DOCLING and use_docling:
        try:
            # Use the enhanced orchestrator with Docling
            return DoclingEnhancedRAGOrchestrator(
                use_docling=True,
                use_langchain=True,
                use_hybrid_search=True,
                use_reranker=True
            )
        except Exception as e:
            logger.error(f"Error initializing DoclingEnhancedRAGOrchestrator: {str(e)}")
            ENABLE_DOCLING = False
    
    # Fall back to original orchestrator
    return RAGPipelineOrchestrator()

def handle_init(args, config):
    """Handle the init command"""
    print("Initializing RAG system...")
    
    # Load custom config if provided
    if args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                custom_config = json.load(f)
                # Merge with default config
                config.update(custom_config)
        except Exception as e:
            print(f"Error loading configuration: {str(e)}")
            return
    
    # Initialize the orchestrator with requested settings
    orchestrator = get_orchestrator(
        use_docling=args.use_docling if hasattr(args, 'use_docling') else True,
        use_ollama=args.use_ollama if hasattr(args, 'use_ollama') else True
    )
    
    # Get system state to verify initialization
    state = orchestrator.get_state()
    
    print("RAG system initialized with the following components:")
    for component, status in state["components"].items():
        print(f"  {component}: {status['status']}")
    
    print("\nSystem is ready to use.")

def handle_process(args, config):
    """Handle the process command"""
    print("Processing documents...")
    
    # Initialize orchestrator
    orchestrator = get_orchestrator(
        use_docling=args.use_docling if hasattr(args, 'use_docling') else True
    )
    
    # Process specific files or directory
    if args.files:
        source_paths = args.files
        print(f"Processing {len(source_paths)} files...")
    else:
        source_paths = None
        print(f"Processing directory {args.dir or 'data'}...")
    
    # Execute processing
    # Check if we're using the Docling orchestrator and have specific files
    if ENABLE_DOCLING and hasattr(orchestrator, 'process_documents') and source_paths:
        result = orchestrator.process_documents(
            source_paths=source_paths,
            subdirectory=args.dir,
            force_reprocess=args.force,
            use_docling=args.use_docling if hasattr(args, 'use_docling') else True,
            async_mode=args.async_mode if hasattr(args, 'async_mode') else False
        )
    else:
        # Original orchestrator or directory processing
        result = orchestrator.process_documents(
            subdirectory=args.dir,
            force_reprocess=args.force,
            async_mode=args.async_mode if hasattr(args, 'async_mode') else False
        )
    
    # Display results
    if hasattr(args, 'async_mode') and args.async_mode:
        print(f"Processing task started: {result.get('task_id', 'unknown')}")
    else:
        print(f"Processing complete:")
        print(f"  Documents processed: {result.get('processed_documents', 0)}")
        print(f"  Documents skipped: {result.get('skipped_documents', 0)}")
        print(f"  Documents failed: {result.get('failed_documents', 0)}")
        print(f"  Total time: {result.get('processing_time_seconds', 0):.2f} seconds")

def handle_index(args, config):
    """Handle the index command"""
    print("Indexing documents...")
    
    # Initialize orchestrator
    orchestrator = get_orchestrator()
    
    # Index specific files or all processed documents
    if args.files and ENABLE_DOCLING and hasattr(orchestrator, 'index_documents') and hasattr(orchestrator.index_documents, '__code__') and 'source_paths' in orchestrator.index_documents.__code__.co_varnames:
        source_paths = args.files
        print(f"Indexing {len(source_paths)} files...")
        
        # Execute indexing with source paths
        result = orchestrator.index_documents(
            source_paths=source_paths,
            force_reindex=args.force,
            use_langchain=args.use_langchain if hasattr(args, 'use_langchain') else True,
            vector_store_id=args.vector_store,
            async_mode=args.async_mode if hasattr(args, 'async_mode') else False
        )
    else:
        # Original orchestrator or all processed documents
        source_paths = None
        print(f"Indexing processed documents...")
        
        # Execute indexing without source paths
        result = orchestrator.index_documents(
            force_reindex=args.force,
            async_mode=args.async_mode if hasattr(args, 'async_mode') else False
        )
    
    # Display results
    if hasattr(args, 'async_mode') and args.async_mode:
        print(f"Indexing task started: {result.get('task_id', 'unknown')}")
    else:
        print("Indexing complete:")
        if "document_count" in result:
            print(f"  Documents indexed: {result['document_count']}")
        elif "indexed_documents" in result:
            print(f"  Documents indexed: {result['indexed_documents']}")
            print(f"  Documents skipped: {result['skipped_documents']}")
        print(f"  Vector store: {args.vector_store if hasattr(args, 'vector_store') else 'default'}")
        print(f"  Total time: {result.get('total_processing_time', 0):.2f} seconds")

def handle_process_index(args, config):
    """Handle the process-index command"""
    print("Processing and indexing documents...")
    
    # Initialize orchestrator
    orchestrator = get_orchestrator(
        use_docling=args.use_docling if hasattr(args, 'use_docling') else True
    )
    
    # Check if we're using Docling orchestrator and it has process_and_index method
    if ENABLE_DOCLING and hasattr(orchestrator, 'process_and_index'):
        # Process and index with Docling
        result = orchestrator.process_and_index(
            source_paths=args.files,
            subdirectory=args.dir,
            force_reprocess=args.force_process,
            force_reindex=args.force_index,
            use_docling=args.use_docling if hasattr(args, 'use_docling') else True,
            use_langchain=args.use_langchain if hasattr(args, 'use_langchain') else True,
            vector_store_id=args.vector_store
        )
        
        # Display results
        print("Processing and indexing complete:")
        
        # Processing results
        process_result = result.get("processing", {})
        print("Processing:")
        print(f"  Documents processed: {process_result.get('processed_documents', 0)}")
        print(f"  Documents skipped: {process_result.get('skipped_documents', 0)}")
        print(f"  Documents failed: {process_result.get('failed_documents', 0)}")
        
        # Indexing results
        index_result = result.get("indexing", {})
        print("Indexing:")
        if "document_count" in index_result:
            print(f"  Documents indexed: {index_result.get('document_count', 0)}")
        else:
            print(f"  Documents indexed: {index_result.get('indexed_documents', 0)}")
        
        print(f"Total time: {result.get('total_time_seconds', 0):.2f} seconds")
    else:
        # Original orchestrator - process then index sequentially
        print("Processing documents...")
        process_result = orchestrator.process_documents(
            subdirectory=args.dir,
            force_reprocess=args.force_process,
            async_mode=False
        )
        
        print(f"Processing complete: {process_result.get('processed_documents', 0)} documents processed")
        
        print("Indexing documents...")
        index_result = orchestrator.index_documents(
            force_reindex=args.force_index,
            async_mode=False
        )
        
        print(f"Indexing complete: {index_result.get('indexed_documents', 0)} documents indexed")

def handle_search(args, config):
    """Handle the search command"""
    print(f"Searching for: {args.query}")
    
    # Parse filters
    filters = parse_filters(args.filter)
    
    # Initialize orchestrator
    orchestrator = get_orchestrator()
    
    # Execute search
    if ENABLE_DOCLING and hasattr(orchestrator, 'search') and hasattr(orchestrator.search, '__code__') and 'use_langchain' in orchestrator.search.__code__.co_varnames:
        # Enhanced search with Docling orchestrator
        result = orchestrator.search(
            query=args.query,
            top_k=args.top_k,
            filter_dict=filters,
            search_type=args.type,
            use_langchain=args.use_langchain if hasattr(args, 'use_langchain') else True,
            vector_store_id=args.vector_store if hasattr(args, 'vector_store') else "default",
            rerank_results=args.rerank if hasattr(args, 'rerank') else True
        )
    else:
        # Original search with base orchestrator
        result = orchestrator.search(
            query=args.query,
            top_k=args.top_k,
            filters=filters,
            search_type=args.type,
            rerank_results=args.rerank if hasattr(args, 'rerank') else True,
            return_all_chunks=False
        )
    
    # Format and display results
    if hasattr(args, 'output') and args.output == "json":
        print(json.dumps(result, indent=2))
        return
    
    # Text output
    print(f"Found {result.get('total_results', 0)} results:")
    
    if "results" in result:
        if isinstance(result["results"], list):
            for i, item in enumerate(result["results"]):
                print(f"\nResult {i+1}:")
                
                # Handle different result structures
                if "text" in item:
                    # Individual chunk result
                    print(f"  Text: {item['text'][:200]}...")
                    print(f"  Score: {item.get('combined_score', item.get('similarity', 0)):.4f}")
                    
                    if "metadata" in item:
                        metadata = item["metadata"]
                        if "document_id" in metadata:
                            print(f"  Document ID: {metadata['document_id']}")
                        if "headings" in metadata:
                            print(f"  Headings: {metadata['headings']}")
                else:
                    # Document result with chunks
                    print(f"  Document ID: {item.get('document_id', 'unknown')}")
                    print(f"  Score: {item.get('max_score', 0):.4f}")
                    
                    if "metadata" in item:
                        metadata = item["metadata"]
                        if "filename" in metadata:
                            print(f"  Filename: {metadata['filename']}")
                    
                    if "chunks" in item and item["chunks"]:
                        print(f"  Top chunk: {item['chunks'][0]['text'][:200]}...")
    
    print(f"\nSearch completed in {result.get('processing_time_seconds', 0):.2f} seconds")

def handle_ask(args, config):
    """Handle the ask command"""
    print(f"Question: {args.question}")
    
    # Parse filters
    filters = parse_filters(args.filter)
    
    # Load custom prompt if provided
    prompt_template = None
    if hasattr(args, 'custom_prompt') and args.custom_prompt:
        try:
            prompt_template = load_custom_prompt(args.custom_prompt)
        except Exception as e:
            print(f"Error loading custom prompt: {str(e)}")
    
    # Initialize orchestrator
    orchestrator = get_orchestrator()
    
    # Execute QA
    if ENABLE_DOCLING and hasattr(orchestrator, 'ask') and hasattr(orchestrator.ask, '__code__') and 'use_langchain' in orchestrator.ask.__code__.co_varnames:
        # Enhanced QA with Docling orchestrator
        result = orchestrator.ask(
            question=args.question,
            top_k=args.top_k,
            filter_dict=filters,
            use_langchain=args.use_langchain if hasattr(args, 'use_langchain') else True,
            vector_store_id=args.vector_store if hasattr(args, 'vector_store') else "default",
            prompt_template=prompt_template,
            streaming=args.stream if hasattr(args, 'stream') else False,
            conversation_id=args.conversation if hasattr(args, 'conversation') else None
        )
    else:
        # Original QA with base orchestrator
        search_params = {
            "top_k": args.top_k,
            "filters": filters,
            "search_type": "hybrid",
            "rerank_results": True
        }
        
        result = orchestrator.ask(
            question=args.question,
            conversation_id=args.conversation if hasattr(args, 'conversation') else None,
            search_params=search_params
        )
    
    # Format and display results
    if hasattr(args, 'output') and args.output == "json":
        print(json.dumps(result, indent=2))
        return
    
    # Text output
    if "answer" in result:
        print("\nAnswer:")
        print(result["answer"])
        
        if "sources" in result and result["sources"]:
            print("\nSources:")
            for i, source in enumerate(result["sources"]):
                print(f"  {i+1}. {source.get('text_preview', '')}")
                if "metadata" in source:
                    metadata = source["metadata"]
                    if "document_id" in metadata:
                        print(f"     Document: {metadata['document_id']}")
                    if "page" in source:
                        print(f"     Page: {source['page']}")
        
        print(f"\nProcessing time: {result.get('processing_time_seconds', 0):.2f} seconds")
    else:
        print("No answer generated.")

def handle_status(args, config):
    """Handle the status command"""
    print("Getting system status...")
    
    # Initialize orchestrator
    orchestrator = get_orchestrator()
    
    # Get system state
    state = orchestrator.get_state()
    
    # Format and display status
    if hasattr(args, 'output') and args.output == "json":
        print(json.dumps(state, indent=2))
        return
    
    # Text output
    print(f"System status: {state['status']}")
    print(f"Last update: {state['last_update']}")
    
    print("\nComponents:")
    for component, status in state["components"].items():
        print(f"  {component}: {status['status']}")
        if "error" in status:
            print(f"    Error: {status['error']}")
    
    print("\nStatistics:")
    stats = state["statistics"]
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Total queries: {stats['total_queries']}")
    
    if stats.get('last_processing_time'):
        print(f"  Last processing: {stats['last_processing_time']}")
    
    if stats.get('last_indexing_time'):
        print(f"  Last indexing: {stats['last_indexing_time']}")

def handle_config(args, config):
    """Handle the config command"""
    if args.config_command == "show":
        # Show configuration
        if hasattr(args, 'output') and args.output == "json":
            print(json.dumps(config, indent=2))
        else:
            print("Current configuration:")
            print(format_output(config))
    else:
        # Update configuration
        updates = {}
        
        # Check which parameters are explicitly set
        if hasattr(args, 'docling') and args.docling is not None:
            updates["use_docling"] = args.docling
        
        if hasattr(args, 'langchain') and args.langchain is not None:
            updates["use_langchain"] = args.langchain
        
        if hasattr(args, 'hybrid_search') and args.hybrid_search is not None:
            updates["use_hybrid_search"] = args.hybrid_search
        
        if hasattr(args, 'reranker') and args.reranker is not None:
            updates["use_reranker"] = args.reranker
        
        if updates:
            # Save to configuration
            user_config = {
                "docling_integration": updates
            }
            
            # Write to user config file
            config_path = Path(os.path.expanduser("~/.rag_config.json"))
            
            try:
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(user_config, indent=2, fp=f)
                
                print("Configuration updated:")
                for key, value in updates.items():
                    print(f"  {key}: {value}")
            except Exception as e:
                print(f"Error saving configuration: {str(e)}")
        else:
            print("No configuration updates specified.")

def main():
    """Main entry point"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config = get_config()
    
    # Handle commands
    if args.command == "init":
        handle_init(args, config)
    elif args.command == "process":
        handle_process(args, config)
    elif args.command == "index":
        handle_index(args, config)
    elif args.command == "process-index":
        handle_process_index(args, config)
    elif args.command == "search":
        handle_search(args, config)
    elif args.command == "ask":
        handle_ask(args, config)
    elif args.command == "status":
        handle_status(args, config)
    elif args.command == "config":
        handle_config(args, config)
    else:
        print("No command specified. Use --help to see available commands.")

if __name__ == "__main__":
    main()