"""
This script creates the necessary Python package structure and fixes import issues
for the Docling RAG integration.
"""

import os
import sys
from pathlib import Path

def create_package_structure():
    """Create the necessary package structure"""
    # Create __init__.py files
    init_files = [
        "__init__.py",
        "langchain_ollama/__init__.py"
    ]
    
    for file_path in init_files:
        path = Path(file_path)
        path.parent.mkdir(exist_ok=True, parents=True)
        
        if not path.exists():
            with open(path, "w") as f:
                if "langchain_ollama" in str(path):
                    f.write('from .ollama import OllamaLLM, OllamaRAGChain\n')
                else:
                    f.write('# RAG System package\n')
    
    # Create the langchain_ollama module structure
    ollama_module_path = Path("langchain_ollama/ollama.py")
    if not ollama_module_path.exists():
        # Copy the content from existing file if it exists
        source_file = Path("langchain_ollama.py")
        if source_file.exists():
            with open(source_file, "r") as src, open(ollama_module_path, "w") as dst:
                dst.write(src.read())
        else:
            print(f"Warning: {source_file} not found. Please create it manually.")
    
    print("Package structure created successfully.")

def fix_imports():
    """Fix import issues in Python files"""
    files_to_check = [
        "docling_rag_orchestrator.py",
        "langchain_integration.py",
        "main_rag.py"
    ]
    
    for file_path in files_to_check:
        path = Path(file_path)
        if not path.exists():
            print(f"Warning: {file_path} not found. Skipping.")
            continue
        
        with open(path, "r") as f:
            content = f.read()
        
        # Fix langchain_ollama import
        if "from langchain_ollama import" in content:
            content = content.replace(
                "from langchain_ollama import",
                "try:\n    from langchain_ollama import\nexcept ImportError:\n    from .langchain_ollama import"
            )
        
        # Fix other potential import issues
        # Add more replacements as needed
        
        with open(path, "w") as f:
            f.write(content)
        
        print(f"Fixed imports in {file_path}")

if __name__ == "__main__":
    create_package_structure()
    fix_imports()
    print("Import issues fixed. Please try running the application again.")