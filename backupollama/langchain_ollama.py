"""
LangChain Ollama Integration Module
This module provides integration with Ollama for LLM capabilities in the RAG system.
"""

from typing import Any, Dict, List, Mapping, Optional, Iterator
import time
import logging
import os
import json
import requests

# LangChain imports
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.schema import Generation, LLMResult
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Import configuration
from config import logger, OLLAMA_CONFIG

class OllamaLLM(LLM):
    """LangChain integration for Ollama models"""
    
    base_url: str = "http://localhost:11434"
    model: str = "mistral"
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 512
    system: Optional[str] = None
    
    def __init__(self, **kwargs):
        """Initialize the Ollama LLM"""
        super().__init__(**kwargs)
    
    @property
    def _llm_type(self) -> str:
        """Return the LLM type"""
        return "ollama"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        """Call the Ollama API"""
        # Prepare the request payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_predict": self.max_tokens,
            "stream": False
        }
        
        # Add system prompt if provided
        if self.system:
            payload["system"] = self.system
        
        # Add stop tokens if provided
        if stop:
            payload["stop"] = stop
        
        # Override with any kwargs
        for k, v in kwargs.items():
            if k in payload:
                payload[k] = v
        
        try:
            # Make the API request
            response = requests.post(
                f"{self.base_url}/api/generate",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=60
            )
            
            # Check for errors
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            
            # Return the generated text
            return result["response"]
            
        except requests.RequestException as e:
            logger.error(f"Error calling Ollama API: {str(e)}")
            raise
    
    def stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ):
        """Stream the response from Ollama"""
        # Prepare the request payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_predict": self.max_tokens,
            "stream": True
        }
        
        # Add system prompt if provided
        if self.system:
            payload["system"] = self.system
        
        # Add stop tokens if provided
        if stop:
            payload["stop"] = stop
        
        # Override with any kwargs
        for k, v in kwargs.items():
            if k in payload:
                payload[k] = v
        
        try:
            # Make the API request
            response = requests.post(
                f"{self.base_url}/api/generate",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                stream=True,
                timeout=60
            )
            
            # Check for errors
            response.raise_for_status()
            
            # Process the streaming response
            for line in response.iter_lines():
                if line:
                    # Parse the JSON response
                    try:
                        chunk = json.loads(line)
                        # If this is the end of the response, break
                        if chunk.get("done", False):
                            break
                        # Yield the token
                        yield chunk.get("response", "")
                    except json.JSONDecodeError:
                        logger.error(f"Error decoding JSON: {line}")
                        continue
                        
        except requests.RequestException as e:
            logger.error(f"Error streaming from Ollama API: {str(e)}")
            raise
    
    def _get_available_models(self) -> List[str]:
        """Get a list of available models from Ollama"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10
            )
            response.raise_for_status()
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        except requests.RequestException as e:
            logger.error(f"Error getting available models from Ollama: {str(e)}")
            return []

class OllamaRAGChain:
    """Retrieval-Augmented Generation chain using Ollama"""
    
    def __init__(
        self,
        retriever: Any,
        ollama_config: Optional[Dict[str, Any]] = None,
        prompt_template: Optional[str] = None
    ):
        """
        Initialize the Ollama RAG Chain.
        
        Args:
            retriever: Document retriever
            ollama_config: Configuration for Ollama
            prompt_template: Custom prompt template
        """
        # Use provided config or default from config module
        ollama_config = ollama_config or OLLAMA_CONFIG
        
        # Initialize the Ollama LLM
        self.llm = OllamaLLM(
            base_url=ollama_config.get("base_url", "http://localhost:11434"),
            model=ollama_config.get("models", {}).get("default", "mistral"),
            temperature=ollama_config.get("parameters", {}).get("temperature", 0.7),
            top_p=ollama_config.get("parameters", {}).get("top_p", 0.95),
            max_tokens=ollama_config.get("parameters", {}).get("max_tokens", 512),
            system=ollama_config.get("parameters", {}).get("system_prompt")
        )
        
        # Set the retriever
        self.retriever = retriever
        
        # Create the prompt template
        if prompt_template:
            self.prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
        else:
            # Default prompt template
            self.prompt = PromptTemplate(
                template="""
You are an AI assistant providing answers based solely on the given context. 
Be factual and concise, and only use information from the provided context.

Context:
{context}

Question: {question}

Answer:
""",
                input_variables=["context", "question"]
            )
        
        # Create the QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )
    
    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the RAG chain on a query.
        
        Args:
            query: User query
            
        Returns:
            Results with answer and source documents
        """
        start_time = time.time()
        
        # Run the chain
        result = self.qa_chain({"query": query})
        
        # Format the result
        formatted_result = {
            "question": query,
            "answer": result["result"],
            "source_documents": result["source_documents"],
            "processing_time_seconds": time.time() - start_time
        }
        
        return formatted_result
    
    def stream(self, query: str):
        """
        Stream the RAG response for a query.
        
        Args:
            query: User query
            
        Yields:
            Response tokens as they're generated
        """
        # Retrieve relevant documents
        docs = self.retriever.get_relevant_documents(query)
        
        # Prepare the context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Prepare the prompt
        prompt = self.prompt.format(context=context, question=query)
        
        # Stream the response
        for token in self.llm.stream(prompt):
            yield token