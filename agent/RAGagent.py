import asyncio
import traceback
import time
import httpx
from enum import Enum
from typing import Dict, Any, Optional, List, Union, TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

# LangChain imports
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

# Configuration
from config import (
    OLLAMA_BASE_URL,
    OLLAMA_DEFAULT_MODEL,
    OLLAMA_DEFAULT_TEMPERATURE,
    OLLAMA_DEFAULT_TOP_K,
    logger
)

class ModelType(Enum):
    DEFAULT = "default"
    OLLAMA = "ollama"

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

class GraphState(TypedDict):
    """
    Represents the state of our Self-RAG graph.
    """
    question: str
    generation: Optional[str]
    documents: List
    model_name: str
    temperature: float
    top_k: int

class SelfRAGService:
    def __init__(
        self, 
        orchestrator, 
        ollama_config: Optional[Dict[str, Any]] = None, 
        default_model: str = OLLAMA_DEFAULT_MODEL,
        default_temperature: float = OLLAMA_DEFAULT_TEMPERATURE,
        default_top_k: int = OLLAMA_DEFAULT_TOP_K
    ):
        """
        Initialize the Self-RAG service using LangChain's ChatOllama
        
        Args:
            orchestrator: The orchestrator for retrieval
            ollama_config: Optional configuration for Ollama (host, headers, etc.)
            default_model: Default Ollama model to use
            default_temperature: Default temperature for generation
            default_top_k: Default number of documents to retrieve
        """
        self.orchestrator = orchestrator
        self.ollama_config = ollama_config or {"base_url": OLLAMA_BASE_URL}
        self.default_model = default_model
        self.default_temperature = default_temperature
        self.default_top_k = default_top_k
        
        # Test connection during initialization
        self._test_ollama_connection()
        
    def _test_ollama_connection(self):
        """Test connection to Ollama server"""
        base_url = self.ollama_config.get("base_url", OLLAMA_BASE_URL)
        try:
            with httpx.Client() as client:
                response = client.get(f"{base_url}/api/version", timeout=5.0)
                if response.status_code == 200:
                    logger.info(f"Successfully connected to Ollama server at {base_url}")
                    return True
                else:
                    logger.warning(f"Error connecting to Ollama: Status code {response.status_code}")
                    return False
        except Exception as e:
            logger.warning(f"Could not connect to Ollama at {base_url}: {str(e)}")
            return False
        
    def _get_llm(self, model_name: str, temperature: float = 0.0):
        """Get a LangChain ChatOllama instance with the specified parameters"""
        try:
            # Create a ChatOllama instance with the given parameters
            llm = ChatOllama(
                model=model_name,
                temperature=temperature,
                **self.ollama_config
            )
            return llm
        except Exception as e:
            logger.error(f"Error creating ChatOllama instance: {e}", exc_info=True)
            raise
            
    async def _query_langchain_ollama(
        self, 
        model_name: str, 
        messages: List[Dict[str, str]],
        temperature: float = 0.0
    ) -> str:
        """
        Query Ollama API using LangChain's ChatOllama
        
        Args:
            model_name: Name of the Ollama model (e.g., "mistral", "llama3")
            messages: List of message objects with role and content
            temperature: Temperature for generation (higher = more creative)
            
        Returns:
            Response content as string
        """
        try:
            logger.debug(f"Starting _query_langchain_ollama with model: {model_name}, "
                    f"messages count: {len(messages)}, temperature: {temperature}")
            
            # Create LangChain messages from the input messages
            lc_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    lc_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    lc_messages.append(HumanMessage(content=msg["content"]))
                # We don't need to handle assistant messages since we're generating, not continuing
            
            # Get the LLM
            llm = self._get_llm(model_name, temperature)
            
            # Use a chain with StrOutputParser to get the string output
            chain = llm | StrOutputParser()
            
            # Execute the request
            logger.debug("Executing LangChain ChatOllama request")
            response = await chain.ainvoke(lc_messages)
            logger.debug("Received response from LangChain ChatOllama")
                
            return response
        
        except Exception as e:
            logger.error(f"Error during _query_langchain_ollama: {e}", exc_info=True)
            raise
    
    async def _query_langchain_ollama_with_retry(
        self, 
        model_name: str, 
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> str:
        """
        Query Ollama API with retry mechanism
        
        Args:
            model_name: Name of the Ollama model
            messages: List of message objects with role and content
            temperature: Temperature for generation
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (doubles with each attempt)
            
        Returns:
            Response content as string
        """
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return await self._query_langchain_ollama(
                    model_name, messages, temperature
                )
            except httpx.ConnectError as e:
                last_exception = e
                logger.warning(f"Connection error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    # Use exponential backoff
                    delay = retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Max retries reached. Failed to connect to Ollama.")
        
        # If we've exhausted all retries
        raise last_exception or RuntimeError("Failed to query Ollama after retries")
            
    async def grade_document_relevance(self, question: str, document: str, model_name: str, temperature: float) -> str:
        """Grade if a document is relevant to the question"""
        system_prompt = """You are a grader assessing relevance of a retrieved document to a user question.
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Retrieved document: \n\n {document} \n\n User question: {question}"}
        ]
        
        try:
            response = await self._query_langchain_ollama_with_retry(model_name, messages, temperature)
        except Exception as e:
            logger.warning(f"Error during relevance grading, assuming document is relevant: {e}")
            return "yes"  # Failsafe: include document if we can't grade it
            
        # Process the response to get yes/no
        if "yes" in response.lower():
            return "yes"
        else:
            return "no"
            
    async def grade_hallucination(self, documents: List, generation: str, model_name: str, temperature: float) -> str:
        """Grade if generation is grounded in the documents"""
        system_prompt = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
        
        # Format documents for the prompt
        docs_text = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(documents)])
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Set of facts: \n\n {docs_text} \n\n LLM generation: {generation}"}
        ]
        
        try:
            response = await self._query_langchain_ollama_with_retry(model_name, messages, temperature)
        except Exception as e:
            logger.warning(f"Error during hallucination grading, assuming answer is grounded: {e}")
            return "yes"  # Failsafe: assume grounded if we can't grade
            
        # Process the response to get yes/no
        if "yes" in response.lower():
            return "yes"
        else:
            return "no"
            
    async def grade_answer(self, question: str, generation: str, model_name: str, temperature: float) -> str:
        """Grade if the answer addresses the question"""
        system_prompt = """You are a grader assessing whether an answer addresses / resolves a question.
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User question: \n\n {question} \n\n LLM generation: {generation}"}
        ]
        
        try:
            response = await self._query_langchain_ollama_with_retry(model_name, messages, temperature)
        except Exception as e:
            logger.warning(f"Error during answer grading, assuming answer is relevant: {e}")
            return "yes"  # Failsafe: assume relevant if we can't grade
            
        # Process the response to get yes/no
        if "yes" in response.lower():
            return "yes"
        else:
            return "no"
            
    async def rewrite_question(self, question: str, model_name: str, temperature: float) -> str:
        """Rewrite the question to optimize for retrieval"""
        system_prompt = """You are a question re-writer that converts an input question to a better version that is optimized
        for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here is the initial question: \n\n {question} \n Formulate an improved question."}
        ]
        
        try:
            response = await self._query_langchain_ollama_with_retry(model_name, messages, temperature)
            return response
        except Exception as e:
            logger.warning(f"Error rewriting question, falling back to original: {e}")
            # Simple keyword expansion fallback
            return self._fallback_rewrite_question(question)
            
    def _fallback_rewrite_question(self, question: str) -> str:
        """Simple fallback method when Ollama is unavailable"""
        # Remove common question words and add some synonyms
        common_words = ["what", "who", "where", "when", "how", "why", "is", "are", "do", "does"]
        words = [w for w in question.lower().split() if w not in common_words]
        
        # Add some common synonyms (very basic)
        synonyms = []
        for word in words:
            if word == "best":
                synonyms.extend(["top", "greatest", "finest"])
            elif word == "find":
                synonyms.extend(["locate", "discover", "search"])
            # Add more synonyms as needed
        
        # Combine original question with keywords
        enhanced_question = f"{question} {' '.join(synonyms)}"
        logger.info(f"Used fallback rewrite: {enhanced_question}")
        return enhanced_question
        
    async def generate_answer(self, question: str, documents: List, model_name: str, temperature: float) -> str:
        """Generate answer based on documents and question"""
        system_prompt = """You are a helpful AI assistant. Use the provided context to answer the user's question.
        If the answer is not in the context, admit that you don't know."""
        
        # Format documents for the context
        context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(documents)])
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
        
        try:
            response = await self._query_langchain_ollama_with_retry(model_name, messages, temperature)
            return response
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            # Fallback to a simple answer based on documents
            return self._fallback_generate_answer(question, documents)
            
    def _fallback_generate_answer(self, question: str, documents: List) -> str:
        """Generate a simple answer when Ollama is unavailable"""
        if not documents:
            return "I'm sorry, I couldn't find any relevant information to answer your question. Please try rephrasing or asking a different question."
            
        # Return a simple answer based on document content
        return f"I found some relevant information but am having trouble processing it right now. Here's what I found:\n\n" + \
               "\n\n".join([f"- {doc.page_content[:200]}..." for doc in documents[:2]])

    # LangGraph nodes
    async def retrieve(self, state: GraphState) -> Dict:
        """Retrieve documents node"""
        logger.info("---RETRIEVE---")
        question = state["question"]
        model_name = state["model_name"]
        top_k = state["top_k"]

        # Retrieval using orchestrator
        documents = await self.orchestrator.search(
            query=question,
            top_k=top_k,
            search_type="hybrid"
        )
        
        return {
            "documents": documents.get("sources", []), 
            "question": question, 
            "model_name": model_name,
            "temperature": state["temperature"],
            "top_k": top_k
        }

    async def grade_documents(self, state: GraphState) -> Dict:
        """Grade documents node"""
        logger.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]
        model_name = state["model_name"]
        temperature = state["temperature"]
        top_k = state["top_k"]

        # Score each doc
        filtered_docs = []
        for doc in documents:
            score = await self.grade_document_relevance(
                question, 
                doc.get("content", ""), 
                model_name,
                temperature
            )
            if score == "yes":
                logger.info("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(doc)
            else:
                logger.info("---GRADE: DOCUMENT NOT RELEVANT---")
        
        return {
            "documents": filtered_docs, 
            "question": question, 
            "model_name": model_name,
            "temperature": temperature,
            "top_k": top_k
        }

    async def generate(self, state: GraphState) -> Dict:
        """Generate answer node"""
        logger.info("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        model_name = state["model_name"]
        temperature = state["temperature"]
        top_k = state["top_k"]

        # Convert document dicts to objects with page_content attribute
        doc_objects = []
        for doc in documents:
            class DocObj:
                def __init__(self, content):
                    self.page_content = content
            doc_obj = DocObj(doc.get("content", ""))
            doc_objects.append(doc_obj)

        # Generate answer
        generation = await self.generate_answer(question, doc_objects, model_name, temperature)
        
        return {
            "documents": documents, 
            "question": question, 
            "generation": generation, 
            "model_name": model_name,
            "temperature": temperature,
            "top_k": top_k
        }

    async def transform_query(self, state: GraphState) -> Dict:
        """Transform query node"""
        logger.info("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]
        model_name = state["model_name"]
        temperature = state["temperature"]
        top_k = state["top_k"]

        # Rewrite question
        better_question = await self.rewrite_question(question, model_name, temperature)
        
        return {
            "documents": documents, 
            "question": better_question, 
            "model_name": model_name,
            "temperature": temperature,
            "top_k": top_k
        }

    # Decision edges
    async def decide_to_generate(self, state: GraphState) -> str:
        """Decide whether to generate or transform query"""
        logger.info("---ASSESS GRADED DOCUMENTS---")
        filtered_documents = state["documents"]

        if not filtered_documents:
            # All documents have been filtered out
            logger.info("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            logger.info("---DECISION: GENERATE---")
            return "generate"

    async def grade_generation(self, state: GraphState) -> str:
        """Grade generation for hallucinations and relevance to question"""
        logger.info("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        model_name = state["model_name"]
        temperature = state["temperature"]

        # Convert document dicts to objects with page_content attribute
        doc_objects = []
        for doc in documents:
            class DocObj:
                def __init__(self, content):
                    self.page_content = content
            doc_obj = DocObj(doc.get("content", ""))
            doc_objects.append(doc_obj)

        # Check for hallucinations
        hallucination_score = await self.grade_hallucination(doc_objects, generation, model_name, temperature)
        
        if hallucination_score == "yes":
            logger.info("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            logger.info("---GRADE GENERATION vs QUESTION---")
            answer_score = await self.grade_answer(question, generation, model_name, temperature)
            
            if answer_score == "yes":
                logger.info("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                logger.info("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not_useful"
        else:
            logger.info("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not_supported"

    async def build_workflow(self) -> StateGraph:
        """Build the Self-RAG workflow graph"""
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)
        workflow.add_node("transform_query", self.transform_query)
        
        # Add edges
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            }
        )
        
        workflow.add_edge("transform_query", "retrieve")
        
        workflow.add_conditional_edges(
            "generate",
            self.grade_generation,
            {
                "not_supported": "generate",
                "useful": END,
                "not_useful": "transform_query",
            }
        )
        
        return workflow

    async def ask_with_selfrag(
        self,
        question: str,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        conversation_id: Optional[str] = None,
        ollama_host: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ask a question using Self-RAG workflow
        
        Args:
            question: Question to ask
            model_name: Ollama model to use (defaults to self.default_model)
            temperature: Temperature for generation (defaults to self.default_temperature)
            top_k: Number of documents to retrieve (defaults to self.default_top_k)
            conversation_id: Optional conversation ID
            ollama_host: Optional override for Ollama host
            
        Returns:
            Answer and sources
        """
        start_time = time.time()
        
        # Set defaults if not provided
        if model_name is None:
            model_name = self.default_model
            
        if temperature is None:
            temperature = self.default_temperature
            
        if top_k is None:
            top_k = self.default_top_k
            
        # Override Ollama host if specified
        original_base_url = None
        if ollama_host:
            original_base_url = self.ollama_config.get("base_url")
            self.ollama_config["base_url"] = ollama_host
            logger.info(f"Temporarily overriding Ollama host to: {ollama_host}")
            
            # Test connection to new host
            if not self._test_ollama_connection():
                logger.error(f"Cannot connect to specified Ollama host: {ollama_host}")
                if original_base_url:
                    self.ollama_config["base_url"] = original_base_url
                    logger.info(f"Restored original Ollama host: {original_base_url}")
                return {
                    "status": "error",
                    "error": f"Cannot connect to Ollama at {ollama_host}",
                    "error_type": "ConnectError"
                }
            
        logger.info(f"Self-RAG: Answering question '{question}' using {model_name} (temp={temperature}, top_k={top_k})")
        
        try:
            # Build the workflow
            workflow = await self.build_workflow()
            app = workflow.compile()
            
            # Run the workflow
            initial_state = {
                "question": question,
                "documents": [],
                "generation": None,
                "model_name": model_name,
                "temperature": temperature,
                "top_k": top_k
            }
            
            # Execute the workflow
            result = None
            async for output in app.astream(initial_state):
                # Keep track of the latest state
                for key, value in output.items():
                    result = value
                    
            # Extract the final answer and sources
            final_answer = result.get("generation", "")
            sources = result.get("documents", [])
            
            processing_time = time.time() - start_time
            
            # Reset original Ollama host if it was overridden
            if ollama_host and original_base_url:
                self.ollama_config["base_url"] = original_base_url
                logger.info(f"Restored original Ollama host: {original_base_url}")
            
            return {
                "status": "success",
                "question": question,
                "answer": final_answer,
                "sources": sources,
                "processing_time": processing_time,
                "conversation_id": conversation_id,
                "model_used": f"ollama/{model_name}",
                "temperature": temperature,
                "top_k": top_k
            }
            
        except Exception as e:
            # Reset original Ollama host if it was overridden
            if ollama_host and original_base_url:
                self.ollama_config["base_url"] = original_base_url
                logger.info(f"Restored original Ollama host: {original_base_url}")
                
            logger.error(f"Error in Self-RAG: {str(e)}", exc_info=True)
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error("Stack trace:")
            traceback.print_exc()
            
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
        
async def list_ollama_models(host: str = "http://localhost:11434"):
    """List available Ollama models from a specific host"""
    try:
        # Use LangChain's ChatOllama to list models
        from langchain_ollama import get_ollama_models
        
        # Get the list of models
        models = get_ollama_models(base_url=host)
        
        return {
            "status": "success",
            "models": [{"name": model} for model in models]
        }
    except ImportError:
        logger.error("LangChain Ollama library not installed")
        return {
            "status": "error",
            "error": "LangChain Ollama library not installed. Install with 'pip install langchain-ollama'"
        }
    except Exception as e:
        logger.error(f"Error listing Ollama models: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

# Function to pull a model from Ollama server
async def pull_ollama_model(model_name: str, host: str = "http://localhost:11434"):
    """Pull an Ollama model from a specific host"""
    try:
        # Use the ollama library for pulling models
        from ollama import Client
        
        # Create client with specific host
        client = Client(host=host)
        
        # Start pulling (this is synchronous and can take a while)
        logger.info(f"Starting to pull Ollama model: {model_name} from {host}")
        client.pull(model_name)
        logger.info(f"Successfully pulled Ollama model: {model_name}")
        
        return {
            "status": "success",
            "message": f"Successfully pulled model {model_name}"
        }
    except ImportError:
        logger.error("Ollama library not installed")
        return {
            "status": "error",
            "error": "Ollama Python library not installed. Install with 'pip install ollama'"
        }
    except Exception as e:
        logger.error(f"Error pulling Ollama model {model_name}: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }