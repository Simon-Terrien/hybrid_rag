from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
import time
from datetime import datetime
import os

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Docling imports
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

# For LLM integration
from langchain_huggingface import HuggingFaceEndpoint

# Import our existing components
from config import (
    DATA_DIR, PROCESSED_DIR, VECTOR_DB_DIR, logger,
    ROOT_DIR, LLM_CONFIG, SEARCH_CONFIG
)
from utils import (
    load_json, save_json, timed, log_exceptions
)

class LangChainRAGAdapter:
    """Adapter for using LangChain components with our RAG system"""
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
        vector_db_dir: Optional[Path] = None,
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v1",
        vector_db_type: str = "faiss",
        use_docling: bool = True,
        use_docling_chunking: bool = True
    ):
        """
        Initialize the LangChain RAG adapter.
        
        Args:
            data_dir: Raw data directory (default: from config)
            processed_dir: Processed data directory (default: from config)
            vector_db_dir: Vector database directory (default: from config)
            embedding_model_name: Name of the embedding model
            vector_db_type: Type of vector database ('faiss' or 'chroma')
            use_docling: Whether to use Docling for document processing
            use_docling_chunking: Whether to use Docling's chunking capabilities
        """
        self.data_dir = data_dir or DATA_DIR
        self.processed_dir = processed_dir or PROCESSED_DIR
        self.vector_db_dir = vector_db_dir or VECTOR_DB_DIR
        self.embedding_model_name = embedding_model_name
        self.vector_db_type = vector_db_type.lower()
        self.use_docling = use_docling
        self.use_docling_chunking = use_docling_chunking
        
        # Initialize the embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cuda" if self._is_cuda_available() else "cpu"}
        )
        
        # Initialize vector store to None (will be created or loaded on demand)
        self.vector_store = None
        
        # Configure LLM
        self.llm = self._initialize_llm()
        
        # Default QA prompt
        self.qa_prompt = PromptTemplate(
            template="""Tu es un assistant IA expert chargé de fournir des réponses précises basées uniquement sur les documents fournis.

Contexte des documents:
{context}

Question de l'utilisateur: {question}

Instructions:
1. Réponds uniquement à partir des informations fournies dans le contexte ci-dessus.
2. Si tu ne trouves pas la réponse dans le contexte, dis simplement que tu ne peux pas répondre.
3. Ne fabrique pas d'informations ou de connaissances qui ne sont pas présentes dans le contexte.
4. Cite la source exacte lorsque tu fournis des informations.
5. Présente ta réponse de manière claire et structurée.

Réponds de façon concise à la question suivante en te basant uniquement sur le contexte fourni:""",
            input_variables=["context", "question"]
        )
        
        logger.info(f"LangChainRAGAdapter initialized with {embedding_model_name} embeddings and {vector_db_type} vector store")
    
    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            logger.warning("PyTorch not installed, using CPU for embeddings")
            return False
    
    def _initialize_llm(self):
        """Initialize the language model based on configuration"""
        llm_config = LLM_CONFIG
        
        if llm_config.get("local_model", True):
            try:
                # Try to use local Hugging Face pipeline
                from langchain_huggingface import HuggingFacePipeline
                
                logger.info(f"Initializing local LLM: {llm_config['model_name']}")
                
                # Import required components
                import torch
                from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
                
                # Check for CUDA
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # Load model and tokenizer
                tokenizer = AutoTokenizer.from_pretrained(llm_config["model_name"])
                
                # Configure model loading options
                model_kwargs = {}
                if device == "cuda":
                    model_kwargs.update({
                        "torch_dtype": torch.float16,
                        "device_map": "auto",
                        "load_in_8bit": llm_config.get("use_8bit_quantization", False)
                    })
                
                # Load the model
                model = AutoModelForCausalLM.from_pretrained(
                    llm_config["model_name"],
                    **model_kwargs
                )
                
                # Create pipeline
                text_generation_pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=llm_config.get("max_new_tokens", 512),
                    temperature=llm_config.get("temperature", 0.7),
                    top_p=llm_config.get("top_p", 0.95),
                    top_k=llm_config.get("top_k", 50),
                    repetition_penalty=llm_config.get("repetition_penalty", 1.1),
                    pad_token_id=tokenizer.eos_token_id
                )
                
                # Create LangChain LLM
                return HuggingFacePipeline(
                    pipeline=text_generation_pipeline,
                    model_id=llm_config["model_name"]
                )
                
            except Exception as e:
                logger.error(f"Error initializing local LLM: {str(e)}")
                logger.warning("Falling back to HuggingFace Inference API")
                
                # Fall back to Hugging Face Inference API
                api_key = llm_config.get("api_key", os.environ.get("HF_TOKEN"))
                api_base = llm_config.get("api_base")
                
                if not api_key:
                    logger.warning("No API key provided for HuggingFace Inference API")
                
                return HuggingFaceEndpoint(
                    repo_id=llm_config["model_name"],
                    huggingfacehub_api_token=api_key,
                    endpoint_url=api_base if api_base else None,
                    max_new_tokens=llm_config.get("max_new_tokens", 512),
                    temperature=llm_config.get("temperature", 0.7),
                    top_p=llm_config.get("top_p", 0.95),
                    top_k=llm_config.get("top_k", 50),
                    repetition_penalty=llm_config.get("repetition_penalty", 1.1)
                )
        else:
            # Use Hugging Face Inference API
            api_key = llm_config.get("api_key", os.environ.get("HF_TOKEN"))
            api_base = llm_config.get("api_base")
            
            if not api_key:
                logger.warning("No API key provided for HuggingFace Inference API")
            
            return HuggingFaceEndpoint(
                repo_id=llm_config["model_name"],
                huggingfacehub_api_token=api_key,
                endpoint_url=api_base if api_base else None,
                max_new_tokens=llm_config.get("max_new_tokens", 512),
                temperature=llm_config.get("temperature", 0.7),
                top_p=llm_config.get("top_p", 0.95),
                top_k=llm_config.get("top_k", 50),
                repetition_penalty=llm_config.get("repetition_penalty", 1.1)
            )
    
    @timed(logger=logger)
    @log_exceptions(logger=logger)
    def load_documents(
        self,
        source_paths: Union[str, List[str], Path, List[Path]],
        use_docling: Optional[bool] = None
    ) -> List[Document]:
        """
        Load documents using LangChain document loaders.
        
        Args:
            source_paths: Path(s) to documents or directories
            use_docling: Whether to use Docling for document loading
            
        Returns:
            List of loaded documents
        """
        # Normalize source_paths to list
        if not isinstance(source_paths, list):
            source_paths = [source_paths]
        
        # Convert to string paths
        source_paths = [str(p) for p in source_paths]
        
        # Determine whether to use Docling
        use_docling_loader = use_docling if use_docling is not None else self.use_docling
        
        documents = []
        
        if use_docling_loader:
            # Use Docling loader
            logger.info(f"Loading documents with Docling: {source_paths}")
            
            # Configure export type based on chunking preference
            export_type = ExportType.DOC_CHUNKS if self.use_docling_chunking else ExportType.MARKDOWN
            
            # Create Docling loader
            loader = DoclingLoader(
                file_path=source_paths,
                export_type=export_type
            )
            
            # Load documents
            docling_docs = loader.load()
            documents.extend(docling_docs)
            
            # If using markdown export and not Docling chunking, we need to chunk the documents
            if export_type == ExportType.MARKDOWN:
                # Use LangChain text splitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=SEARCH_CONFIG["max_tokens_per_chunk"],
                    chunk_overlap=SEARCH_CONFIG["max_tokens_per_chunk"] // 5,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                
                # Split documents
                documents = text_splitter.split_documents(documents)
        else:
            # Use standard LangChain loaders
            logger.info(f"Loading documents with standard LangChain loaders: {source_paths}")
            
            for source_path in source_paths:
                path = Path(source_path)
                
                if path.is_dir():
                    # Load from directory
                    loader = DirectoryLoader(source_path, glob="**/*.*")
                    dir_docs = loader.load()
                    
                    # Apply text splitting
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=SEARCH_CONFIG["max_tokens_per_chunk"],
                        chunk_overlap=SEARCH_CONFIG["max_tokens_per_chunk"] // 5,
                        separators=["\n\n", "\n", ". ", " ", ""]
                    )
                    
                    split_docs = text_splitter.split_documents(dir_docs)
                    documents.extend(split_docs)
                else:
                    # Load single file
                    if path.suffix.lower() == '.txt':
                        loader = TextLoader(source_path)
                        file_docs = loader.load()
                        
                        # Apply text splitting
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=SEARCH_CONFIG["max_tokens_per_chunk"],
                            chunk_overlap=SEARCH_CONFIG["max_tokens_per_chunk"] // 5,
                            separators=["\n\n", "\n", ". ", " ", ""]
                        )
                        
                        split_docs = text_splitter.split_documents(file_docs)
                        documents.extend(split_docs)
                    else:
                        logger.warning(f"Unsupported file type for standard loader: {path.suffix}")
        
        logger.info(f"Loaded {len(documents)} documents/chunks")
        return documents
    
    @timed(logger=logger)
    @log_exceptions(logger=logger)
    def create_vector_store(
        self,
        documents: List[Document],
        vector_store_id: str = "default"
    ) -> None:
        """
        Create a vector store from documents.
        
        Args:
            documents: List of documents to index
            vector_store_id: Identifier for the vector store
        """
        logger.info(f"Creating {self.vector_db_type} vector store with {len(documents)} documents")
        
        # Ensure vector_db_dir exists
        self.vector_db_dir.mkdir(exist_ok=True, parents=True)
        
        if self.vector_db_type == "faiss":
            # Create FAISS vector store
            vector_store = FAISS.from_documents(
                documents,
                self.embeddings
            )
            
            # Save vector store
            vector_store_path = self.vector_db_dir / f"{vector_store_id}"
            vector_store.save_local(str(vector_store_path))
            
            self.vector_store = vector_store
            
        elif self.vector_db_type == "chroma":
            # Create Chroma vector store
            persist_directory = str(self.vector_db_dir / f"{vector_store_id}_chroma")
            
            vector_store = Chroma.from_documents(
                documents,
                self.embeddings,
                persist_directory=persist_directory
            )
            
            vector_store.persist()
            self.vector_store = vector_store
            
        else:
            raise ValueError(f"Unsupported vector store type: {self.vector_db_type}")
        
        logger.info(f"Vector store created and saved to {self.vector_db_dir}")
    
    @timed(logger=logger)
    @log_exceptions(logger=logger)
    def load_vector_store(self, vector_store_id: str = "default") -> None:
        """
        Load a vector store.
        
        Args:
            vector_store_id: Identifier for the vector store
        """
        if self.vector_db_type == "faiss":
            # Check if vector store exists
            vector_store_path = self.vector_db_dir / f"{vector_store_id}"
            index_path = vector_store_path / "index.faiss"
            
            if not index_path.exists():
                raise FileNotFoundError(f"Vector store not found at {vector_store_path}")
            
            # Load FAISS vector store
            self.vector_store = FAISS.load_local(
                str(vector_store_path),
                self.embeddings
            )
            
        elif self.vector_db_type == "chroma":
            # Check if vector store exists
            persist_directory = self.vector_db_dir / f"{vector_store_id}_chroma"
            
            if not persist_directory.exists():
                raise FileNotFoundError(f"Vector store not found at {persist_directory}")
            
            # Load Chroma vector store
            self.vector_store = Chroma(
                persist_directory=str(persist_directory),
                embedding_function=self.embeddings
            )
            
        else:
            raise ValueError(f"Unsupported vector store type: {self.vector_db_type}")
        
        logger.info(f"Vector store loaded from {self.vector_db_dir}")
    
    @timed(logger=logger)
    @log_exceptions(logger=logger)
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search the vector store for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_dict: Dictionary of metadata filters
            
        Returns:
            List of relevant documents
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Please create or load a vector store first.")
        
        # Perform similarity search
        results = self.vector_store.similarity_search(
            query,
            k=top_k,
            filter=filter_dict
        )
        
        return results
    
    @timed(logger=logger)
    @log_exceptions(logger=logger)
    def ask(
        self,
        question: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        prompt_template: Optional[str] = None,
        streaming: bool = False
    ) -> Dict[str, Any]:
        """
        Ask a question using the RAG system.
        
        Args:
            question: Question to ask
            top_k: Number of context documents to retrieve
            filter_dict: Dictionary of metadata filters
            prompt_template: Custom prompt template
            streaming: Whether to stream the response
            
        Returns:
            Dictionary with answer and source documents
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Please create or load a vector store first.")
        
        # Create retriever
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": top_k, "filter": filter_dict}
        )
        
        # Create callback manager for streaming if enabled
        callbacks = None
        if streaming:
            callbacks = CallbackManager([StreamingStdOutCallbackHandler()])
        
        # Create custom prompt if provided
        if prompt_template:
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
        else:
            prompt = self.qa_prompt
        
        # Create QA chain
        qa_chain = load_qa_chain(
            self.llm,
            chain_type="stuff",
            prompt=prompt,
            verbose=False
        )
        
        # Create retrieval QA chain
        qa = RetrievalQA(
            retriever=retriever,
            combine_documents_chain=qa_chain,
            return_source_documents=True,
            callbacks=callbacks
        )
        
        # Run the chain
        start_time = time.time()
        result = qa({"query": question})
        processing_time = time.time() - start_time
        
        # Format the result
        answer = {
            "question": question,
            "answer": result["result"],
            "source_documents": result["source_documents"],
            "processing_time_seconds": processing_time
        }
        
        return answer
    
    @timed(logger=logger)
    @log_exceptions(logger=logger)
    def process_and_index_documents(
        self,
        source_paths: Union[str, List[str], Path, List[Path]],
        vector_store_id: str = "default",
        use_docling: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Process and index documents in a single operation.
        
        Args:
            source_paths: Path(s) to documents or directories
            vector_store_id: Identifier for the vector store
            use_docling: Whether to use Docling for document loading
            
        Returns:
            Processing and indexing result information
        """
        start_time = time.time()
        
        # Load documents
        documents = self.load_documents(source_paths, use_docling)
        
        # Create vector store
        self.create_vector_store(documents, vector_store_id)
        
        processing_time = time.time() - start_time
        
        result = {
            "source_paths": source_paths,
            "document_count": len(documents),
            "vector_store_id": vector_store_id,
            "vector_store_type": self.vector_db_type,
            "embedding_model": self.embedding_model_name,
            "processing_time_seconds": processing_time,
            "use_docling": use_docling if use_docling is not None else self.use_docling
        }
        
        return result