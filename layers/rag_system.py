"""
RAG System: Retrieve relevant biological methods from papers using LangChain.
"""

import logging
from pathlib import Path
from typing import List
import time

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from core.data_models import RAGContext
import config

logger = logging.getLogger(__name__)


class RAGSystem:
    """RAG system for retrieving calcium imaging analysis methods from papers."""

    def __init__(self, papers_dir: str = None, vector_db_path: str = None):
        """
        Initialize RAG system with paper documents.

        Args:
            papers_dir: Directory containing paper text files
            vector_db_path: Path to ChromaDB vector database
        """
        self.papers_dir = Path(papers_dir) if papers_dir else config.PAPERS_DIR
        self.vector_db_path = Path(vector_db_path) if vector_db_path else config.VECTOR_DB_PATH

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            openai_api_key=config.OPENAI_API_KEY
        )

        # Load or build vector database
        self.vectorstore = self._initialize_vectorstore()
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": config.TOP_K_CHUNKS}
        )

    def _initialize_vectorstore(self) -> Chroma:
        """
        Load existing vector database or build new one from papers.

        Returns:
            Chroma vectorstore instance
        """
        # Check if vector database already exists
        # ChromaDB creates a chroma.sqlite3 file in the persist directory
        db_file = self.vector_db_path / "chroma.sqlite3"

        if db_file.exists():
            logger.info(f"Loading existing vector database from {self.vector_db_path}")
            try:
                vectorstore = Chroma(
                    persist_directory=str(self.vector_db_path),
                    embedding_function=self.embeddings
                )
                return vectorstore
            except Exception as e:
                logger.warning(f"Failed to load existing database: {e}. Rebuilding...")

        # Build new vector database
        logger.info("Building new vector database from papers")
        return self._build_vectorstore()

    def _build_vectorstore(self) -> Chroma:
        """
        Build vector database from papers directory.

        Returns:
            Chroma vectorstore instance
        """
        # Validate papers directory
        if not self.papers_dir.exists():
            raise ValueError(f"Papers directory does not exist: {self.papers_dir}")

        # Load documents
        loader = DirectoryLoader(
            str(self.papers_dir),
            glob="*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()

        if not documents:
            raise ValueError(f"No .txt files found in {self.papers_dir}")

        logger.info(f"Loaded {len(documents)} papers")

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
        )
        splits = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(splits)} chunks")

        # Create vector database
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=str(self.vector_db_path)
        )

        logger.info(f"Vector database created at {self.vector_db_path}")
        return vectorstore

    def retrieve(self, query: str, top_k: int = None) -> RAGContext:
        """
        Retrieve relevant chunks from papers.

        Args:
            query: Search query (user request)
            top_k: Number of chunks to retrieve (default from config)

        Returns:
            RAGContext with retrieved chunks
        """
        start_time = time.time()
        logger.info(f"Retrieving context for query: '{query}'")

        # Update retriever k if specified
        if top_k:
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": top_k}
            )

        # Retrieve documents
        docs = self.retriever.get_relevant_documents(query)

        # Extract chunks and metadata
        chunks = [doc.page_content for doc in docs]
        sources = [doc.metadata.get('source', 'unknown') for doc in docs]

        # Extract just the filename from full path
        sources = [Path(src).name for src in sources]

        # ChromaDB doesn't return scores with basic retriever
        # For similarity scores, we'd need to use similarity_search_with_score
        scores = []
        try:
            results_with_scores = self.vectorstore.similarity_search_with_score(query, k=top_k or config.TOP_K_CHUNKS)
            scores = [float(score) for _, score in results_with_scores]
        except Exception as e:
            logger.debug(f"Could not retrieve similarity scores: {e}")

        retrieval_time = time.time() - start_time

        metadata = {
            'query': query,
            'retrieval_time': retrieval_time,
            'n_chunks': len(chunks)
        }

        logger.info(f"Retrieved {len(chunks)} chunks from sources: {', '.join(set(sources))}")

        return RAGContext(
            chunks=chunks,
            sources=sources,
            scores=scores,
            metadata=metadata
        )

    def rebuild(self):
        """Force rebuild of vector database."""
        logger.info("Forcing rebuild of vector database")
        self.vectorstore = self._build_vectorstore()
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": config.TOP_K_CHUNKS}
        )
