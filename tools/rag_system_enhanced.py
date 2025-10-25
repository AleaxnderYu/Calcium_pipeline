"""
Enhanced RAG System with Docling section-based chunking and hybrid retrieval.

Features:
- Section-based chunking (preserves paper structure)
- Hybrid approach: Multiple sections from same paper → return full paper
- Dynamic top-k controlled by orchestrator
- Better context preservation
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
import time

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

from core.data_models import RAGContext
import config

logger = logging.getLogger(__name__)


class EnhancedRAGSystem:
    """
    Enhanced RAG system with section-based chunking and intelligent retrieval.

    Key improvements:
    1. Section-based chunking using Docling's structure detection
    2. Hybrid retrieval: Multiple sections from same paper → full paper
    3. Dynamic top-k parameter
    4. Preserves paper structure and context
    """

    def __init__(self, papers_dir: str = None, vector_db_path: str = None):
        """
        Initialize enhanced RAG system.

        Args:
            papers_dir: Directory containing paper PDFs
            vector_db_path: Path to ChromaDB vector database
        """
        self.papers_dir = Path(papers_dir) if papers_dir else config.PAPERS_DIR
        self.vector_db_path = Path(vector_db_path) if vector_db_path else config.VECTOR_DB_PATH

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            openai_api_key=config.OPENAI_API_KEY
        )

        # Cache for full paper content (for hybrid retrieval)
        self.paper_cache: Dict[str, str] = {}

        # Load or build vector database
        self.vectorstore = self._initialize_vectorstore()

    def _initialize_vectorstore(self) -> Chroma:
        """Load existing or build new vector database."""
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

        logger.info("Building new vector database with section-based chunking")
        return self._build_vectorstore_with_sections()

    def _extract_sections_from_pdf(self, pdf_path: Path) -> List[Document]:
        """
        Extract sections from PDF using Docling's structure detection.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of Documents, one per section
        """
        if not DOCLING_AVAILABLE:
            logger.warning(f"Docling not available for {pdf_path.name}, using fallback chunking")
            return self._fallback_chunking(pdf_path)

        try:
            # Configure Docling with section detection and OCR
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = config.DOCLING_ENABLE_OCR

            # Use default OCR (RapidOCR - fastest option)
            converter = DocumentConverter(
                format_options={PdfFormatOption: pipeline_options}
            )

            # Convert PDF to structured document
            result = converter.convert(str(pdf_path))
            doc_content = result.document.export_to_markdown()

            # Split by markdown headers to get sections
            sections = []
            current_section = []
            current_header = "Introduction"  # Default section

            for line in doc_content.split('\n'):
                # Detect section headers (# or ##)
                if line.startswith('# ') or line.startswith('## '):
                    # Save previous section
                    if current_section:
                        section_text = '\n'.join(current_section).strip()
                        if section_text:  # Only add non-empty sections
                            sections.append(Document(
                                page_content=section_text,
                                metadata={
                                    "source": pdf_path.name,
                                    "section": current_header,
                                    "type": "section"
                                }
                            ))
                    # Start new section
                    current_header = line.lstrip('#').strip()
                    current_section = [line]
                else:
                    current_section.append(line)

            # Add last section
            if current_section:
                section_text = '\n'.join(current_section).strip()
                if section_text:
                    sections.append(Document(
                        page_content=section_text,
                        metadata={
                            "source": pdf_path.name,
                            "section": current_header,
                            "type": "section"
                        }
                    ))

            # Clean and cache full paper content for hybrid retrieval
            # Remove consecutive duplicate lines (common in Docling extractions)
            lines = doc_content.split('\n')
            cleaned_lines = []
            prev_line = None
            for line in lines:
                stripped = line.strip()
                # Skip consecutive duplicate lines
                if stripped != prev_line or not stripped:
                    cleaned_lines.append(line)
                prev_line = stripped if stripped else prev_line

            cleaned_content = '\n'.join(cleaned_lines)
            self.paper_cache[pdf_path.name] = cleaned_content

            logger.info(f"Extracted {len(sections)} sections from {pdf_path.name}")
            return sections

        except Exception as e:
            logger.error(f"Docling extraction failed for {pdf_path.name}: {e}")
            return self._fallback_chunking(pdf_path)

    def _fallback_chunking(self, pdf_path: Path) -> List[Document]:
        """Fallback to character-based chunking if Docling fails."""
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()

        # Cache full content
        full_content = "\n\n".join([doc.page_content for doc in documents])
        self.paper_cache[pdf_path.name] = full_content

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(documents)

        # Update metadata
        for split in splits:
            split.metadata["type"] = "chunk"
            split.metadata["section"] = "unknown"

        return splits

    def _build_vectorstore_with_sections(self) -> Chroma:
        """
        Build vector database using section-based chunking.

        Returns:
            Chroma vectorstore instance
        """
        if not self.papers_dir.exists():
            raise ValueError(f"Papers directory does not exist: {self.papers_dir}")

        all_sections = []

        # Process all PDFs
        pdf_files = list(self.papers_dir.glob("*.pdf"))
        logger.info(f"Processing {len(pdf_files)} PDF files with section-based chunking...")

        for pdf_path in pdf_files:
            sections = self._extract_sections_from_pdf(pdf_path)
            all_sections.extend(sections)

        # Also load text files (if any)
        txt_files = list(self.papers_dir.glob("*.txt"))
        if txt_files:
            txt_loader = DirectoryLoader(
                str(self.papers_dir),
                glob="*.txt",
                loader_cls=TextLoader
            )
            txt_documents = txt_loader.load()
            all_sections.extend(txt_documents)

        if not all_sections:
            raise ValueError(f"No sections extracted from {self.papers_dir}")

        logger.info(f"Total sections/chunks: {len(all_sections)}")

        # Create vector database with batched embeddings
        # OpenAI has a limit of 300K tokens per request
        # Batch size of 100 documents is safe (~150K tokens typically)
        batch_size = 100
        logger.info(f"Creating embeddings in batches of {batch_size}...")

        # Create empty vectorstore first
        vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=str(self.vector_db_path)
        )

        # Add documents in batches
        for i in range(0, len(all_sections), batch_size):
            batch = all_sections[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(all_sections) + batch_size - 1)//batch_size} ({len(batch)} sections)...")
            vectorstore.add_documents(batch)

        logger.info(f"Vector database created at {self.vector_db_path} with {len(all_sections)} sections")
        return vectorstore

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        return_full_papers: bool = True,
        multi_section_threshold: int = 2
    ) -> RAGContext:
        """
        Retrieve relevant sections/papers with hybrid approach.

        Args:
            query: Search query
            top_k: Number of sections to retrieve (default from config)
            return_full_papers: If True, return full paper when multiple sections match
            multi_section_threshold: Min sections from same paper to return full paper

        Returns:
            RAGContext with chunks (sections or full papers)
        """
        start_time = time.time()
        logger.info(f"Retrieving context for query: '{query}'")

        # Use provided top_k or default
        k = top_k if top_k is not None else config.TOP_K_CHUNKS

        # Retrieve top sections
        results_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)

        # Group by source paper
        paper_sections = defaultdict(list)
        for doc, score in results_with_scores:
            source = doc.metadata.get("source", "unknown")
            paper_sections[source].append({
                "content": doc.page_content,
                "section": doc.metadata.get("section", "unknown"),
                "score": float(score),
                "type": doc.metadata.get("type", "section")
            })

        # Apply hybrid approach
        chunks = []
        sources = []
        scores = []
        pages = []  # Page numbers for citations
        full_paths = []  # Full paths for hyperlinks

        for paper, sections in paper_sections.items():
            # Construct full path for hyperlink
            paper_full_path = str(config.PAPERS_DIR / paper)

            # If multiple sections from same paper, return full paper
            if return_full_papers and len(sections) >= multi_section_threshold:
                logger.info(f"Found {len(sections)} sections from {paper}, returning full paper")

                if paper in self.paper_cache:
                    chunks.append(self.paper_cache[paper])
                    sources.append(f"{paper} (full paper)")
                    # Use average score of sections
                    avg_score = sum(s["score"] for s in sections) / len(sections)
                    scores.append(avg_score)
                    pages.append(0)  # Indicates full paper
                    full_paths.append(paper_full_path)
                else:
                    # Fallback: concatenate sections
                    full_content = "\n\n---\n\n".join([
                        f"## {s['section']}\n{s['content']}"
                        for s in sections
                    ])
                    chunks.append(full_content)
                    sources.append(f"{paper} ({len(sections)} sections)")
                    avg_score = sum(s["score"] for s in sections) / len(sections)
                    scores.append(avg_score)
                    pages.append(0)  # Indicates multiple sections
                    full_paths.append(paper_full_path)
            else:
                # Return individual sections
                for section_data in sections:
                    chunks.append(section_data["content"])
                    sources.append(f"{paper} - {section_data['section']}")
                    scores.append(section_data["score"])
                    pages.append(0)  # Section-based, not page-based
                    full_paths.append(paper_full_path)

        retrieval_time = time.time() - start_time

        metadata = {
            'query': query,
            'retrieval_time': retrieval_time,
            'n_chunks': len(chunks),
            'papers_retrieved': len(paper_sections),
            'hybrid_used': return_full_papers,
            'multi_section_threshold': multi_section_threshold
        }

        logger.info(
            f"Retrieved {len(chunks)} chunks from {len(paper_sections)} papers "
            f"({', '.join(paper_sections.keys())})"
        )

        return RAGContext(
            chunks=chunks,
            sources=sources,
            scores=scores,
            metadata=metadata,
            pages=pages,
            full_paths=full_paths
        )

    def rebuild(self):
        """Force rebuild of vector database."""
        logger.info("Forcing rebuild of vector database with section-based chunking")
        self.paper_cache.clear()
        self.vectorstore = self._build_vectorstore_with_sections()


# Singleton instance
_enhanced_rag = None


def get_enhanced_rag_system() -> EnhancedRAGSystem:
    """Get or create EnhancedRAGSystem singleton."""
    global _enhanced_rag
    if _enhanced_rag is None:
        _enhanced_rag = EnhancedRAGSystem()
    return _enhanced_rag
