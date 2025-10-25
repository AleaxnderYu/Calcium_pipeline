"""
Citation formatter for RAG-retrieved papers.
Generates properly formatted citations with hyperlinks for PDF sources.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import quote
from core.data_models import RAGContext
import config

logger = logging.getLogger(__name__)


class CitationFormatter:
    """Formats citations from RAG context with hyperlinks."""

    @staticmethod
    def _get_paper_url(full_path: str) -> str:
        """
        Generate URL for paper (HTTP or file://).

        If API_SERVER_URL is configured, generates HTTP URL.
        Otherwise, falls back to file:// URL.

        Args:
            full_path: Absolute path to PDF file

        Returns:
            URL string for the paper
        """
        # Get filename relative to PAPERS_DIR
        try:
            paper_path = Path(full_path)
            papers_dir = Path(config.PAPERS_DIR).resolve()
            relative_path = paper_path.relative_to(papers_dir)

            # Use HTTP URL if API server is configured
            api_url = getattr(config, 'API_SERVER_URL', None)
            if api_url:
                # URL-encode the filename to handle spaces and special characters
                encoded_filename = quote(str(relative_path))
                return f"{api_url}/papers/{encoded_filename}"
            else:
                # Default to localhost:8000 (FastAPI default)
                encoded_filename = quote(str(relative_path))
                return f"http://localhost:8000/papers/{encoded_filename}"

        except ValueError:
            # If path is not relative to PAPERS_DIR, fall back to file:// URL
            logger.warning(f"Path {full_path} is not relative to PAPERS_DIR, using file:// URL")
            return f"file://{Path(full_path).resolve()}"

    @staticmethod
    def format_citation_list(rag_context: RAGContext, format: str = "markdown") -> str:
        """
        Generate formatted citation list from RAG context.

        Args:
            rag_context: RAG context with source papers
            format: Output format ("markdown", "html", or "plain")

        Returns:
            Formatted citation list as string
        """
        if not rag_context or not rag_context.chunks:
            return ""

        unique_sources = rag_context.get_unique_sources()

        if not unique_sources:
            return ""

        if format == "markdown":
            return CitationFormatter._format_markdown_citations(unique_sources)
        elif format == "html":
            return CitationFormatter._format_html_citations(unique_sources)
        else:
            return CitationFormatter._format_plain_citations(unique_sources)

    @staticmethod
    def _format_markdown_citations(sources: List[Dict[str, Any]]) -> str:
        """Format citations as numbered markdown list with hyperlinks."""
        lines = ["## References\n"]

        for i, source_info in enumerate(sources, 1):
            filename = source_info['filename']
            full_path = source_info['full_path']

            # Create HTTP or file:// URL for PDF
            if full_path:
                # Use numbered list so [1], [2], [3] in answer matches this list
                paper_url = CitationFormatter._get_paper_url(full_path)
                citation = f"{i}. [{filename}]({paper_url})"
            else:
                citation = f"{i}. {filename}"

            lines.append(citation)

        return "\n".join(lines)

    @staticmethod
    def _format_html_citations(sources: List[Dict[str, Any]]) -> str:
        """Format citations as HTML with clickable links."""
        lines = ["<h2>References</h2>", "<ol>"]

        for source_info in sources:
            filename = source_info['filename']
            full_path = source_info['full_path']
            pages = source_info['pages']

            if full_path:
                paper_url = CitationFormatter._get_paper_url(full_path)
                citation = f'<a href="{paper_url}" target="_blank">{filename}</a>'
            else:
                citation = filename

            # Add page information
            if pages and pages != [0]:
                if len(pages) == 1:
                    citation += f" (p. {pages[0]})"
                else:
                    page_ranges = CitationFormatter._group_page_ranges(pages)
                    citation += f" (pp. {page_ranges})"

            lines.append(f"<li>{citation}</li>")

        lines.append("</ol>")
        return "\n".join(lines)

    @staticmethod
    def _format_plain_citations(sources: List[Dict[str, Any]]) -> str:
        """Format citations as plain text."""
        lines = ["References:", ""]

        for i, source_info in enumerate(sources, 1):
            filename = source_info['filename']
            full_path = source_info['full_path']
            pages = source_info['pages']

            citation = f"[{i}] {filename}"

            if pages and pages != [0]:
                if len(pages) == 1:
                    citation += f" (p. {pages[0]})"
                else:
                    page_ranges = CitationFormatter._group_page_ranges(pages)
                    citation += f" (pp. {page_ranges})"

            if full_path:
                citation += f"\n    Path: {full_path}"

            lines.append(citation)

        return "\n".join(lines)

    @staticmethod
    def _group_page_ranges(pages: List[int]) -> str:
        """
        Group consecutive page numbers into ranges.

        Example:
            [1, 2, 3, 5, 7, 8, 9] -> "1-3, 5, 7-9"

        Args:
            pages: List of page numbers

        Returns:
            Formatted page range string
        """
        if not pages:
            return ""

        # Remove duplicates and sort
        pages = sorted(set(p for p in pages if p > 0))

        if not pages:
            return ""

        ranges = []
        start = pages[0]
        end = pages[0]

        for i in range(1, len(pages)):
            if pages[i] == end + 1:
                # Continue range
                end = pages[i]
            else:
                # End of range
                if start == end:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{end}")
                start = pages[i]
                end = pages[i]

        # Add last range
        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")

        return ", ".join(ranges)

    @staticmethod
    def format_inline_citation(source: str, page: int = None) -> str:
        """
        Format an inline citation for use in generated text.

        Args:
            source: Source filename
            page: Optional page number

        Returns:
            Inline citation string (e.g., "[Smith2020, p.5]")
        """
        # Extract author/year from filename if present
        # Common patterns: "AuthorYear.pdf", "Author_Year.pdf", etc.
        citation = Path(source).stem  # Remove .pdf extension

        if page and page > 0:
            return f"[{citation}, p.{page}]"
        else:
            return f"[{citation}]"

    @staticmethod
    def add_citations_to_text(text: str, rag_context: RAGContext) -> str:
        """
        Add citation markers to generated text based on RAG context.

        Args:
            text: Generated text
            rag_context: RAG context with sources

        Returns:
            Text with inline citations added
        """
        # This is a placeholder for future implementation
        # Could use NLP to match text segments to source chunks
        # and insert appropriate citations
        return text

    @staticmethod
    def format_rag_summary(rag_context: RAGContext) -> str:
        """
        Generate a summary of retrieved papers for logging/display.

        Args:
            rag_context: RAG context

        Returns:
            Summary string (e.g., "Retrieved 3 chunks from 2 papers: Paper1.pdf, Paper2.pdf")
        """
        if not rag_context or not rag_context.chunks:
            return "No papers retrieved"

        unique_sources = rag_context.get_unique_sources()
        n_papers = len(unique_sources)
        n_chunks = len(rag_context.chunks)

        paper_names = [s['filename'] for s in unique_sources]

        if n_papers == 1:
            return f"Retrieved {n_chunks} chunk(s) from 1 paper: {paper_names[0]}"
        elif n_papers <= 3:
            return f"Retrieved {n_chunks} chunks from {n_papers} papers: {', '.join(paper_names)}"
        else:
            return f"Retrieved {n_chunks} chunks from {n_papers} papers: {', '.join(paper_names[:3])}, and {n_papers - 3} more"


# Example usage
if __name__ == "__main__":
    # Test citation formatting
    from core.data_models import RAGContext

    # Mock RAG context
    test_context = RAGContext(
        chunks=["chunk1", "chunk2", "chunk3", "chunk4"],
        sources=["paper1.pdf", "paper2.pdf", "paper1.pdf", "paper3.pdf"],
        scores=[0.9, 0.85, 0.82, 0.78],
        pages=[5, 12, 7, 3],
        full_paths=[
            "/data/papers/paper1.pdf",
            "/data/papers/paper2.pdf",
            "/data/papers/paper1.pdf",
            "/data/papers/paper3.pdf"
        ]
    )

    formatter = CitationFormatter()

    print("=== Markdown Format ===")
    print(formatter.format_citation_list(test_context, "markdown"))
    print()

    print("=== HTML Format ===")
    print(formatter.format_citation_list(test_context, "html"))
    print()

    print("=== Plain Format ===")
    print(formatter.format_citation_list(test_context, "plain"))
    print()

    print("=== Summary ===")
    print(formatter.format_rag_summary(test_context))
