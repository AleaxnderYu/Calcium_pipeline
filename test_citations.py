#!/usr/bin/env python3
"""
Test script to verify citation system is working correctly.
"""

from core.data_models import RAGContext
from core.citation_formatter import CitationFormatter
import config

# Create a test RAG context
test_context = RAGContext(
    chunks=[
        "Baseline fluorescence is calculated...",
        "Calcium transients can be detected...",
        "The F0 value represents..."
    ],
    sources=[
        "baseline_calculation.pdf",
        "calcium_signaling.pdf",
        "baseline_calculation.pdf"
    ],
    scores=[0.92, 0.88, 0.85],
    pages=[5, 12, 7],
    full_paths=[
        str(config.PAPERS_DIR / "baseline_calculation.pdf"),
        str(config.PAPERS_DIR / "calcium_signaling.pdf"),
        str(config.PAPERS_DIR / "baseline_calculation.pdf")
    ]
)

print("=" * 80)
print("CITATION SYSTEM TEST")
print("=" * 80)
print()

print("1. RAG Context Summary:")
print(CitationFormatter.format_rag_summary(test_context))
print()

print("2. Unique Sources:")
unique = test_context.get_unique_sources()
for source in unique:
    print(f"   - {source['filename']}: pages {source['pages']}")
print()

print("3. Markdown Citation List:")
print(CitationFormatter.format_citation_list(test_context, format="markdown"))
print()

print("4. HTML Citation List:")
print(CitationFormatter.format_citation_list(test_context, format="html"))
print()

print("5. Plain Text Citation List:")
print(CitationFormatter.format_citation_list(test_context, format="plain"))
print()

print("=" * 80)
print("TEST COMPLETE")
print("=" * 80)
