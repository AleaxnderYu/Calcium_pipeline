# Citation System Documentation

**Last Updated:** October 24, 2025

## Overview

The Calcium Imaging Agentic AI System now includes an automatic citation system that:
- Tracks all papers retrieved from RAG (Retrieval-Augmented Generation)
- Generates properly formatted reference lists with clickable hyperlinks
- Encourages the LLM to mention sources naturally in generated text
- Displays all cited papers even if not explicitly mentioned in the answer

## Key Features

### 1. Automatic Reference Tracking
- Every RAG retrieval stores complete metadata about source papers
- Includes paper filename, full file path, and page/section numbers
- Tracks which papers were used across multiple tool calls
- Deduplicates sources across the entire workflow

### 2. Hyperlinked Citations
- Reference lists include `file://` URLs to PDF sources
- Clicking a reference opens the PDF directly in your default PDF viewer
- Works in markdown viewers, web browsers, and compatible text editors

### 3. Complete Paper List
- Shows ALL papers retrieved by RAG, not just those mentioned in text
- Ensures transparency about what knowledge was used
- Helps users verify information sources

## Architecture

### Data Flow

```
User Query
    ↓
RAG Retrieval (tools/rag_system_enhanced.py)
    ↓ Returns RAGContext with:
    ├─ chunks: [text segments]
    ├─ sources: [paper filenames]
    ├─ pages: [page/section numbers]
    ├─ full_paths: [absolute file paths]
    └─ scores: [similarity scores]
    ↓
Orchestrator (core/orchestrator.py)
    ↓ Stores RAGContext in tool_outputs
    ↓
Report Generator (core/report_generator.py)
    ↓ LLM synthesizes answer with natural source mentions
    ↓
Format Output Node (graph/workflow.py)
    ↓ Collects all RAGContexts from tool_outputs
    ↓ Merges into comprehensive citation list
    ↓
Citation Formatter (core/citation_formatter.py)
    ↓ Generates formatted reference list
    ↓
Final Output
    ├─ Answer (with inline source mentions)
    └─ References section (with hyperlinks)
```

### Core Components

#### 1. RAGContext Data Model (`core/data_models.py`)

Enhanced to include full citation metadata:

```python
@dataclass
class RAGContext:
    chunks: List[str]           # Text content
    sources: List[str]          # Filenames (e.g., "paper.pdf")
    scores: List[float]         # Similarity scores
    pages: List[int]            # Page/section numbers
    full_paths: List[str]       # Absolute paths for hyperlinks
    metadata: Dict[str, Any]    # Additional metadata

    def get_unique_sources(self) -> List[Dict[str, Any]]:
        """Get deduplicated list of source papers with page ranges."""
```

#### 2. Enhanced RAG System (`tools/rag_system_enhanced.py`)

Updated to capture complete metadata:

```python
def retrieve(self, query: str, ...) -> RAGContext:
    # Returns RAGContext with all fields populated
    return RAGContext(
        chunks=chunks,
        sources=sources,          # Filenames
        scores=scores,
        pages=pages,              # Page numbers
        full_paths=full_paths     # Full file paths
    )
```

#### 3. Citation Formatter (`core/citation_formatter.py`)

**New utility class** for formatting citations:

```python
class CitationFormatter:
    @staticmethod
    def format_citation_list(
        rag_context: RAGContext,
        format: str = "markdown"  # "markdown", "html", or "plain"
    ) -> str:
        """Generate formatted reference list with hyperlinks."""

    @staticmethod
    def format_rag_summary(rag_context: RAGContext) -> str:
        """Generate summary for logging (e.g., 'Retrieved 3 chunks from 2 papers')."""

    @staticmethod
    def _group_page_ranges(pages: List[int]) -> str:
        """Group consecutive pages: [1,2,3,5,7,8,9] → '1-3, 5, 7-9'"""
```

#### 4. Orchestrator Update (`core/orchestrator.py`)

Modified to store full RAGContext:

```python
def _execute_rag(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
    rag_context = rag_system.retrieve(...)

    return {
        "chunks": rag_context.chunks,
        "sources": rag_context.sources,
        # ... other fields ...
        "rag_context": rag_context  # ← Added for citations
    }
```

#### 5. Format Output Node (`graph/workflow.py`)

Enhanced to collect and format all citations:

```python
def format_output_node(state: PipelineState) -> PipelineState:
    # Collect all RAG contexts from tool outputs
    rag_contexts = []
    for tool_id, output in tool_outputs.items():
        if "rag_context" in output:
            rag_contexts.append(output["rag_context"])

    # Merge all contexts
    merged_rag_context = RAGContext(...)

    # Generate citation list
    citation_list = CitationFormatter.format_citation_list(
        merged_rag_context, format="markdown"
    )

    # Append to summary
    summary += f"\n\n{citation_list}\n"
```

#### 6. Report Generator Update (`core/report_generator.py`)

Updated system prompt to encourage natural source mentions:

```python
system_prompt = """
Guidelines:
- When using information from papers, mention the source naturally
  (e.g., "According to [Paper.pdf]...")
- A complete reference list with hyperlinks will be automatically appended
- Focus on accuracy and clarity over formality
"""
```

## Output Format

### Example Output

**User Question:** "How is baseline fluorescence calculated in calcium imaging?"

**System Response:**

```markdown
## Answer

Baseline fluorescence in calcium imaging is typically calculated using the
average or median fluorescence intensity over a defined time window before
stimulation. According to [calcium_signaling.pdf], a common approach is to
use a rolling average with a window of 10-30 seconds prior to the stimulus.

The baseline can be calculated using several methods:

1. **Simple average**: Mean intensity over pre-stimulus period
2. **Percentile-based**: 10th or 25th percentile to avoid transient contamination
3. **Rolling window**: Adaptive baseline that accounts for drift

[baseline_calculation.pdf] recommends using the 10th percentile method for
noisy data or data with spontaneous activity, as it provides more robust
estimates compared to simple averaging.

For ΔF/F₀ calculations, F₀ represents this baseline fluorescence:

ΔF/F₀ = (F - F₀) / F₀

where F is the current fluorescence and F₀ is the baseline.

## Analysis Results

{mean_baseline: 0.23, method: "10th_percentile", window_size: 30}

## References

1. [calcium_signaling.pdf](file:///home/user/Calcium_pipeline/data/papers/calcium_signaling.pdf)
2. [baseline_calculation.pdf](file:///home/user/Calcium_pipeline/data/papers/baseline_calculation.pdf) (pp. 5, 12)
3. [transient_detection.pdf](file:///home/user/Calcium_pipeline/data/papers/transient_detection.pdf) (p. 3)
```

### Key Features of Output

1. **Natural inline mentions**: Papers mentioned naturally in text (e.g., "According to [paper.pdf]...")
2. **Complete reference list**: ALL retrieved papers listed, even if not explicitly mentioned
3. **Clickable hyperlinks**: Each reference is a clickable `file://` URL
4. **Page numbers**: Shows specific pages/sections when available
5. **Page ranges**: Consecutive pages grouped (e.g., "pp. 1-3, 5, 7-9")

## Format Variations

The citation formatter supports multiple output formats:

### Markdown (default)
```markdown
## References

1. [paper1.pdf](file:///path/to/paper1.pdf) (pp. 5-7)
2. [paper2.pdf](file:///path/to/paper2.pdf) (p. 12)
```

### HTML
```html
<h2>References</h2>
<ol>
  <li><a href="file:///path/to/paper1.pdf" target="_blank">paper1.pdf</a> (pp. 5-7)</li>
  <li><a href="file:///path/to/paper2.pdf" target="_blank">paper2.pdf</a> (p. 12)</li>
</ol>
```

### Plain Text
```
References:

[1] paper1.pdf (pp. 5-7)
    Path: /home/user/Calcium_pipeline/data/papers/paper1.pdf
[2] paper2.pdf (p. 12)
    Path: /home/user/Calcium_pipeline/data/papers/paper2.pdf
```

## Configuration

The citation system uses existing RAG configuration:

```python
# config.py
TOP_K_CHUNKS = 3  # Number of chunks to retrieve (affects citation count)
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"  # Location of PDF sources
```

## Usage Examples

### In Python Code

```python
from core.citation_formatter import CitationFormatter
from core.data_models import RAGContext

# Create RAG context
rag_context = RAGContext(
    chunks=["content1", "content2"],
    sources=["paper1.pdf", "paper2.pdf"],
    scores=[0.9, 0.85],
    pages=[5, 12],
    full_paths=[
        "/data/papers/paper1.pdf",
        "/data/papers/paper2.pdf"
    ]
)

# Format citations
citations = CitationFormatter.format_citation_list(rag_context, format="markdown")
print(citations)

# Get summary for logging
summary = CitationFormatter.format_rag_summary(rag_context)
print(summary)  # "Retrieved 2 chunks from 2 papers: paper1.pdf, paper2.pdf"
```

### In Workflow

Citations are automatically added to all analysis results. No manual intervention needed!

```bash
# CLI
python main.py --request "Explain baseline calculation" --images ./data/images

# Output includes automatic citation list at the end
```

## Logging

The system logs citation information at multiple levels:

```
[INFO] [RAG] Retrieved 3 chunks from 2 papers: paper1.pdf (p.5), paper2.pdf (pp.12-14)
[INFO] [ORCHESTRATOR] ✓ rag_tool_1 completed in 0.85s
[INFO] [NODE: format_output] Adding citations from 1 RAG context(s)
[INFO] [NODE: format_output] Retrieved 3 chunks from 2 papers: paper1.pdf, paper2.pdf
[INFO] [NODE: format_output] ✓ Output formatted with citations
```

## Benefits

### For Users

1. **Transparency**: See exactly which papers informed the answer
2. **Verification**: Click links to read original sources
3. **Completeness**: All retrieved papers shown, not just mentioned ones
4. **Traceability**: Understand the knowledge base used

### For Researchers

1. **Reproducibility**: Citations enable verification of results
2. **Literature Tracking**: Automatically tracks paper usage
3. **Knowledge Gaps**: Identify when system lacks relevant papers
4. **Citation Analysis**: See which papers are most frequently used

### For Developers

1. **Debugging**: Trace which papers influenced which answers
2. **Quality Assurance**: Verify RAG retrieval is working correctly
3. **Performance Metrics**: Track paper usage over time
4. **System Transparency**: Clear audit trail of information sources

## Troubleshooting

### Issue: No citations appearing

**Cause**: RAG tool not being called or RAG context not stored

**Solution**:
1. Check tool plan includes RAG retrieval:
   ```bash
   grep "ToolType.RAG" logs/pipeline.log
   ```
2. Verify RAG context in tool_outputs:
   ```python
   logger.info(f"Tool outputs: {tool_outputs}")
   ```

### Issue: Hyperlinks not working

**Cause**: PDF viewer not associated with `file://` protocol

**Solution**:
- **Linux**: Set default PDF handler in Desktop Environment settings
- **macOS**: Links should work by default
- **Windows**: Associate `.pdf` extension with PDF reader
- **Alternative**: Copy path from plain text format and open manually

### Issue: Page numbers showing as 0

**Cause**: Section-based retrieval (not page-based)

**Explanation**: The enhanced RAG system uses semantic sections instead of fixed pages. Page 0 indicates "full paper" or "section-based" retrieval. This is expected behavior.

### Issue: Too many citations (cluttered output)

**Cause**: High `TOP_K_CHUNKS` value retrieving many papers

**Solution**:
```python
# config.py
TOP_K_CHUNKS = 3  # Reduce from default (currently 3)
```

Or request-specific:
```python
# In planner or user input
rag_tool.inputs["top_k"] = 2
```

### Issue: Missing papers in citation list

**Cause**: Papers mentioned in text but not retrieved by RAG

**Explanation**: The citation list only includes papers actually retrieved by RAG. If the LLM mentions a paper not in the RAG results, it won't appear in the reference list.

**Solution**: This is expected behavior. The citation list reflects actual sources used, not hallucinated references.

## Future Enhancements

### Planned Features

1. **Inline Citation Markers**
   - Automatically insert `[1]`, `[2]` markers in text
   - Match text segments to source chunks
   - NLP-based citation insertion

2. **Citation Styles**
   - Support for APA, MLA, Chicago styles
   - Configurable citation format
   - Author/year extraction from filenames

3. **Citation Analytics**
   - Track most-cited papers
   - Citation frequency dashboard
   - Paper usage heatmaps

4. **PDF Page Links**
   - Deep links to specific PDF pages (e.g., `file://path.pdf#page=5`)
   - Requires PDF viewer support

5. **BibTeX Export**
   - Generate `.bib` files from citations
   - Integration with reference managers (Zotero, Mendeley)

6. **Citation Validation**
   - Verify all cited papers exist
   - Check for broken file paths
   - Warn about missing PDFs

## Technical Details

### Page Number Conventions

- **Page 0**: Indicates full paper or section-based retrieval (not page-specific)
- **Page > 0**: Actual page number from PDF (1-indexed)
- **Multiple pages**: Automatically grouped into ranges (e.g., "pp. 1-3, 5, 7-9")

### File Path Handling

```python
# Full path stored for hyperlinks
full_path = "/home/user/Calcium_pipeline/data/papers/paper.pdf"

# Converted to file:// URL
file_url = f"file://{Path(full_path).resolve()}"
# Result: "file:///home/user/Calcium_pipeline/data/papers/paper.pdf"
```

### Page Range Grouping Algorithm

```python
def _group_page_ranges(pages: List[int]) -> str:
    """
    Groups consecutive pages into ranges.

    Example:
        [1, 2, 3, 5, 7, 8, 9] → "1-3, 5, 7-9"
    """
    pages = sorted(set(p for p in pages if p > 0))  # Remove page 0, deduplicate
    ranges = []
    start = end = pages[0]

    for page in pages[1:]:
        if page == end + 1:
            end = page  # Extend range
        else:
            # Save current range, start new one
            ranges.append(f"{start}-{end}" if start != end else str(start))
            start = end = page

    # Add final range
    ranges.append(f"{start}-{end}" if start != end else str(start))
    return ", ".join(ranges)
```

## Integration with Other Tools

### Streamlit UI (`app.py`)
- Citations display automatically in chat interface
- Hyperlinks are clickable in markdown viewer

### FastAPI (`api_backend.py`)
- Citations included in JSON responses
- `metadata.papers_cited` field added
- Full reference list in `summary` field

### CLI (`main.py`)
- Citations print to console
- File paths copyable for manual opening

## Version History

| Version | Date | Changes |
|---------|------|---------|
| **1.0** | Oct 24, 2025 | Initial citation system implementation |
|         |              | - Added RAGContext page/path fields |
|         |              | - Created CitationFormatter utility |
|         |              | - Updated format_output_node |
|         |              | - Enhanced report generator prompts |

## Related Documentation

- [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) - Overall system design
- [ENHANCED_RAG.md](ENHANCED_RAG.md) - Enhanced RAG system details
- [core/citation_formatter.py](core/citation_formatter.py) - Citation formatter source code
- [core/data_models.py](core/data_models.py) - RAGContext data model

---

**Questions or Issues?**
- Check logs for citation-related messages
- Test citation formatter directly (see "Usage Examples")
- Verify RAG system is retrieving papers correctly
