# Citation System Troubleshooting Guide

## Issue: Citations Not Appearing in Output

Based on your output, the citation system has been implemented but citations are not appearing. Here's what to check:

## What Should Appear

Your output should end with:

```markdown
## References

1. [Generation, control, and processing of cellular calcium signals.pdf](file:///path/to/paper.pdf)
2. [Calcium signalling dynamics, homeostasis and remodelling.pdf](file:///path/to/paper.pdf)
3. [Spatial and temporal aspects of cellular calcium signaling.pdf](file:///path/to/paper.pdf)
... (all 5 papers you mentioned)
```

## Why It's Not Appearing

The most likely cause is that the **RAG context is not being stored** in tool_outputs during orchestration.

## Diagnostic Steps

### Step 1: Check if RAG context is being stored

Add this after running a query:

```python
# In graph/workflow.py format_output_node, check the logs
logger.info(f"Tool outputs keys: {list(tool_outputs.keys())}")
for tool_id, output in tool_outputs.items():
    logger.info(f"{tool_id}: has_rag_context={('rag_context' in output)}")
```

**Expected output:**
```
Tool outputs keys: ['t1']
t1: has_rag_context=True
[NODE: format_output] Found RAG context in t1
[NODE: format_output] Collected 1 RAG context(s) from 1 tool outputs
```

**If you see `has_rag_context=False`:**
- The orchestrator is not storing the RAG context
- Check `core/orchestrator.py` line 208: `"rag_context": rag_context`

### Step 2: Check if citations are being generated

Look for this log message:

```
[NODE: format_output] Adding citations from 1 RAG context(s)
[NODE: format_output] Retrieved 3 chunks from 2 papers: paper1.pdf, paper2.pdf
```

**If you don't see this:**
- RAG contexts are not being collected properly
- Check the `if rag_contexts:` block in format_output_node

### Step 3: Check if citations are being streamed

Look for streaming events in the logs:

```
[STREAMING] citations: ## References...
```

**If you don't see this:**
- Citations are being added to summary but not streamed
- The streaming happens in `graph/workflow.py` line 413-414

## Quick Fix

The issue is likely that you're viewing output **before** the format_output_node runs, because the answer is streamed during report synthesis.

### Solution: Stream citations immediately after report synthesis

Update `core/report_generator.py` to stream citations right after the answer:

```python
# After streaming the synthesized answer (line ~150)
# Add citation streaming here

# Check if rag_context exists in inputs
if "rag_context" in context and context["rag_context"]:
    from core.citation_formatter import CitationFormatter

    citation_list = CitationFormatter.format_citation_list(
        context["rag_context"],
        format="markdown"
    )

    if self.streaming_reporter and citation_list:
        self.streaming_reporter.emit_event("citations", f"\n\n{citation_list}\n")
```

## Testing

Run this to verify citations work:

```bash
python test_citations.py
```

Should output:

```
## References

1. [baseline_calculation.pdf](file:///.../baseline_calculation.pdf) (pp. 5, 7)
2. [calcium_signaling.pdf](file:///.../calcium_signaling.pdf) (p. 12)
```

## Manual Verification

Run a query and check the final_output summary field:

```python
# After query completes
final_output = state["final_output"]
print(final_output.summary)
```

The summary should contain the "## References" section at the end.

## Alternative: Check Non-Streaming Output

Try using the non-streaming API endpoint or CLI to see if citations appear in the final summary:

```bash
# CLI
python main.py --request "explain calcium oscillations" --images ./data/images

# Should output final summary with citations at the end
```

## Root Cause Analysis

The system has two output phases:

1. **Streaming phase** (real-time): Report synthesis node streams answer
2. **Formatting phase** (after completion): format_output_node adds citations

If you're only seeing Phase 1 output, citations won't appear because they're added in Phase 2.

### Fix: Add citations to streaming phase

Modify the orchestrator to pass RAG contexts to the report generator so citations can be streamed immediately.

## Files to Check

1. **core/orchestrator.py:208** - Verify `rag_context` is in return dict
2. **graph/workflow.py:358-362** - Verify RAG contexts are being collected
3. **graph/workflow.py:413-414** - Verify citations are being streamed
4. **core/streaming_progress.py** - Verify "citations" events are emitted

## Expected Log Sequence

```
[ORCHESTRATOR] ✓ t1 completed in 0.85s
[NODE: format_output] Formatting results
[NODE: format_output] Found RAG context in t1
[NODE: format_output] Collected 1 RAG context(s) from 1 tool outputs
[NODE: format_output] Adding citations from 1 RAG context(s)
[NODE: format_output] Retrieved 3 chunks from 2 papers: paper1.pdf, paper2.pdf
[STREAMING] citations:
## References

1. [paper1.pdf](file:///.../paper1.pdf)
2. [paper2.pdf](file:///.../paper2.pdf)

[NODE: format_output] ✓ Output formatted with citations
```

## Contact Points

If citations still don't appear after checking above:

1. Enable DEBUG logging: `LOG_LEVEL=DEBUG` in .env
2. Check full workflow log for "rag_context" mentions
3. Verify Open WebUI is rendering markdown links (some viewers strip `file://` links)
4. Try the Streamlit UI or CLI to rule out Open WebUI rendering issues
