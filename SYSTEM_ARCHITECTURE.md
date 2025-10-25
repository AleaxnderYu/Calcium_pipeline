# Calcium Imaging Pipeline: Agentic System Architecture

## Overview

Your system is a **multi-agent LangGraph workflow** that processes calcium imaging analysis requests through intelligent routing, planning, execution, and verification.

## Core Structure Files

### 1. [graph/workflow.py](graph/workflow.py) - The Orchestrator
**What it does**: Defines the entire workflow graph structure and execution flow.

**Key components**:
- Creates the LangGraph `StateGraph` with all nodes
- Defines routing logic (informational vs analysis paths)
- Manages workflow execution and state flow

**Workflow paths**:
```
User Request
    ↓
[Router Node] ──→ Route Type?
    ├─ "informational" → [Informational Response] → END
    │                     (RAG-only answer, no code)
    │
    └─ "analysis" → [Clarifier] → [Planner] → [Multi-Step Executor]
                                                      ↓
                                            [Save Capability] → [Format Output] → END
```

### 2. [graph/state.py](graph/state.py) - The State Container
**What it does**: Defines `PipelineState` - the data structure that flows through all nodes.

**Key fields**:
- `user_request`: Original query
- `route_type`: "informational" or "analysis"
- `clarification`: Clarified request with assumptions
- `execution_plan`: Multi-step plan from planner
- `rag_context`: Retrieved scientific literature chunks
- `generated_code`: Python code to execute
- `execution_results`: Results from code execution
- `final_output`: Final formatted response
- `errors`: List of errors encountered

**Think of it as**: A shopping cart that gets filled as it moves through checkout stations.

### 3. [graph/nodes.py](graph/nodes.py) - The Workers
**What it does**: Implements each node function that processes the state.

**All nodes**:
1. `router_node` - Routes query to appropriate path
2. `informational_response_node` - RAG-only answers
3. `clarifier_node` - Clarifies ambiguous requests
4. `planner_node` - Creates execution plan
5. `multi_step_executor_node` - Executes plan steps
6. `save_capability_node` - Saves reusable code
7. `format_output_node` - Formats final response

---

## How Your Three Systems Work

### 1. RAG System ([layers/rag_system.py](layers/rag_system.py))

**Purpose**: Retrieve relevant scientific methods from calcium imaging papers.

**How it works**:
```python
# 1. Initialization (one-time)
rag = RAGSystem()
# - Loads 91 PDF papers from data/papers/
# - Chunks into 500-char pieces with 100-char overlap
# - Embeds using OpenAI text-embedding-3-small
# - Stores in ChromaDB vector database

# 2. Retrieval (per query)
rag_context = rag.retrieve("How to detect calcium transients?")
# - Embeds query
# - Searches vector DB for top 5 most similar chunks
# - Returns chunks with sources and scores
```

**When it's used**:
- **Informational path**: Generates RAG-only answer (no code)
- **Analysis path**: Provides scientific context to code generator

**Key methods**:
- `retrieve(query)`: Get relevant chunks
- `rebuild()`: Force rebuild vector DB (use after adding papers)

**Current setup**:
- 91 PDF papers
- Chunk size: 500 characters
- Overlap: 100 characters
- Top-K: 5 chunks retrieved per query
- Embedding model: `text-embedding-3-small`

---

### 2. Planner System ([core/planner.py](core/planner.py))

**Purpose**: Decompose complex requests into multi-step execution plans.

**How it works**:
```python
planner = ExecutionPlanner()
plan = planner.create_plan(
    clarified_request="Segment cells and count them",
    assumptions=["Use watershed segmentation", "Count connected components"]
)

# Returns ExecutionPlan with steps:
# Step 1: "Segment cells using watershed" (action: segment_cells)
# Step 2: "Count segmented cells" (action: calculate_statistics, depends_on: [step_1])
```

**Execution modes**:
1. **Sequential**: Steps run in order (step 2 needs step 1's output)
   - Example: Segment → Count → Plot

2. **Parallel**: Steps run simultaneously (independent)
   - Example: Calculate mean AND std at the same time

3. **DAG** (Directed Acyclic Graph): Complex dependencies
   - Example: Segment → [Measure Size, Measure Intensity] → Plot Both

**When it's used**:
- After clarification
- Before code generation
- Only in "analysis" path

**Key features**:
- Uses GPT-3.5-turbo (lightweight, fast)
- Prefers single-step plans when possible
- Fallback to single-step if planning fails

**Available actions**:
- `preprocess_images`: Filtering, normalization
- `segment_cells`: Watershed, thresholding
- `detect_transients`: Find calcium events
- `calculate_statistics`: Mean, std, correlation
- `measure_properties`: Cell size, intensity
- `plot_results`: Visualization
- `extract_timeseries`: Get signals over time

---

### 3. Verifier System ([core/verifier.py](core/verifier.py))

**Purpose**: Validate execution results for sanity and correctness.

**How it works**:
```python
verifier = ResultVerifier()
verification = verifier.verify(
    step=execution_step,
    result=execution_result,
    data_context={"image_size": (512, 512), "pixel_range": [0, 255]}
)

# Returns VerificationResult:
# - passed: True/False
# - confidence: 0.0 to 1.0
# - issues: ["List of problems"]
# - suggestions: ["How to fix"]
# - should_retry: True/False
```

**What it checks**:
1. **Data type correctness**: Counts should be integers, not negative
2. **Value reasonableness**: Cell count shouldn't be 10,000 in small image
3. **Output completeness**: Did it produce expected fields?
4. **Physical plausibility**: Intensity shouldn't exceed pixel range

**Examples of what it catches**:

❌ **Bad**: Mean intensity = -50 (pixels are [0, 255])
```json
{
  "passed": false,
  "issues": ["Mean intensity is negative"],
  "suggestions": ["Check normalization"],
  "should_retry": true
}
```

✅ **Good**: Cell count = 42 in 512x512 image
```json
{
  "passed": true,
  "confidence": 0.95,
  "issues": [],
  "should_retry": false
}
```

**When it's used**:
- After each execution step completes
- Before moving to next step
- Can trigger retry if verification fails

**Key features**:
- Uses GPT-3.5-turbo (lightweight)
- Semantic verification (understands context)
- Conservative (assumes pass on error)

---

## Complete Workflow Walkthrough

### Scenario: "Count cells in the image"

**Step 1: Router** ([core/router.py](core/router.py))
```
Input: "Count cells in the image"
Process: Classify query type
Output: route_type = "analysis" (needs code execution)
```

**Step 2: Clarifier** ([core/clarifier.py](core/clarifier.py))
```
Input: "Count cells in the image"
Process: Make assumptions explicit
Output: ClarificationResult
  - clarified: "Count cells in the image using watershed segmentation"
  - assumptions: ["Use first frame", "Apply watershed", "Threshold at Otsu"]
  - confidence: 0.85
```

**Step 3: RAG Retrieval** (within planner_node)
```
Query: "Count cells using watershed segmentation"
Process: Search 91 papers for relevant methods
Output: RAGContext with 5 chunks about segmentation methods
```

**Step 4: Planner** ([core/planner.py](core/planner.py))
```
Input: Clarified request + RAG context
Process: Decompose into steps
Output: ExecutionPlan
  - Mode: sequential
  - Steps:
    1. "Segment cells using watershed" (action: segment_cells)
    2. "Count segmented cells" (action: calculate_statistics, depends_on: [1])
```

**Step 5: Multi-Step Executor** (within multi_step_executor_node)
```
For each step:
  a. Generate code using RAG context + step description
  b. Execute code in Docker sandbox
  c. Verify results
  d. Update step status (pending → in_progress → completed)
  e. Pass outputs to next step

Step 1 execution:
  - Code: Load image → Apply watershed → Return segmentation mask
  - Result: {"segmentation_map": array, "num_regions": 42}
  - Verification: PASS (reasonable cell count)

Step 2 execution:
  - Code: Count regions in segmentation_map
  - Result: {"cell_count": 42}
  - Verification: PASS
```

**Step 6: Save Capability** ([layers/capability_manager.py](layers/capability_manager.py))
```
Process: Save reusable code to capability store
Output: Capability saved with ID "cell_counting_watershed_20250122"
```

**Step 7: Format Output**
```
Process: Combine all results into AnalysisResult
Output: Final formatted response with:
  - Data: {"cell_count": 42}
  - Figures: [segmentation visualization]
  - Summary: "Found 42 cells using watershed segmentation"
  - Code: Full Python code used
  - Metadata: Execution time, sources, etc.
```

---

## How Systems Interact

### RAG → Planner → Code Generator
```
User: "Detect calcium transients"
    ↓
RAG retrieves: Papers about OASIS, Suite2p, peak detection
    ↓
Planner uses RAG context: Creates plan for transient detection
    ↓
Code Generator uses RAG context: Implements detection algorithm from papers
```

### Planner → Verifier → Retry Loop
```
Plan Step: "Calculate mean intensity"
    ↓
Execute: Returns {"mean": -50}
    ↓
Verifier: FAIL - negative intensity impossible
    ↓
Retry: Regenerate code with fix
    ↓
Execute: Returns {"mean": 120.5}
    ↓
Verifier: PASS
    ↓
Continue to next step
```

### Clarifier → Planner → Executor
```
Ambiguous: "Analyze the cells"
    ↓
Clarifier: "Segment cells, measure size and intensity, plot distributions"
    ↓
Planner: Creates 3-step DAG plan
    ↓
Executor: Runs steps in dependency order
```

---

## Configuration Settings

### Models Used ([config.py](config.py))

| Component | Model | Purpose |
|-----------|-------|---------|
| Router | `gpt-3.5-turbo` | Fast query classification |
| Clarifier | `gpt-3.5-turbo` | Clarify requests |
| Planner | `gpt-3.5-turbo` | Create execution plans |
| Verifier | `gpt-3.5-turbo` | Verify results |
| Code Generator | `gpt-4` | Generate Python code |
| Informational Response | `gpt-4` | Answer RAG questions |
| RAG Embeddings | `text-embedding-3-small` | Vector embeddings |

### RAG Settings
```python
CHUNK_SIZE = 500              # Characters per chunk
CHUNK_OVERLAP = 100           # Overlap between chunks
TOP_K_CHUNKS = 5              # Chunks retrieved per query
EMBEDDING_MODEL = "text-embedding-3-small"
```

### Execution Settings
```python
CODE_TIMEOUT_SECONDS = 300    # 5 minutes max execution
MAX_RETRIES = 3               # Retry failed steps
```

---

## How to Modify Your System

### 1. Change Routing Behavior

**File**: [core/router.py](core/router.py)

**Current test mode** (forces all to informational):
```python
def route(self, user_request: str) -> RouteType:
    # TESTING MODE: Always route to informational (RAG only)
    return "informational"
```

**Restore normal routing**:
```python
def route(self, user_request: str) -> RouteType:
    # Remove test mode, uncomment original classification logic
    system_prompt = '''You are a query classifier...'''
    # (rest of original code)
```

### 2. Modify Planning Strategy

**File**: [core/planner.py](core/planner.py)

**Make planner prefer parallel execution**:
```python
# Line 56: Change guideline
Guidelines:
- **PREFER PARALLEL**: Use parallel mode whenever steps are independent
- Sequential: Only when steps MUST execute in order
```

**Add new action types**:
```python
# Line 64: Add to available actions
Available actions:
- motion_correction: Correct drift in time series
- roi_selection: Manually select regions of interest
- batch_processing: Process multiple files
```

### 3. Adjust Verification Strictness

**File**: [core/verifier.py](core/verifier.py)

**Make verifier more lenient**:
```python
# Line 158: Change fallback behavior
except Exception as e:
    return VerificationResult(
        passed=True,        # Always pass on error
        confidence=0.8,     # Higher confidence
        should_retry=False  # Don't retry
    )
```

**Add custom checks**:
```python
# Add before LLM verification (line 53)
# Quick sanity checks
if result.results.get("cell_count", 0) > 10000:
    return VerificationResult(
        passed=False,
        confidence=0.95,
        issues=["Cell count suspiciously high (>10k)"],
        should_retry=True
    )
```

### 4. Enhance RAG System

**File**: [layers/rag_system.py](layers/rag_system.py)

**Increase chunk size for more context**:
```python
# Line 125: Modify chunk size
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # Increased from 500
    chunk_overlap=200,  # Increased from 100
)
```

**Retrieve more chunks**:
```python
# config.py
TOP_K_CHUNKS = 10  # Increased from 5
```

**Add reranking**:
```python
# After line 164
# Rerank by relevance
from sentence_transformers import CrossEncoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
scores = reranker.predict([(query, chunk) for chunk in chunks])
# Sort and take top results
```

### 5. Modify Workflow Structure

**File**: [graph/workflow.py](graph/workflow.py)

**Add new node** (e.g., result reviewer):
```python
# Line 66: Add node
workflow.add_node("reviewer", reviewer_node)

# Line 89: Insert in flow
workflow.add_edge("multi_step_executor", "reviewer")
workflow.add_edge("reviewer", "save_capability")
```

**Add conditional branching**:
```python
# Add conditional edge with retry logic
def should_retry(state: PipelineState) -> str:
    if state["execution_results"].success:
        return "save_capability"
    elif state.get("retry_count", 0) < 3:
        return "planner"  # Retry with new plan
    else:
        return "format_output"  # Give up, return error

workflow.add_conditional_edges(
    "multi_step_executor",
    should_retry,
    {
        "save_capability": "save_capability",
        "planner": "planner",
        "format_output": "format_output"
    }
)
```

### 6. Add Progress Tracking

**File**: [graph/nodes.py](graph/nodes.py)

**Report progress to user**:
```python
# In multi_step_executor_node
for step in plan.steps:
    # Report progress
    if state.get("progress_callback"):
        state["progress_callback"](
            f"Executing step {step.step_id}",
            {"step": step.description, "total": len(plan.steps)}
        )
```

---

## System Strengths

✅ **Intelligent routing**: Separates simple questions from complex analyses
✅ **Multi-step planning**: Breaks down complex tasks automatically
✅ **Result verification**: Catches errors before they propagate
✅ **RAG-powered**: Uses scientific literature to inform code generation
✅ **Capability reuse**: Saves and reuses successful analysis patterns
✅ **Error recovery**: Retries failed steps with improved code
✅ **Modular design**: Easy to swap components or add new nodes

---

## Current Limitations & Improvement Opportunities

### 1. Verifier Only Checks Final Results
**Issue**: Verifier runs after code execution, not during.

**Improvement**: Add code review before execution.
```python
# New node: code_reviewer_node
def code_reviewer_node(state):
    # Static analysis, security checks, syntax validation
    # BEFORE executing in Docker
```

### 2. No Learning from Failures
**Issue**: System doesn't remember what failed.

**Improvement**: Save failed attempts to avoid repeating.
```python
# Add to capability_store
def save_failure(request, error, code):
    # Store failures to inform future attempts
```

### 3. RAG Retrieval is Static
**Issue**: Always retrieves same top-K chunks.

**Improvement**: Adaptive retrieval based on plan steps.
```python
# Per-step RAG retrieval
for step in plan.steps:
    step_context = rag.retrieve(step.description, top_k=3)
    # Use step-specific context
```

### 4. No User Interaction During Execution
**Issue**: Can't ask for clarification mid-execution.

**Improvement**: Add human-in-the-loop for ambiguous steps.
```python
# Add approval node
def approval_node(state):
    if step_requires_user_input(state):
        # Pause and request user input
```

### 5. Limited Observability
**Issue**: Hard to debug what went wrong.

**Improvement**: Add detailed logging and visualization.
```python
# Save execution trace
state["trace"] = {
    "router": router_decision,
    "planner": plan_created,
    "executor": step_results,
    "verifier": verification_results
}
```

---

## Quick Reference: Key Files

| File | Purpose | When to Edit |
|------|---------|--------------|
| [graph/workflow.py](graph/workflow.py) | Workflow structure | Add/remove nodes, change routing |
| [graph/state.py](graph/state.py) | State definition | Add new state fields |
| [graph/nodes.py](graph/nodes.py) | Node implementations | Modify node behavior |
| [core/router.py](core/router.py) | Query classification | Change routing logic |
| [core/clarifier.py](core/clarifier.py) | Request clarification | Improve clarification prompts |
| [core/planner.py](core/planner.py) | Execution planning | Add actions, change planning strategy |
| [core/verifier.py](core/verifier.py) | Result verification | Add verification checks |
| [layers/rag_system.py](layers/rag_system.py) | RAG retrieval | Change chunk size, top-K, embeddings |
| [config.py](config.py) | Configuration | Change models, timeouts, parameters |

---

## Testing Your Changes

### Test Router Changes
```python
from core.router import get_router

router = get_router()
route = router.route("How do calcium channels work?")
assert route == "informational"

route = router.route("Count cells in my image")
assert route == "analysis"
```

### Test Planner Changes
```python
from core.planner import get_planner

planner = get_planner()
plan = planner.create_plan("Count cells", ["Use watershed"])
assert len(plan.steps) >= 1
assert plan.execution_mode in ["sequential", "parallel", "dag"]
```

### Test RAG Changes
```python
from layers.rag_system import RAGSystem

rag = RAGSystem()
context = rag.retrieve("calcium transient detection")
assert len(context.chunks) == 5  # TOP_K_CHUNKS
assert all(len(chunk) <= 500 for chunk in context.chunks)
```

### Test Full Workflow
```bash
# Run with test mode enabled (informational path only)
streamlit run app.py
# Enter query: "What are calcium transients?"
# Should see RAG-only response, no code execution

# Disable test mode in router.py
# Enter query: "Count cells in my image"
# Should see full pipeline: clarify → plan → execute → verify
```

---

## Summary

Your system is a sophisticated **multi-agent pipeline** with:

1. **Router**: Intelligent query classification
2. **Clarifier**: Makes assumptions explicit
3. **Planner**: Multi-step execution planning (sequential/parallel/DAG)
4. **RAG**: Scientific literature retrieval from 91 papers
5. **Executor**: Docker-based code execution
6. **Verifier**: Result validation and sanity checking
7. **Capability Store**: Reusable code patterns

The **main workflow file** is [graph/workflow.py](graph/workflow.py) - this defines how everything connects.

To modify behavior, edit the corresponding files above based on what you want to change. The system is designed to be modular and extensible!
