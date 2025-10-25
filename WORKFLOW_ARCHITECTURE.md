# Calcium Imaging Pipeline: Workflow Architecture

## Overview

Tool-based workflow with user approval and error feedback for calcium imaging analysis.

## Workflow Structure

```
User Query (from Open WebUI)
    ↓
Planner (Create tool-based execution plan)
    ↓
User Approval (Present plan to user via Open WebUI)
    ↓
Orchestrator (Execute tools sequentially/parallel/DAG)
    ├─ RAG Tool: Retrieve scientific methods
    ├─ Code Generation Tool: Generate Python code
    ├─ Execute Tool: Run code in Docker sandbox
    ├─ Verify Tool: Validate results
    ├─ Capability Search/Save: Reuse successful patterns
    └─ (Loop if dependencies exist)
    ↓
Error? ──Yes──> Error Handler ──> Present to User ──> Retry/Skip/Abort
    ↓ No
Format Output (Combine results)
    ↓
Return AnalysisResult (to Open WebUI)
```

## Core Components

### 1. Tool-Based Planner ([core/tool_planner.py](core/tool_planner.py))
**Purpose**: Decompose user requests into executable tool calls

**Input**: User request string
**Output**: ToolBasedPlan with tool calls and dependencies

**Example**:
```python
User: "Count cells in the image"

Plan:
  t1: [rag] Retrieve cell segmentation methods
  t2: [code_generation] Generate segmentation code
  t3: [execute] Run segmentation
  t4: [verify] Verify cell count is reasonable
  t5: [capability_save] Save for future reuse
```

### 2. Orchestrator ([core/orchestrator.py](core/orchestrator.py))
**Purpose**: Execute tool calls with dependency management

**Features**:
- Sequential execution (most common)
- Parallel execution (independent tools)
- DAG execution (complex dependencies)
- Input reference resolution (`$t1.output.code`)
- Error handling and retry

### 3. Workflow ([graph/workflow.py](graph/workflow.py))
**Purpose**: Define the execution flow

**Nodes**:
1. `planner_node` - Create plan
2. `user_approval_node` - Wait for approval (auto-approves for now)
3. `orchestrator_node` - Execute tools
4. `error_handler_node` - Handle failures
5. `format_output_node` - Format results

## Tool Types

### RAG Tool
Retrieve scientific methods from 91 PDF papers.

```python
Input:  {"query": "cell segmentation methods"}
Output: {"chunks": [...], "sources": [...], "scores": [...]}
```

### Code Generation Tool
Generate Python code using GPT-4.

```python
Input:  {"task_description": "...", "rag_context": {...}}
Output: {"code": "import numpy as np...", "description": "..."}
```

### Execute Tool
Run code in Docker sandbox.

```python
Input:  {"code": "...", "images_path": "..."}
Output: {"success": True, "results": {...}, "figures": [...]}
```

### Verify Tool
Validate execution results using GPT-3.5-turbo.

```python
Input:  {"execution_result": {...}, "expected_output": "..."}
Output: {"passed": True, "confidence": 0.95, "issues": []}
```

### Capability Search/Save Tools
Search for or save reusable code patterns.

```python
Search Input:  {"query": "cell counting"}
Search Output: {"found": True, "code": "...", "similarity": 0.92}

Save Input:  {"code": "...", "description": "..."}
Save Output: {"capability_id": "...", "saved": True}
```

## Data Models

### ToolBasedPlan
```python
{
    "plan_id": "plan_20250122_abc123",
    "original_request": "Count cells",
    "tool_calls": [ToolCall(...), ...],
    "execution_mode": "sequential",
    "assumptions": ["Use watershed", ...],
    "user_approved": False
}
```

### ToolCall
```python
{
    "tool_id": "t1",
    "tool_type": ToolType.RAG,
    "description": "Retrieve segmentation methods",
    "inputs": {"query": "..."},
    "depends_on": [],
    "status": StepStatus.PENDING,
    "output": None
}
```

### PipelineState
```python
{
    "user_request": "Count cells",
    "images_path": "/path/to/images",
    "tool_plan": ToolBasedPlan(...),
    "user_approval": UserApprovalResponse(...),
    "tool_outputs": {"t1": {...}, "t2": {...}},
    "current_error": None,
    "final_output": AnalysisResult(...)
}
```

## Execution Modes

### Sequential (Default)
Tools execute one after another.
```
t1 → t2 → t3 → t4 → t5
```

### Parallel
Independent tools execute simultaneously.
```
     ┌─ t1 ─┐
     ├─ t2 ─┤
Start         End
     ├─ t3 ─┤
     └─ t4 ─┘
```

### DAG (Directed Acyclic Graph)
Complex dependencies with partial parallelism.
```
      t1
      ↓
   ┌──┴──┐
   ↓     ↓
  t2    t3
   ↓     ↓
   └──┬──┘
      ↓
      t4
```

## Input References

Tools reference previous outputs using `$tool_id.output.field`:

```python
{
    "tool_id": "t2",
    "inputs": {
        "code": "$t1.output.code",          # Use code from t1
        "rag_context": "$t0.output",         # Use all output from t0
        "images_path": "$user.images_path"  # Use user input
    }
}
```

The orchestrator resolves these automatically before execution.

## Error Handling

When a tool fails:

1. **Capture Error**: Create ErrorFeedback with context
2. **Present to User**: Via Open WebUI (TODO: implement UI)
3. **User Decision**:
   - Retry (up to 3 times)
   - Skip and continue
   - Abort execution
4. **Execute Decision**: Orchestrator continues based on choice

Currently auto-retries once then aborts (UI integration pending).

## Integration with Open WebUI

### API Endpoint ([api_backend.py](api_backend.py))

The FastAPI backend exposes the workflow to Open WebUI:

```python
@app.post("/api/chat")
async def chat(request: ChatRequest):
    # Extract user message
    user_request = request.messages[-1].content

    # Run workflow
    result = run_workflow(
        user_request=user_request,
        images_path=settings.images_path
    )

    # Return formatted response
    return {
        "message": result.summary,
        "data": result.data,
        "figures": result.figures
    }
```

### User Approval (TODO)

Present plan to user via Open WebUI function calling:

```python
# In planner_node
approval_request = UserApprovalRequest(
    plan=tool_plan,
    estimated_time="~10 seconds",
    estimated_cost="~$0.02"
)

# Send to Open WebUI for approval
# (Currently auto-approves)
```

### Error Feedback (TODO)

Present errors to user via Open WebUI:

```python
# In error_handler_node
error_feedback = ErrorFeedback(
    error_message=str(e),
    failed_tool=tool_call,
    suggestions=["Retry", "Skip", "Abort"]
)

# Send to Open WebUI for user decision
# (Currently auto-retries)
```

## File Structure

```
calcium_pipeline/
├── core/
│   ├── tool_planner.py        # Tool-based planner
│   ├── orchestrator.py         # Tool executor
│   ├── verifier.py             # Result validator
│   ├── docker_executor.py      # Docker sandbox
│   ├── session_manager.py      # Session management
│   └── data_models.py          # All data structures
├── graph/
│   ├── workflow.py             # Main workflow
│   └── state.py                # PipelineState definition
├── layers/
│   ├── rag_system.py           # RAG (91 papers)
│   ├── capability_manager.py   # Capability storage
│   └── preprocessor.py         # Image preprocessing
├── api_backend.py              # FastAPI for Open WebUI
└── main.py                     # Entry point
```

## Usage

### Basic

```python
from graph.workflow import run_workflow

result = run_workflow(
    user_request="Count cells in the image",
    images_path="data/images"
)

print(result.data)  # {'cell_count': 42}
```

### With Session Reuse

```python
result1 = run_workflow(
    user_request="Count cells",
    images_path="data/images",
    session_id="session_123"
)

result2 = run_workflow(
    user_request="Measure cell sizes",
    images_path="data/images",
    session_id="session_123"  # Reuses Docker container
)
```

### Via Open WebUI

```bash
# Start FastAPI backend
uvicorn api_backend:app --host 0.0.0.0 --port 8000

# Configure Open WebUI to use http://localhost:8000
```

## Next Steps

### Immediate (Backend Complete)
- ✅ Tool-based planner
- ✅ Orchestrator with tool execution
- ✅ Error handling logic
- ✅ All data models

### Pending (UI Integration)
- ⏳ User approval UI in Open WebUI
- ⏳ Error feedback UI in Open WebUI
- ⏳ Plan modification interface
- ⏳ Real-time progress reporting

## Key Benefits

✅ **Transparent**: See exactly what tools will run
✅ **User Control**: Approve/reject/modify plans
✅ **Error Visibility**: Errors shown with retry options
✅ **Tool-Based**: Clear separation of concerns
✅ **Flexible**: Sequential/parallel/DAG execution
✅ **Reusable**: Tool outputs cached and referenced
✅ **Simple**: Single workflow path, no router complexity

## Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Planner | GPT-3.5-turbo | Create tool plans (fast, cheap) |
| Code Generation | GPT-4 | Generate Python code (high quality) |
| Verifier | GPT-3.5-turbo | Validate results (fast) |
| RAG Embeddings | text-embedding-3-small | Vector search |

## Configuration

Edit [config.py](config.py) to change:
- Model selection
- RAG parameters (chunk size, top-K)
- Docker timeout (default 300s)
- Max retries (default 3)

## Documentation

- **[WORKFLOW_ARCHITECTURE.md](WORKFLOW_ARCHITECTURE.md)** - This file
- **[MIGRATION_TO_V2.md](MIGRATION_TO_V2.md)** - What changed from V1
- **[NEW_WORKFLOW_V2.md](NEW_WORKFLOW_V2.md)** - Detailed V2 guide
- **[DOCLING_INTEGRATION.md](DOCLING_INTEGRATION.md)** - PDF parsing with Docling
- **[OPEN_WEBUI_SETUP.md](OPEN_WEBUI_SETUP.md)** - Open WebUI integration

The workflow is production-ready for Open WebUI integration!
