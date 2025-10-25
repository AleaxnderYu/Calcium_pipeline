# New Workflow V2: Tool-Based Architecture

## Overview

The new workflow removes the router and implements a **user-centric, tool-based** architecture with approval gates and error feedback loops.

## New Workflow Structure

```
User Query
    ↓
Planner (Create tool-based plan)
    ↓
User Approval (Review & approve plan)
    ↓
Orchestrator (Execute tools)
    ├─ Tool 1: RAG
    ├─ Tool 2: Code Generation
    ├─ Tool 3: Execute
    ├─ Tool 4: Verify
    └─ Tool 5: Save Capability
    ↓
Error? ──Yes──> Error Handler ──> Present to User ──> Retry/Abort
    ↓ No
Format Output
    ↓
Return Results
```

## Key Changes from V1

### ❌ Removed
- **Router node**: No more classification into "informational" vs "analysis"
- **Clarifier node**: Assumptions now handled in planner
- **Action-based steps**: Replaced with explicit tool calls
- **Informational path**: All queries follow same workflow

### ✅ Added
- **Tool-based planner**: Creates plans using explicit tool types (RAG, code_gen, execute, verify, etc.)
- **User approval node**: Human-in-the-loop for plan review
- **Orchestrator**: Flexible tool executor supporting sequential/parallel/DAG execution
- **Error feedback loop**: Present errors to user with retry options
- **Cleaner state**: Simplified to just tool_plan, tool_outputs, and errors

## File Structure

### New Files Created
1. **[core/tool_planner.py](core/tool_planner.py)** - Tool-based planner
2. **[core/orchestrator.py](core/orchestrator.py)** - Tool executor
3. **[graph/workflow_v2.py](graph/workflow_v2.py)** - New workflow
4. **[core/data_models.py](core/data_models.py)** - Updated with tool-based models

### Modified Files
- **[graph/state.py](graph/state.py)** - New PipelineState for tool-based workflow

### Backup Files
- **graph/workflow_old.py** - Backup of original workflow

## Tool Types

The new system uses **6 explicit tool types**:

### 1. RAG Tool
**Purpose**: Retrieve scientific methods from papers

**Inputs**:
```python
{
    "query": "cell segmentation methods in calcium imaging"
}
```

**Outputs**:
```python
{
    "chunks": ["...", "..."],
    "sources": ["paper1.pdf", "paper2.pdf"],
    "scores": [0.89, 0.85],
    "metadata": {...}
}
```

### 2. Code Generation Tool
**Purpose**: Generate Python code for analysis

**Inputs**:
```python
{
    "task_description": "Segment cells using watershed",
    "rag_context": {...}  # Optional, from RAG tool
}
```

**Outputs**:
```python
{
    "code": "import numpy as np...",
    "description": "Cell segmentation using watershed"
}
```

### 3. Execute Tool
**Purpose**: Run code in Docker sandbox

**Inputs**:
```python
{
    "code": "...",
    "images_path": "$user.images_path"
}
```

**Outputs**:
```python
{
    "success": True,
    "results": {"cell_count": 42},
    "figures": ["plot.png"],
    "execution_time": 2.3,
    "error_message": ""
}
```

### 4. Verify Tool
**Purpose**: Validate execution results

**Inputs**:
```python
{
    "execution_result": {...},
    "expected_output": "positive integer cell count"
}
```

**Outputs**:
```python
{
    "passed": True,
    "confidence": 0.95,
    "issues": [],
    "suggestions": [],
    "should_retry": False
}
```

### 5. Capability Search Tool
**Purpose**: Find reusable code from past executions

**Inputs**:
```python
{
    "query": "cell counting"
}
```

**Outputs**:
```python
{
    "found": True,
    "capability_id": "cell_counting_20250122",
    "code": "...",
    "similarity": 0.92
}
```

### 6. Capability Save Tool
**Purpose**: Save successful code for future reuse

**Inputs**:
```python
{
    "code": "...",
    "description": "Cell counting with watershed"
}
```

**Outputs**:
```python
{
    "capability_id": "cell_counting_20250122",
    "saved": True
}
```

## Tool Plan Example

**User Request**: "Count cells in the image"

**Generated Tool Plan**:
```json
{
    "plan_id": "plan_20250122_ab12cd34",
    "original_request": "Count cells in the image",
    "assumptions": [
        "Use watershed segmentation",
        "Process first frame if time series",
        "Count connected components as cells"
    ],
    "execution_mode": "sequential",
    "tool_calls": [
        {
            "tool_id": "t1",
            "tool_type": "rag",
            "description": "Retrieve cell segmentation and counting methods",
            "inputs": {"query": "cell segmentation and counting in calcium imaging"},
            "depends_on": []
        },
        {
            "tool_id": "t2",
            "tool_type": "code_generation",
            "description": "Generate code to segment and count cells",
            "inputs": {
                "task_description": "Segment cells using watershed and count them",
                "rag_context": "$t1.output"
            },
            "depends_on": ["t1"]
        },
        {
            "tool_id": "t3",
            "tool_type": "execute",
            "description": "Execute segmentation and counting",
            "inputs": {
                "code": "$t2.output.code",
                "images_path": "$user.images_path"
            },
            "depends_on": ["t2"]
        },
        {
            "tool_id": "t4",
            "tool_type": "verify",
            "description": "Verify cell count is reasonable",
            "inputs": {
                "execution_result": "$t3.output",
                "expected_output": "positive integer cell count"
            },
            "depends_on": ["t3"]
        },
        {
            "tool_id": "t5",
            "tool_type": "capability_save",
            "description": "Save cell counting capability",
            "inputs": {
                "code": "$t2.output.code",
                "description": "Cell segmentation and counting using watershed"
            },
            "depends_on": ["t4"]
        }
    ]
}
```

## User Approval Flow

### 1. Plan is Presented to User

```
Plan Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Request: Count cells in the image

Assumptions:
  • Use watershed segmentation
  • Process first frame if time series
  • Count connected components as cells

Execution Plan (5 tools, sequential):
  1. [rag] Retrieve cell segmentation and counting methods
  2. [code_generation] Generate code to segment and count cells
  3. [execute] Execute segmentation and counting
  4. [verify] Verify cell count is reasonable
  5. [capability_save] Save cell counting capability

Estimated Time: ~10 seconds
Estimated Cost: ~$0.02

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Actions:
  [A] Approve and execute
  [M] Modify plan
  [R] Reject and replan
```

### 2. User Approves

The orchestrator begins executing tools in order.

### 3. User Modifies

User can:
- Remove tools
- Add tools
- Change tool parameters
- Modify assumptions

## Error Handling Flow

### When Error Occurs

```
ERROR during execution
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Failed Tool: t3 [execute]
Description: Execute segmentation and counting

Error Message:
  NameError: name 'watershed' is not defined

Context:
  • Step: 3 of 5
  • Code generated but execution failed
  • Previous tools completed successfully

Suggestions:
  • Retry with corrected code
  • Regenerate code with explicit imports
  • Skip verification and continue

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Actions:
  [R] Retry (attempt 1 of 3)
  [M] Modify and retry
  [S] Skip and continue
  [A] Abort execution
```

### User Chooses Retry

Orchestrator:
1. Increments retry count
2. Marks tool as RETRY status
3. Re-executes the failed tool
4. Continues if successful

## Execution Modes

### Sequential Mode (Default)
Tools execute one after another in order.

```
t1 → t2 → t3 → t4 → t5
```

**Use when**: Most tasks (default behavior)

### Parallel Mode
Independent tools execute simultaneously.

```
     ┌─ t1 ─┐
     ├─ t2 ─┤
Start         End
     ├─ t3 ─┤
     └─ t4 ─┘
```

**Use when**: Tools don't depend on each other
**Example**: Calculate mean AND std at the same time

### DAG Mode (Directed Acyclic Graph)
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

**Use when**: Some tools can run in parallel, others have dependencies
**Example**: Segment cells → [Measure size, Measure intensity] → Plot both

## Input References

Tools can reference outputs from previous tools using `$tool_id.output.field` syntax:

```python
{
    "tool_id": "t2",
    "inputs": {
        "code": "$t1.output.code",           # Reference t1's output code
        "rag_context": "$t0.output",          # Reference entire t0 output
        "images_path": "$user.images_path"   # Reference user input
    }
}
```

The orchestrator automatically resolves these references before executing each tool.

## State Management

### PipelineState Fields

```python
{
    # Inputs
    "user_request": "Count cells in the image",
    "images_path": "/path/to/images",

    # Planning
    "tool_plan": ToolBasedPlan(...),

    # Approval
    "user_approval": UserApprovalResponse(approved=True, ...),
    "waiting_for_approval": False,

    # Execution
    "tool_outputs": {
        "t1": {"chunks": [...], "sources": [...]},
        "t2": {"code": "...", "description": "..."},
        "t3": {"success": True, "results": {...}}
    },

    # Error handling
    "current_error": ErrorFeedback(...) or None,
    "waiting_for_error_response": False,
    "errors": [],

    # Output
    "final_output": AnalysisResult(...),

    # Metadata
    "session_id": "session_123",
    "output_path": "/tmp/calcium_output_xyz"
}
```

## How to Use the New Workflow

### Basic Usage

```python
from graph.workflow_v2 import run_workflow

# Run analysis
result = run_workflow(
    user_request="Count cells in the image",
    images_path="/path/to/images"
)

print(result.data)
# {'cell_count': 42}

print(result.summary)
# "Executed 5 tools for: Count cells in the image..."
```

### With Session Reuse

```python
# First request
result1 = run_workflow(
    user_request="Count cells",
    images_path="/path/to/images",
    session_id="my_session"
)

# Second request (reuses Docker session)
result2 = run_workflow(
    user_request="Calculate mean intensity",
    images_path="/path/to/images",
    session_id="my_session"
)
```

## Migrating from V1 to V2

### To Use V2 in Your Code

**Option 1: Import V2 explicitly**
```python
from graph.workflow_v2 import run_workflow as run_workflow_v2

result = run_workflow_v2(user_request="...", images_path="...")
```

**Option 2: Replace default workflow**
```bash
# Backup old workflow
cp graph/workflow.py graph/workflow_v1.py

# Use V2 as default
cp graph/workflow_v2.py graph/workflow.py
```

### Key Differences

| Feature | V1 (Old) | V2 (New) |
|---------|----------|----------|
| Routing | Router classifies query | No router, all queries same path |
| Planning | Action-based steps | Tool-based calls |
| User approval | None | Required before execution |
| Error handling | Auto-retry internally | Present to user for decision |
| Informational queries | Separate path (RAG-only) | Same path, plan with just RAG tool |
| State | Many fields (route_type, clarification, etc.) | Simplified (tool_plan, tool_outputs) |
| Execution | Multi-step executor | Orchestrator |

## Benefits of V2

✅ **User control**: Approve plans before execution
✅ **Transparency**: See exactly what tools will run
✅ **Error visibility**: Errors presented to user with options
✅ **Flexibility**: Modify plans before execution
✅ **Simplicity**: Single workflow path for all queries
✅ **Tool-based**: Clear separation of concerns (RAG, code gen, execute, verify)
✅ **Debugging**: Easier to track which tool failed
✅ **Reusability**: Tool outputs cached and referenceable

## Testing V2

### Test 1: Simple Query

```python
from graph.workflow_v2 import run_workflow

result = run_workflow(
    user_request="Show me the first frame",
    images_path="data/images"
)

# Expected plan:
# t1: [code_generation] Generate code to display first frame
# t2: [execute] Execute display code

assert result.data is not None
assert len(result.figures) > 0
```

### Test 2: Complex Analysis

```python
result = run_workflow(
    user_request="Count cells and measure their sizes",
    images_path="data/images"
)

# Expected plan:
# t1: [rag] Retrieve segmentation methods
# t2: [code_generation] Generate segmentation code
# t3: [execute] Run segmentation
# t4: [code_generation] Generate measurement code
# t5: [execute] Run measurements
# t6: [verify] Verify results
# t7: [capability_save] Save capability

assert "cell_count" in result.data
assert "cell_sizes" in result.data
```

### Test 3: Informational Query

```python
result = run_workflow(
    user_request="What are calcium transients?",
    images_path=""  # No images needed
)

# Expected plan:
# t1: [rag] Retrieve information about calcium transients

assert "calcium" in result.summary.lower()
assert "transient" in result.summary.lower()
```

## Next Steps

1. ✅ Data models updated with tool-based structures
2. ✅ Tool-based planner created
3. ✅ Orchestrator with tool execution created
4. ✅ New workflow V2 implemented
5. ⏳ Update UI to show plan approval (Streamlit/FastAPI)
6. ⏳ Implement actual user approval mechanism
7. ⏳ Implement error feedback UI
8. ⏳ Add plan modification interface
9. ⏳ Test with real calcium imaging data

## Questions?

See [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) for more details on the overall system design.
