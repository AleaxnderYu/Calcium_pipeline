# Calcium Imaging System V2 Architecture

## Overview

Enhanced agentic system with clarification, multi-step planning, progressive execution, verification, and user interruption capabilities.

## System Flow

```
User Query
    ↓
[ROUTER] → Classify: "analysis" or "informational"
    ↓
    ├─ "informational" ─────→ [RAG Response] → Display answer → END
    │
    └─ "analysis" ──────────→ [CLARIFIER] ← Show assumptions & suggestions
                                  ↓
                              [PLANNER] ← Create multi-step plan
                                  ↓
                              Display plan to user
                                  ↓
                              [MULTI-STEP EXECUTOR]
                                  ↓
                         For each step in plan:
                              ├─ Preprocess (if needed)
                              ├─ RAG Retrieval
                              ├─ Capability Search
                              ├─ Code Generation
                              ├─ Execute Code → Show results
                              ├─ [VERIFIER] → Check results
                              │     ↓
                              │   Pass? ──No──→ Retry once
                              │     ↓
                              │    Yes
                              └─ Mark step complete
                                  ↓
                              [Check for interruption]
                                  ↓
                              User interrupted? ──Yes──→ STOP
                                  ↓
                                 No
                                  ↓
                              Next step
                                  ↓
                              All steps complete
                                  ↓
                              [FORMAT FINAL OUTPUT]
                                  ↓
                              Display summary
                                  ↓
                              END
```

## Key Components

### 1. Router (`core/router.py`)
- **Model**: GPT-3.5-turbo
- **Purpose**: Classify query intent
- **Output**: "analysis" or "informational"

### 2. Clarifier (`core/clarifier.py`)
- **Model**: GPT-3.5-turbo
- **Purpose**: Make ambiguities explicit
- **Output**: ClarificationResult
  - `clarified_request`: Unambiguous version
  - `assumptions`: List of assumptions made
  - `suggestions`: Alternative approaches
  - `parameters`: Extracted parameters

**Example**:
```
Input: "Count cells"
Output:
  Clarified: "Segment cells using watershed algorithm and count them across all frames"
  Assumptions:
    • Using all available frames
    • Using watershed segmentation (standard for calcium imaging)
    • Entire image region
  Suggestions:
    • Consider using adaptive thresholding if cells have varying brightness
```

### 3. Planner (`core/planner.py`)
- **Model**: GPT-3.5-turbo
- **Purpose**: Decompose complex requests
- **Output**: ExecutionPlan with steps and dependencies

**Execution Modes**:
- **Sequential**: Steps must run in order
  - Example: "Segment → Count → Plot"
- **Parallel**: Steps can run simultaneously
  - Example: "Calculate mean AND std"
- **DAG**: Complex dependencies
  - Example: "Segment → [Count, Measure] → Plot both"

**Example Plan**:
```
Request: "Segment cells, measure their properties, then plot the distribution"
Mode: DAG
Steps:
  1. Segment cells using watershed (no dependencies)
  2. Measure cell sizes (depends on step 1)
  3. Measure cell intensities (depends on step 1)
  4. Plot distributions (depends on steps 2 & 3)
```

### 4. Verifier (`core/verifier.py`)
- **Model**: GPT-3.5-turbo
- **Purpose**: Validate execution results
- **Output**: VerificationResult
  - `passed`: bool
  - `confidence`: 0.0 to 1.0
  - `issues`: List of problems
  - `suggestions`: How to fix
  - `should_retry`: bool

**Checks**:
- Data type correctness (e.g., counts should be integers)
- Value reasonableness (e.g., cell count shouldn't be 10000 in small image)
- Output completeness (e.g., expected fields present?)
- Physical plausibility (e.g., intensity shouldn't exceed pixel range)

**Example Verification**:
```
Step: "Calculate mean intensity"
Result: {"mean_intensity": -50.5}
Context: Pixel range is [0, 255]
Verification:
  Passed: False
  Confidence: 0.9
  Issues: ["Mean intensity is negative, but pixel range is [0, 255]"]
  Suggestions: ["Check if normalization was applied incorrectly"]
  Should Retry: True
```

### 5. Progress Reporter (`core/progress.py`)
- **Purpose**: Transparent communication with user
- **Events**: router, clarifier, planner, step_start, step_complete, verification, etc.
- **Outputs**: Console logs + optional callback for UI integration

**Progressive Output Example**:
```
💭 Clarified request: Segment cells using watershed and count them
   Assumptions:
   • Using all frames (1-10)
   • Standard watershed parameters

📋 Execution Plan (sequential mode):
   1. Segment cells using watershed
   2. Count segmented cells

📚 Retrieved knowledge from: segmentation_methods.txt

▶️  Executing: Segment cells using watershed
✓ Complete
   Results: {"num_cells": 42, "segmentation_map": ...}
   🔍 Verified (confidence: 95%)

▶️  Executing: Count segmented cells
✓ Complete
   Results: {"total_count": 42}
   🔍 Verified (confidence: 100%)
```

### 6. Interruption Mechanism
- User can set `state["interrupted"] = True` at any time
- System checks before each step
- Gracefully stops and returns partial results

## Data Models

### ExecutionPlan
```python
@dataclass
class ExecutionPlan:
    plan_id: str
    original_request: str
    clarified_request: str
    assumptions: List[str]
    steps: List[ExecutionStep]
    execution_mode: Literal["sequential", "parallel", "dag"]
    current_step_index: int
    is_complete: bool
```

### ExecutionStep
```python
@dataclass
class ExecutionStep:
    step_id: str
    description: str
    action: str  # e.g., "segment_cells"
    depends_on: List[str]  # Step IDs this depends on
    status: StepStatus  # PENDING, RUNNING, COMPLETED, FAILED, VERIFIED
    code: Optional[str]
    result: Optional[ExecutionResult]
    verification_passed: bool
    verification_message: str
    retry_count: int
```

### ClarificationResult
```python
@dataclass
class ClarificationResult:
    clarified_request: str
    assumptions: List[str]
    suggestions: List[str]
    parameters: Dict[str, Any]
```

### VerificationResult
```python
@dataclass
class VerificationResult:
    passed: bool
    confidence: float  # 0.0 to 1.0
    issues: List[str]
    suggestions: List[str]
    should_retry: bool
```

## Updated Workflow

1. **Router**: Classify query
2. **Clarifier**: Make assumptions explicit (for analysis queries)
3. **Planner**: Create execution plan
4. **For each step in plan**:
   a. Report step start
   b. Preprocess (if first step)
   c. RAG retrieval
   d. Capability search
   e. Code generation
   f. Execute code
   g. **Verify results**
   h. If verification fails: Retry once with modified approach
   i. Report step complete
   j. **Check for interruption**
   k. If interrupted: Stop and return partial results
5. **Format final output**: Combine all step results

## User Experience

### CLI Example
```bash
$ python main.py --request "Segment cells and count them" --images ./data/images

💭 Clarified request: Segment cells using watershed algorithm and count them across all frames
   Assumptions:
   • Using all frames (1-10)
   • Using watershed segmentation

📋 Execution Plan (sequential mode):
   1. Segment cells using watershed
   2. Count segmented cells

▶️  Executing: Segment cells using watershed
   📚 Retrieved knowledge from: segmentation_methods.txt
   ✓ Code generated (42 lines)
   ✓ Executed in 3.2s
   🔍 Verified (confidence: 95%)

▶️  Executing: Count segmented cells
   ✓ Code generated (12 lines)
   ✓ Executed in 0.5s
   🔍 Verified (confidence: 100%)

✅ Analysis complete!
Results:
  - num_cells: 42
  - average_cell_size: 156.3 pixels
```

### Streamlit UI Integration
- Progress events displayed in expandable sections
- Each step shows:
  - Status (running/complete/failed)
  - Retrieved RAG sources
  - Generated code (in expander)
  - Results
  - Verification status
- "Stop" button sets `interrupted` flag

## Benefits

1. **Transparency**: User sees what system is doing at each step
2. **Correctness**: Verification catches errors before they propagate
3. **Flexibility**: Multi-step plans handle complex requests
4. **Control**: User can interrupt at any time
5. **Cost Efficiency**: Verification prevents wasted computation on bad results
6. **Smart Retry**: Auto-retry on verification failures
7. **Explainability**: Assumptions and suggestions make system reasoning clear

## Next Steps for Full Implementation

1. ✅ Create data models (DONE)
2. ✅ Create clarifier, planner, verifier components (DONE)
3. ✅ Create progress reporter (DONE)
4. ⏳ Update workflow nodes to use new components
5. ⏳ Add multi-step execution loop
6. ⏳ Integrate verification and retry logic
7. ⏳ Add interruption checks
8. ⏳ Update Streamlit UI for progressive display
9. ⏳ Add "Stop" button to UI
10. ⏳ Test with complex multi-step queries

## Example Multi-Step Query

**Request**: "Segment cells, measure their mean intensity over time, detect calcium transients, and plot everything"

**Clarified**: "Segment cells using watershed, extract intensity timeseries for each cell, detect transients using peak detection, create visualization with segmentation map, timeseries plots, and transient markers"

**Plan** (DAG mode):
1. Segment cells → outputs segmentation_map
2. Extract timeseries (depends on 1) → outputs cell_timeseries
3. Detect transients (depends on 2) → outputs transient_events
4. Plot segmentation (depends on 1)
5. Plot timeseries (depends on 2, 3)
6. Combine plots → final figure

**Execution**: 6 steps with dependencies, each verified, progressive output shown

