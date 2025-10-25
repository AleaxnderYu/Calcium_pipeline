# Calcium Imaging Pipeline: Visual Workflow Diagram

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         USER REQUEST                                  │
│               "Count cells in the calcium imaging data"              │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          ROUTER NODE                                     │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Classifier (GPT-3.5-turbo)                                      │  │
│  │  "Does this need code execution or just information?"            │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────┬────────────────────────────────────┬───────────────────┘
                 │                                    │
    "informational"                              "analysis"
                 │                                    │
                 ▼                                    ▼
    ┌────────────────────────┐      ┌──────────────────────────────────┐
    │  INFORMATIONAL PATH    │      │     ANALYSIS PATH                │
    │  (RAG-Only Answer)     │      │  (Full Pipeline)                 │
    └────────────────────────┘      └──────────────────────────────────┘
                 │                                    │
                 │                                    │
                 ▼                                    ▼
    ┌────────────────────────┐      ┌──────────────────────────────────┐
    │  RAG Retrieval         │      │  CLARIFIER NODE                  │
    │  - Search 91 papers    │      │  - Make assumptions explicit     │
    │  - Get top 5 chunks    │      │  - Determine frame selection     │
    └────────┬───────────────┘      └────────────┬─────────────────────┘
             │                                   │
             ▼                                   ▼
    ┌────────────────────────┐      ┌──────────────────────────────────┐
    │  LLM Answer            │      │  RAG Retrieval                   │
    │  - Use RAG context     │      │  - Search for methods            │
    │  - Cite sources        │      │  - Get relevant chunks           │
    │  - No code execution   │      └────────────┬─────────────────────┘
    └────────┬───────────────┘                   │
             │                                   ▼
             ▼                       ┌──────────────────────────────────┐
         [ END ]                     │  PLANNER NODE                    │
                                     │  - Decompose into steps          │
                                     │  - Determine dependencies        │
                                     │  - Sequential/Parallel/DAG mode  │
                                     └────────────┬─────────────────────┘
                                                  │
                                                  ▼
                                     ┌──────────────────────────────────┐
                                     │  MULTI-STEP EXECUTOR             │
                                     │  ┌────────────────────────────┐ │
                                     │  │  For each step:            │ │
                                     │  │  1. Generate code (GPT-4)  │ │
                                     │  │  2. Execute in Docker      │ │
                                     │  │  3. Verify results         │ │
                                     │  │  4. Update status          │ │
                                     │  └────────────────────────────┘ │
                                     └────────────┬─────────────────────┘
                                                  │
                                                  ▼
                                     ┌──────────────────────────────────┐
                                     │  SAVE CAPABILITY NODE            │
                                     │  - Store reusable code           │
                                     │  - Version control with Git      │
                                     └────────────┬─────────────────────┘
                                                  │
                                                  ▼
                                     ┌──────────────────────────────────┐
                                     │  FORMAT OUTPUT NODE              │
                                     │  - Combine results               │
                                     │  - Create AnalysisResult         │
                                     └────────────┬─────────────────────┘
                                                  │
                                                  ▼
                                              [ END ]
```

---

## Detailed Multi-Step Executor Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      MULTI-STEP EXECUTOR NODE                             │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────┐
                    │  Execution Plan from      │
                    │  Planner:                 │
                    │  - Step 1: Segment cells  │
                    │  - Step 2: Count cells    │
                    │  - Step 3: Plot results   │
                    └───────────┬───────────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
                ▼                               ▼
    ┌──────────────────────┐     ┌──────────────────────┐
    │  Sequential Mode     │     │  Parallel Mode       │
    │  (Steps in order)    │     │  (Steps concurrent)  │
    └──────────┬───────────┘     └──────────┬───────────┘
               │                            │
               │                            │
               └──────────┬─────────────────┘
                          │
                          ▼
            ┌─────────────────────────────┐
            │  FOR EACH STEP:             │
            └─────────────┬───────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────┐
        │  1. CODE GENERATION                 │
        │  ┌───────────────────────────────┐  │
        │  │  Inputs:                      │  │
        │  │  - Step description           │  │
        │  │  - RAG context (methods)      │  │
        │  │  - Previous step outputs      │  │
        │  │                               │  │
        │  │  LLM (GPT-4):                 │  │
        │  │  Generate Python code         │  │
        │  │                               │  │
        │  │  Output: Python code string   │  │
        │  └───────────────────────────────┘  │
        └─────────────┬───────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────────────┐
        │  2. CODE EXECUTION                  │
        │  ┌───────────────────────────────┐  │
        │  │  Docker Container:            │  │
        │  │  - Mount image directory      │  │
        │  │  - Mount output directory     │  │
        │  │  - Execute Python code        │  │
        │  │  - Capture stdout/stderr      │  │
        │  │  - Save figures/results       │  │
        │  │                               │  │
        │  │  Timeout: 300 seconds         │  │
        │  └───────────────────────────────┘  │
        └─────────────┬───────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────────────┐
        │  3. RESULT VERIFICATION             │
        │  ┌───────────────────────────────┐  │
        │  │  Verifier (GPT-3.5-turbo):    │  │
        │  │                               │  │
        │  │  Checks:                      │  │
        │  │  ✓ Data types correct?        │  │
        │  │  ✓ Values reasonable?         │  │
        │  │  ✓ Output complete?           │  │
        │  │  ✓ Physically plausible?      │  │
        │  │                               │  │
        │  │  Output: VerificationResult   │  │
        │  │  - passed: True/False         │  │
        │  │  - confidence: 0.0-1.0        │  │
        │  │  - should_retry: True/False   │  │
        │  └───────────────────────────────┘  │
        └─────────────┬───────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────────────┐
        │  4. STATUS UPDATE                   │
        │  ┌───────────────────────────────┐  │
        │  │  Update step in plan:         │  │
        │  │  - status: completed          │  │
        │  │  - result: ExecutionResult    │  │
        │  │  - verification: Passed       │  │
        │  └───────────────────────────────┘  │
        └─────────────┬───────────────────────┘
                      │
                      ▼
                ┌─────────┐      ┌──────────────┐
                │ Success │  OR  │ Retry Failed │
                └────┬────┘      └──────┬───────┘
                     │                  │
                     │                  ▼
                     │          ┌───────────────┐
                     │          │ Regenerate    │
                     │          │ code with fix │
                     │          └───────┬───────┘
                     │                  │
                     └──────────┬───────┘
                                │
                                ▼
                    ┌───────────────────┐
                    │  Next Step        │
                    └───────────────────┘
```

---

## RAG System Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                      RAG SYSTEM                                │
└───────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  INITIALIZATION (One-time)                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  data/papers/                                                   │
│  ├── paper1.pdf  ──┐                                            │
│  ├── paper2.pdf  ──┤                                            │
│  ├── paper3.pdf  ──┤                                            │
│  └── ... (91)    ──┤                                            │
│                    │                                            │
│                    ▼                                            │
│         ┌──────────────────────┐                                │
│         │  PDF Loader          │                                │
│         │  (Docling/PyPDF)     │                                │
│         └──────────┬───────────┘                                │
│                    │                                            │
│                    ▼                                            │
│         ┌──────────────────────┐                                │
│         │  Text Splitter       │                                │
│         │  - Chunk: 500 chars  │                                │
│         │  - Overlap: 100      │                                │
│         └──────────┬───────────┘                                │
│                    │                                            │
│                    ▼                                            │
│         ┌──────────────────────┐                                │
│         │  OpenAI Embeddings   │                                │
│         │  (text-embedding-    │                                │
│         │   3-small)           │                                │
│         └──────────┬───────────┘                                │
│                    │                                            │
│                    ▼                                            │
│         ┌──────────────────────┐                                │
│         │  ChromaDB            │                                │
│         │  Vector Database     │                                │
│         │  data/vector_db/     │                                │
│         └──────────────────────┘                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  RETRIEVAL (Per Query)                                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User Query:                                                    │
│  "How to detect calcium transients?"                            │
│                    │                                            │
│                    ▼                                            │
│         ┌──────────────────────┐                                │
│         │  Embed Query         │                                │
│         │  (text-embedding-    │                                │
│         │   3-small)           │                                │
│         └──────────┬───────────┘                                │
│                    │                                            │
│                    ▼                                            │
│         ┌──────────────────────┐                                │
│         │  Similarity Search   │                                │
│         │  (Cosine distance)   │                                │
│         │  Top-K = 5           │                                │
│         └──────────┬───────────┘                                │
│                    │                                            │
│                    ▼                                            │
│         ┌──────────────────────────────────────┐                │
│         │  Retrieved Chunks (RAGContext):      │                │
│         │  ┌────────────────────────────────┐  │                │
│         │  │ Chunk 1: "OASIS algorithm..." │  │                │
│         │  │ Source: transient_detection.pdf│  │                │
│         │  │ Score: 0.89                    │  │                │
│         │  └────────────────────────────────┘  │                │
│         │  ┌────────────────────────────────┐  │                │
│         │  │ Chunk 2: "Peak detection..."   │  │                │
│         │  │ Source: analysis_methods.pdf   │  │                │
│         │  │ Score: 0.85                    │  │                │
│         │  └────────────────────────────────┘  │                │
│         │  ... (3 more chunks)                 │                │
│         └──────────────────────────────────────┘                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Planner Execution Modes

### 1. Sequential Mode
```
Request: "Count cells in the image"

┌──────────────────┐
│  Step 1:         │
│  Segment cells   │
│  (watershed)     │
└────────┬─────────┘
         │
         │ outputs: segmentation_map
         │
         ▼
┌────────────────────┐
│  Step 2:           │
│  Count regions     │
│  (from seg_map)    │
│  depends_on: [1]   │
└────────┬───────────┘
         │
         │ outputs: cell_count
         │
         ▼
┌────────────────────┐
│  Step 3:           │
│  Plot results      │
│  depends_on: [2]   │
└────────────────────┘
```

### 2. Parallel Mode
```
Request: "Calculate mean and std intensity"

         ┌──────────────────┐
    ┌───▶│  Step 1:         │
    │    │  Calculate mean  │
    │    └────────┬─────────┘
    │             │
    │             │ outputs: mean_intensity
    │             │
Start               ▼
    │    ┌──────────────────┐
    │    │  Step 3:         │
    │    │  Plot both       │────▶ END
    │    │  depends_on:     │
    │    │  [1, 2]          │
    │    └────────▲─────────┘
    │             │
    │             │ outputs: std_intensity
    │             │
    │    ┌────────┴─────────┐
    └───▶│  Step 2:         │
         │  Calculate std   │
         └──────────────────┘

(Steps 1 and 2 run in parallel)
```

### 3. DAG (Directed Acyclic Graph) Mode
```
Request: "Segment cells, measure size and intensity, plot both"

┌────────────────┐
│  Step 1:       │
│  Segment cells │
└────┬───────────┘
     │
     │ outputs: segmentation_map
     │
     ├──────────────────┬─────────────────┐
     │                  │                 │
     ▼                  ▼                 ▼
┌────────────┐  ┌────────────────┐  ┌────────────────┐
│  Step 2:   │  │  Step 3:       │  │  Step 4:       │
│  Measure   │  │  Measure       │  │  Extract       │
│  size      │  │  intensity     │  │  timeseries    │
│  [1]       │  │  [1]           │  │  [1]           │
└─────┬──────┘  └────────┬───────┘  └────────┬───────┘
      │                  │                   │
      │                  │                   │
      └──────────────────┼───────────────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  Step 5:        │
                │  Plot all       │
                │  distributions  │
                │  depends_on:    │
                │  [2, 3, 4]      │
                └─────────────────┘

(Steps 2, 3, 4 run in parallel after Step 1 completes)
```

---

## Verifier Decision Flow

```
┌───────────────────────────────────────────────────────┐
│              EXECUTION RESULT                          │
│  {                                                     │
│    "success": True,                                    │
│    "results": {"cell_count": 42},                     │
│    "execution_time": 2.3,                              │
│    "figure": "plot.png"                                │
│  }                                                     │
└────────────────────┬──────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │  Quick Checks          │
        │  ✓ success = True?     │
        │  ✓ results not empty?  │
        └────────┬───────────────┘
                 │
                 ▼
        ┌────────────────────────────────────┐
        │  LLM Verification (GPT-3.5)        │
        │                                    │
        │  Prompt:                           │
        │  "Step: Count cells               │
        │   Result: {cell_count: 42}        │
        │   Context: 512x512 image          │
        │   Is this reasonable?"            │
        └────────┬───────────────────────────┘
                 │
                 ▼
        ┌────────────────────────────────────┐
        │  Verification Result               │
        │  {                                 │
        │    "passed": True,                 │
        │    "confidence": 0.95,             │
        │    "issues": [],                   │
        │    "suggestions": [],              │
        │    "should_retry": False           │
        │  }                                 │
        └────────┬───────────────────────────┘
                 │
          ┌──────┴───────┐
          │              │
          ▼              ▼
    ┌─────────┐    ┌──────────┐
    │  PASS   │    │  FAIL    │
    │  ✓      │    │  ✗       │
    └────┬────┘    └────┬─────┘
         │              │
         │              ▼
         │     ┌────────────────┐
         │     │  Retry Logic:  │
         │     │  - Retry < 3?  │
         │     │  → Regenerate  │
         │     │    code        │
         │     │  - Retry >= 3? │
         │     │  → Fail step   │
         │     └────────────────┘
         │
         ▼
   ┌──────────────┐
   │  Continue to │
   │  next step   │
   └──────────────┘
```

---

## State Flow Through Workflow

```
Initial State                 After Router              After Clarifier
┌──────────────┐             ┌──────────────┐          ┌──────────────────┐
│ user_request │────────────▶│ route_type   │─────────▶│ clarification    │
│ images_path  │             │ = "analysis" │          │ - clarified_req  │
│ ...          │             │              │          │ - assumptions    │
└──────────────┘             └──────────────┘          │ - confidence     │
                                                       └──────────────────┘
                                                                │
                                                                ▼
        After Planner                After RAG
┌─────────────────────┐         ┌──────────────────┐
│ execution_plan      │◀────────│ rag_context      │
│ - steps: [...]      │         │ - chunks: [...]  │
│ - mode: sequential  │         │ - sources: [...]  │
│ - dependencies      │         │ - scores: [...]  │
└─────────────────────┘         └──────────────────┘
          │
          ▼
After Executor                  After Save                After Format
┌──────────────────┐           ┌────────────────┐        ┌──────────────┐
│ execution_results│──────────▶│ capability_id  │───────▶│ final_output │
│ - data           │           │ saved_code     │        │ (Analysis    │
│ - figures        │           │                │        │  Result)     │
│ - success: True  │           └────────────────┘        └──────────────┘
└──────────────────┘
```

---

## Component Interaction Map

```
┌─────────────────────────────────────────────────────────────────────┐
│                         COMPONENT LAYERS                             │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  USER INTERFACE                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  Streamlit   │  │  FastAPI     │  │  Open WebUI  │              │
│  │  app.py      │  │  Backend     │  │  (External)  │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
└─────────┼──────────────────┼──────────────────┼──────────────────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
┌────────────────────────────┼──────────────────────────────────────────┐
│  WORKFLOW ORCHESTRATION    ▼                                          │
│  ┌───────────────────────────────────────────────────────────┐       │
│  │  graph/workflow.py                                        │       │
│  │  - StateGraph definition                                  │       │
│  │  - Node connections                                       │       │
│  │  - Conditional routing                                    │       │
│  └───────────────────────────────────────────────────────────┘       │
│                             │                                         │
│                             ▼                                         │
│  ┌───────────────────────────────────────────────────────────┐       │
│  │  graph/nodes.py                                           │       │
│  │  - router_node                                            │       │
│  │  - clarifier_node                                         │       │
│  │  - planner_node                                           │       │
│  │  - executor_node                                          │       │
│  │  - verifier (called within executor)                      │       │
│  └───────────────────────────────────────────────────────────┘       │
└────────────────────────────┬──────────────────────────────────────────┘
                             │
┌────────────────────────────┼──────────────────────────────────────────┐
│  CORE AGENTS               ▼                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐                │
│  │  Router     │  │  Clarifier   │  │  Planner     │                │
│  │  (GPT-3.5)  │  │  (GPT-3.5)   │  │  (GPT-3.5)   │                │
│  └─────────────┘  └──────────────┘  └──────────────┘                │
│  ┌─────────────┐  ┌──────────────┐                                  │
│  │  Verifier   │  │  Code Gen    │                                  │
│  │  (GPT-3.5)  │  │  (GPT-4)     │                                  │
│  └─────────────┘  └──────────────┘                                  │
└────────────────────────────┬──────────────────────────────────────────┘
                             │
┌────────────────────────────┼──────────────────────────────────────────┐
│  KNOWLEDGE & EXECUTION     ▼                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐      │
│  │  RAG System     │  │  Capability     │  │  Docker         │      │
│  │  - ChromaDB     │  │  Manager        │  │  Executor       │      │
│  │  - 91 papers    │  │  - Store        │  │  - Sandbox      │      │
│  │  - Embeddings   │  │  - Git version  │  │  - 300s timeout │      │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘

Inter-component Communication:
Router ───────────▶ Workflow (route decision)
Clarifier ────────▶ Planner (clarified request)
RAG ──────────────▶ Planner, Code Gen (scientific methods)
Planner ──────────▶ Executor (execution plan)
Executor ─────────▶ Verifier (results for validation)
Verifier ─────────▶ Executor (retry decision)
Executor ─────────▶ Capability Manager (save successful code)
```

---

## Summary

This visual guide shows:

1. **High-level workflow**: Two paths (informational vs analysis)
2. **Multi-step executor**: Detailed step-by-step execution
3. **RAG system**: Initialization and retrieval flows
4. **Planner modes**: Sequential, parallel, and DAG execution
5. **Verifier logic**: Result validation and retry decisions
6. **State evolution**: How state changes through nodes
7. **Component layers**: How all pieces fit together

Use these diagrams to understand how data flows through your system and where to make modifications!
