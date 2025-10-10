# Proof of Concept Plan: Calcium Imaging Agentic System
**Updated for LangGraph + LangChain Stack**

---

## **1. PROJECT OVERVIEW**

### **Objective**
Build a minimal agentic system where LangGraph orchestrates a workflow that processes calcium imaging data through RAG-enhanced code generation. The system uses LangChain for paper retrieval, GPT-4/Claude for code generation, and validates inter-component coordination.

### **Technology Stack**
- **Orchestration**: LangGraph (state machine workflow)
- **Task Decomposition**: GPT-4 for breaking complex tasks into atomic subtasks
- **RAG System**: LangChain + ChromaDB (paper retrieval)
- **Preprocessing**: Hard-coded Python (load/normalize PNGs)
- **Code Generation**: GPT-4 or Claude 3.5 Sonnet via API (atomic capabilities)
- **Capability Composition**: Chain atomic capabilities into complete pipelines
- **Execution**: Simple Python exec() with timeout (POC level)

### **Success Criteria**
- User submits natural language request → receives analysis results
- LangGraph workflow executes all nodes in correct sequence
- RAG system retrieves relevant paper chunks from vector database
- Code generator produces syntactically valid Python using RAG context
- Generated code executes successfully on preprocessed data
- System logs all state transitions for debugging

---

## **2. SYSTEM ARCHITECTURE**

### **LangGraph Workflow Structure**
```
START
  ↓
[preprocess_node] Load and normalize images
  ↓
[rag_retrieval_node] Get scientific context from papers
  ↓
[task_decomposition_node] Break request into atomic subtasks
  ↓
[subtask_resolution_node] For each subtask:
  ├─ Search for existing atomic capability
  ├─ Generate new atomic capability if not found
  └─ Save atomic capability
  ↓
[composition_node] Chain atomic capabilities into pipeline
  ↓
[execution_node] Run composed pipeline
  ↓
[format_output_node] Package results
  ↓
END
```

**Key Change:** Instead of generating one monolithic capability per request, the system now:
1. Decomposes complex requests into atomic subtasks
2. Finds/generates small, reusable atomic capabilities for each subtask
3. Composes atomic capabilities into a complete pipeline
4. Enables high reuse rate across different complex requests

### **State Object (Flows Through Graph)**
```
PipelineState:
  - user_request: str
  - images_path: str
  - preprocessed_data: PreprocessedData
  - rag_context: RAGContext

  # Task decomposition
  - subtasks: List[Subtask]  # Decomposed atomic subtasks
  - atomic_capabilities: Dict[str, Capability]  # Subtask ID → capability
  - composed_code: str  # Final pipeline code

  - execution_results: Dict
  - final_output: AnalysisResult
  - errors: List[str]

  # Capability tracking
  - capability_reused: bool
  - capabilities_reused_count: int  # How many subtasks reused
  - capabilities_generated_count: int  # How many new subtasks
```

---

## **3. FILE STRUCTURE**

```
calcium_pipeline/
├── README.md                          # Setup instructions
├── requirements.txt                   # Dependencies
├── .env.example                       # API key template
├── config.py                          # Configuration
├── main.py                            # Entry point
│
├── graph/
│   ├── __init__.py
│   ├── workflow.py                    # LangGraph workflow definition
│   ├── state.py                       # PipelineState TypedDict
│   └── nodes.py                       # Node functions (each layer)
│
├── layers/
│   ├── __init__.py
│   ├── task_decomposer.py            # GPT-4 task decomposition
│   ├── rag_system.py                 # LangChain RAG implementation
│   ├── preprocessor.py               # Image loading/normalization
│   ├── capability_manager.py         # GPT-4 atomic code generation
│   ├── capability_store.py           # Atomic capability storage & retrieval
│   └── capability_composer.py        # Chain atomic capabilities
│
├── core/
│   ├── __init__.py
│   ├── executor.py                   # Safe code execution
│   └── data_models.py                # Shared dataclasses
│
├── utils/
│   ├── __init__.py
│   ├── logging_config.py             # Structured logging
│   └── helpers.py                    # Utility functions
│
├── data/
│   ├── papers/                        # Mock calcium imaging papers
│   │   ├── segmentation_methods.txt
│   │   ├── transient_detection.txt
│   │   └── baseline_calculation.txt
│   ├── images/                        # Sample PNG sequences
│   │   ├── frame_001.png
│   │   ├── frame_002.png
│   │   └── frame_003.png
│   ├── vector_db/                     # ChromaDB for papers (auto-created)
│   └── capability_store/              # NEW: Generated capabilities storage
│       ├── .git/                      # Version control for generated code
│       ├── capabilities/              # Stored capability files
│       │   ├── cap_xxx.py            # Generated Python scripts
│       │   └── cap_xxx.json          # Capability metadata
│       └── capability_db/             # ChromaDB for capability search
│
├── outputs/                           # Generated results
│
└── tests/
    ├── test_workflow.py               # Integration test
    ├── test_rag.py                    # L2 unit tests
    ├── test_executor.py               # Execution safety tests
    └── test_capability_store.py       # NEW: Capability store tests
```

---

## **4. DEPENDENCIES** (`requirements.txt`)

```
# Core orchestration
langgraph>=0.0.20
langchain>=0.1.0
langchain-openai>=0.0.5
langchain-community>=0.0.10

# Vector database
chromadb>=0.4.0

# LLM API
openai>=1.0.0

# Data processing
numpy>=1.24.0
scipy>=1.11.0
scikit-image>=0.21.0
matplotlib>=3.7.0
Pillow>=10.0.0

# Utilities
python-dotenv>=1.0.0
pydantic>=2.0.0

# Version control for capabilities
gitpython>=3.1.0

# Optional (for PDF support later)
# pypdf>=3.0.0
```

---

## **5. CONFIGURATION** (`config.py`)

### **Purpose**
Centralize all configuration with environment variable support

### **Required Settings**

**API Configuration:**
- `OPENAI_API_KEY`: From environment variable (required)
- `OPENAI_MODEL`: Default "gpt-4" (can be "gpt-4-turbo" or "claude-3-5-sonnet-20241022")
- `EMBEDDING_MODEL`: "text-embedding-3-small"

**Paths:**
- `PAPERS_DIR`: "./data/papers"
- `IMAGES_DIR`: "./data/images"
- `VECTOR_DB_PATH`: "./data/vector_db"
- `OUTPUT_DIR`: "./outputs"
- `CAPABILITY_STORE_PATH`: "./data/capability_store" (NEW)

**Execution Settings:**
- `CODE_TIMEOUT_SECONDS`: 30
- `MAX_MEMORY_MB`: 512 (future use)
- `ALLOWED_IMPORTS`: ["numpy", "scipy", "matplotlib", "skimage"]

**RAG Settings:**
- `CHUNK_SIZE`: 1000 characters
- `CHUNK_OVERLAP`: 200 characters
- `TOP_K_CHUNKS`: 3

**Capability Store Settings:** (NEW)
- `CAPABILITY_SIMILARITY_THRESHOLD`: 0.85 (minimum similarity to reuse)
- `MAX_CAPABILITY_AGE_DAYS`: 90 (optional: expire old capabilities)
- `ENABLE_CAPABILITY_REUSE`: True (can disable for testing)

**Logging:**
- `LOG_LEVEL`: "INFO"
- `LOG_FILE`: "./pipeline.log"

### **Implementation Notes**
- Use `python-dotenv` to load `.env` file
- Validate required settings on import (fail fast if API key missing)
- Provide clear error messages for missing configuration

---

## **6. DATA MODELS** (`core/data_models.py`)

### **Purpose**
Define typed data structures for inter-component communication

### **Required Dataclasses**

**1. PreprocessedData**
```
Fields:
  - images: np.ndarray (shape: T×H×W, dtype: float32)
  - metadata: Dict[str, Any]
    - n_frames: int
    - height: int
    - width: int
    - normalized: bool
    - pixel_range: Tuple[float, float]
```

**2. RAGContext**
```
Fields:
  - chunks: List[str] (retrieved text chunks)
  - sources: List[str] (source document names)
  - scores: List[float] (similarity scores)
  - metadata: Dict[str, Any]
    - query: str
    - retrieval_time: float
```

**3. GeneratedCapability**
```
Fields:
  - code: str (Python code as text)
  - description: str (what the code does)
  - imports_used: List[str] (detected imports)
  - estimated_runtime: str (e.g., "fast", "medium")
```

**4. ExecutionResult**
```
Fields:
  - success: bool
  - results: Dict[str, Any] (output data from code)
  - figure: Any (matplotlib figure object or None)
  - execution_time: float (seconds)
  - error_message: str (if failed)
```

**5. AnalysisResult**
```
Fields:
  - data: Dict[str, Any] (numerical outputs)
  - figures: List[str] (paths to saved figures)
  - summary: str (natural language explanation)
  - code_used: str (for reproducibility)
  - metadata: Dict[str, Any]
    - request: str
    - timestamp: str
    - versions: Dict (library versions)
```

### **PipelineState (TypedDict for LangGraph)**
```
Fields:
  - user_request: str
  - images_path: str
  - preprocessed_data: Optional[PreprocessedData]
  - rag_context: Optional[RAGContext]
  - generated_code: Optional[str]
  - execution_results: Optional[ExecutionResult]
  - final_output: Optional[AnalysisResult]
  - errors: List[str]
  
  # NEW: Capability store fields
  - capability_reused: bool (whether code was reused)
  - capability_id: Optional[str] (ID of capability used/created)
  - capability_similarity: Optional[float] (similarity score if reused)
  - generated_capability: Optional[GeneratedCapability] (for saving new ones)
```

### **Implementation Notes**
- Use `@dataclass` from dataclasses module
- All dataclasses should have `__post_init__` validation where appropriate
- Provide `to_dict()` methods for serialization
- Use `Optional` typing where fields may be None during workflow

---

## **7. PREPROCESSOR** (`layers/preprocessor.py`)

### **Purpose**
Hard-coded data loading and normalization (not AI-driven)

### **Class: Preprocessor**

### **Method: process(images_path: str) -> PreprocessedData**

**Processing Steps:**
1. Discover all PNG files in directory (sorted alphanumerically)
2. Load each image using PIL/Pillow
3. Convert to grayscale if RGB (use .convert('L'))
4. Verify all images have identical dimensions
5. Stack into 3D numpy array (Time × Height × Width)
6. Convert to float32
7. Normalize pixel values to [0, 1] range: `images = images / 255.0`
8. Extract metadata
9. Return PreprocessedData object

**Error Handling:**
- Check directory exists and contains PNGs
- Raise ValueError if images have different dimensions
- Handle corrupted images gracefully (skip with warning)
- Log number of frames loaded

**Edge Cases:**
- Empty directory → raise ValueError with clear message
- Single image → still return 3D array with T=1
- Non-PNG files → ignore them

**Logging:**
- INFO: "Loading {n} frames from {path}"
- INFO: "Preprocessed data: {frames} frames, {H}×{W} pixels"
- WARNING: "Skipped corrupted image: {filename}"

### **Dependencies**
- PIL/Pillow for image loading
- numpy for array operations
- pathlib for file system operations

---

## **8. RAG SYSTEM** (`layers/rag_system.py`)

### **Purpose**
Retrieve relevant biological methods from papers using LangChain

### **Class: RAGSystem**

### **Constructor: __init__(papers_dir: str, vector_db_path: str)**

**Initialization Steps:**
1. Check if vector database already exists at `vector_db_path`
2. If exists: Load existing ChromaDB
3. If not exists: Build new vector database:
   - Load all .txt files from `papers_dir` using LangChain DirectoryLoader
   - Split documents using RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
   - Create embeddings using OpenAIEmbeddings
   - Store in ChromaDB at `vector_db_path` with `.persist()`
4. Create retriever with similarity search, k=3

**Error Handling:**
- Validate papers_dir exists and contains .txt files
- Handle OpenAI API failures gracefully
- Log embedding creation progress

### **Method: retrieve(query: str, top_k: int = 3) -> RAGContext**

**Retrieval Steps:**
1. Log incoming query
2. Use retriever to get relevant documents: `retriever.get_relevant_documents(query)`
3. Extract text chunks from returned documents
4. Extract source metadata (document filenames)
5. Extract similarity scores if available
6. Package into RAGContext object
7. Log number of chunks retrieved
8. Return RAGContext

**Optimization:**
- Cache retriever instance
- Don't re-embed papers on every query

**Logging:**
- INFO: "Building vector database from {n} papers"
- INFO: "Vector database loaded from {path}"
- INFO: "Retrieved {n} chunks for query: '{query}'"

### **LangChain Components Used**
- `DirectoryLoader` with `TextLoader` for loading papers
- `RecursiveCharacterTextSplitter` for chunking
- `OpenAIEmbeddings` for embeddings
- `Chroma` for vector storage
- `.as_retriever()` for query interface

### **Future Enhancement Notes**
- Add support for PDF loading (PyPDFLoader)
- Implement hybrid search (keyword + semantic)
- Add metadata filtering (e.g., by paper date)

---

## **8.5. TASK DECOMPOSER** (`layers/task_decomposer.py`)

### **Purpose**
Break complex analysis requests into atomic, reusable subtasks to maximize capability reuse and enable compositional analysis pipelines.

### **Class: TaskDecomposer**

### **Constructor: __init__(model: str = "gpt-4")**
- Store model name
- Initialize OpenAI client
- Load decomposition prompt template
- Set temperature=0.3 (slightly creative for task identification)

### **Method: decompose(user_request: str, rag_context: RAGContext, data_info: Dict) -> List[Subtask]**

**Purpose:** Transform a complex request into a sequence of atomic subtasks

**Decomposition Steps:**
1. Log incoming request
2. Build decomposition prompt (see template below)
3. Call OpenAI Chat Completions API
4. Parse response JSON containing subtask list
5. Validate subtask structure
6. Create Subtask objects for each decomposed task
7. Log number of subtasks generated
8. Return list of Subtask objects

**System Prompt Template:**
```
You are a task decomposition system for calcium imaging analysis.

GOAL: Break complex analysis requests into atomic, reusable subtasks that can be solved independently and composed into a complete pipeline.

PRINCIPLES:
1. Atomic: Each subtask should do ONE thing (load data, apply filter, detect peaks, etc.)
2. Reusable: Subtasks should be general enough to work in different contexts
3. Composable: Subtasks should chain together (output of task N is input to task N+1)
4. Self-contained: Each subtask should be solvable with available libraries

EXAMPLE DECOMPOSITIONS:

Request: "Segment cells and calculate their mean intensity over time"
Subtasks:
1. "Apply preprocessing filters to reduce noise"
2. "Segment cells using blob detection or watershed"
3. "Extract region properties for each segmented cell"
4. "Calculate mean intensity per cell across all time frames"

Request: "Detect calcium transients and measure their amplitude"
Subtasks:
1. "Calculate baseline fluorescence using percentile method"
2. "Compute ΔF/F (normalized fluorescence change)"
3. "Detect peaks in ΔF/F time series"
4. "Measure amplitude of each detected transient"

Request: "Count bright spots in the images"
Subtasks:
1. "Detect bright spots using blob detection"
2. "Count detected spots per frame"

GUIDELINES:
- Prefer 2-5 subtasks (not too fine-grained, not monolithic)
- Each subtask should map to ~20-50 lines of code
- Subtasks should follow data flow order
- Include data transformation steps (normalize, filter, etc.)
- Include computation steps (detect, measure, calculate)
- Avoid UI or visualization subtasks (handled separately)

OUTPUT FORMAT:
Return ONLY a JSON array of subtask objects:
[
  {
    "subtask_id": "subtask_1",
    "description": "Clear, actionable description",
    "input_variables": ["images"],
    "output_variables": ["filtered_images"],
    "dependencies": []
  },
  {
    "subtask_id": "subtask_2",
    "description": "Segment cells using watershed",
    "input_variables": ["filtered_images"],
    "output_variables": ["labels", "n_cells"],
    "dependencies": ["subtask_1"]
  }
]

Fields:
- subtask_id: Unique identifier (subtask_1, subtask_2, ...)
- description: Natural language task description (what the code should do)
- input_variables: List of variable names this subtask needs
- output_variables: List of variable names this subtask produces
- dependencies: List of subtask_ids that must complete first
```

**User Prompt Template:**
```
USER REQUEST:
{user_request}

SCIENTIFIC CONTEXT (from literature):
{rag_context.chunks[0]}

{rag_context.chunks[1]}

DATA SPECIFICATIONS:
- Shape: ({data_info['n_frames']}, {data_info['height']}, {data_info['width']})
- Type: Calcium imaging time-series
- Available input: `images` (numpy array, float32, range [0,1])

TASK:
Decompose the user's request into atomic subtasks following the principles above.
Each subtask should be solvable using: numpy, scipy, matplotlib, scikit-image
Return ONLY the JSON array of subtasks.
```

**Post-Processing:**
- Parse JSON response
- Validate JSON structure (has required fields)
- Check dependencies are valid (reference existing subtask_ids)
- Verify input/output variables form a valid chain
- Ensure first subtask takes "images" as input
- If parsing fails, retry once with clarifying prompt

**Error Handling:**
- OpenAI API timeout → retry once
- Invalid JSON → log error and retry with explicit formatting instructions
- Circular dependencies → raise ValueError
- Empty subtask list → treat as single atomic task

**Logging:**
- INFO: "Decomposing request: '{user_request}'"
- INFO: "Generated {n} subtasks"
- DEBUG: "Subtask chain: {subtask_1} → {subtask_2} → ... → {subtask_n}"
- WARNING: "Invalid decomposition, retrying: {error}"

### **Dataclass: Subtask** (Add to `core/data_models.py`)
```python
@dataclass
class Subtask:
    subtask_id: str
    description: str
    input_variables: List[str]
    output_variables: List[str]
    dependencies: List[str]

    # Filled by workflow
    code: Optional[str] = None  # Generated or reused code
    capability_id: Optional[str] = None  # If reused
    execution_result: Optional[Dict] = None
    reused: bool = False
```

---

## **9. CAPABILITY MANAGER** (`layers/capability_manager.py`)

### **Purpose**
Generate executable Python code using GPT-4/Claude with RAG context

### **Class: CapabilityManager**

### **Constructor: __init__(model: str = "gpt-4")**
- Store model name
- Initialize OpenAI client
- Load system prompt template
- Set temperature=0.2 (deterministic code generation)

### **Method: generate(user_request: str, rag_context: RAGContext, data_info: Dict) -> GeneratedCapability**

**Generation Steps:**
1. Log incoming request
2. Build system prompt (see template below)
3. Build user prompt using request + RAG context + data info
4. Call OpenAI Chat Completions API with messages
5. Extract response text
6. Parse code from markdown if present (handle ```python fences)
7. Validate code (check for forbidden operations)
8. Extract imports used
9. Package into GeneratedCapability object
10. Log code length and imports
11. Return GeneratedCapability

**System Prompt Template:**
```
You are a code generation system for calcium imaging analysis.

TASK: Generate complete, executable Python code based on user requests and scientific literature.

INPUT VARIABLES PROVIDED TO YOUR CODE:
- `images`: numpy.ndarray, shape (T, H, W), dtype float32, range [0, 1]
  - T = number of time frames
  - H, W = image height and width in pixels

OUTPUT REQUIREMENTS:
Your code must create two variables:
1. `results`: dict containing numerical outputs
   - Example: {'n_cells': 42, 'mean_intensity': 0.65}
2. `figure`: matplotlib figure object or None
   - Use plt.figure() to create visualizations

ALLOWED IMPORTS:
- numpy (as np)
- scipy (scipy.signal, scipy.ndimage)
- matplotlib.pyplot (as plt)
- skimage (skimage.measure, skimage.segmentation, skimage.filters)

CODE STYLE:
- Include docstring explaining biological context
- Add comments for non-obvious steps
- Use descriptive variable names
- Keep functions under 50 lines each
- Total code under 150 lines

SAFETY CONSTRAINTS:
- NO file I/O operations (no open, read, write)
- NO network calls (no requests, urllib)
- NO system commands (no os.system, subprocess)
- NO eval or exec
- NO infinite loops

RETURN FORMAT:
Return ONLY Python code inside a ```python code fence.
```

**User Prompt Template:**
```
USER REQUEST:
{user_request}

RELEVANT METHODS FROM SCIENTIFIC LITERATURE:
{rag_context.chunks[0]}

{rag_context.chunks[1]}

{rag_context.chunks[2]}

DATA SPECIFICATIONS:
- Shape: ({data_info['n_frames']}, {data_info['height']}, {data_info['width']})
- Type: Calcium imaging time-series
- Pixel range: [0.0, 1.0] (normalized)

TASK:
Generate Python code that accomplishes the user's request using the scientific methods described above.
Ensure the code creates both `results` dict and `figure` object as specified.
```

**Post-Processing:**
- Strip markdown fences: `code = code.split("```python")[1].split("```")[0]` if present
- Validate imports against allowed list
- Check for forbidden patterns: "open(", "os.system", "exec(", "eval("
- If forbidden code detected, raise ValueError with explanation

**Error Handling:**
- OpenAI API timeout → retry once
- API error → return error capability with explanatory code
- Invalid response format → log and raise clear error

**Logging:**
- INFO: "Generating code for: '{user_request}'"
- INFO: "Using {n} RAG chunks from sources: {sources}"
- INFO: "Generated {n} lines of code with imports: {imports}"
- WARNING: "Detected forbidden pattern: {pattern}"

---

## **9.5 CAPABILITY STORE** (`layers/capability_store.py`)

### **Purpose**
Store, version control, and retrieve generated capabilities to enable code reuse and continuous improvement

### **Class: CapabilityStore**

### **Constructor: __init__(store_path: str = "./data/capability_store")**

**Initialization Steps:**
1. Create store directory structure if not exists
2. Initialize or open Git repository at `store_path/.git`
3. Initialize ChromaDB for semantic search at `store_path/capability_db`
4. Create collection named "capabilities" with cosine similarity
5. Load existing capabilities into memory (optional index)

**Directory Structure Created:**
```
capability_store/
├── .git/                    # Git version control
├── capabilities/            # Generated code files
│   ├── cap_xxx.py          # Python scripts
│   └── cap_xxx.json        # Metadata
└── capability_db/          # ChromaDB for search
```

**Error Handling:**
- Create directories with proper permissions
- Handle existing git repo gracefully
- Log initialization progress

### **Method: save_capability(request: str, capability: GeneratedCapability, execution_result: ExecutionResult) -> str**

**Purpose:** Save newly generated capability with version control and indexing

**Steps:**
1. Generate unique capability ID:
   - Format: `cap_{timestamp}_{request_hash}`
   - Timestamp: `YYYYMMDD_HHMMSS`
   - Hash: First 6 chars of MD5(request)
2. Write Python code to `capabilities/{cap_id}.py`
3. Write metadata JSON to `capabilities/{cap_id}.json`:
   ```json
   {
     "request": "original user request",
     "created_at": "ISO timestamp",
     "imports": ["numpy", "scipy"],
     "success": true,
     "execution_time": 1.2,
     "reuse_count": 0,
     "last_used": null
   }
   ```
4. Git commit both files with message: `"Add capability: {request[:50]}\n\nID: {cap_id}"`
5. Index in ChromaDB:
   - Document: The user request text (embedded automatically)
   - Metadata: success, execution_time, created_at
   - ID: cap_id
6. Return capability ID

**Error Handling:**
- Check disk space before writing
- Handle git conflicts gracefully
- Log save success with capability ID

**Logging:**
- INFO: "Saving capability {cap_id} for request: '{request[:50]}'"
- INFO: "Git commit: {commit_hash}"
- INFO: "Indexed in vector database"

### **Method: search_similar(request: str, threshold: float = 0.85, top_k: int = 3) -> List[Dict]**

**Purpose:** Find existing capabilities similar to the request

**Steps:**
1. Query ChromaDB with request text
2. Set n_results = top_k
3. Filter by success=True (only return working capabilities)
4. Get results with distances
5. Convert distance to similarity: `similarity = 1 - distance`
6. Filter by threshold: keep only if similarity >= threshold
7. Sort by similarity (descending)
8. Return list of dicts:
   ```python
   {
     'cap_id': 'cap_20251007_143045_abc123',
     'similarity': 0.92,
     'metadata': {...},
     'request': 'original request text'
   }
   ```

**Similarity Interpretation:**
- 0.95-1.00: Nearly identical
- 0.85-0.95: Very similar (safe to reuse)
- 0.70-0.85: Somewhat similar (risky to reuse)
- <0.70: Different (don't reuse)

**Logging:**
- INFO: "Searching for similar capabilities to: '{request[:50]}'"
- INFO: "Found {n} capabilities above threshold {threshold}"
- DEBUG: "Top match: {cap_id} (similarity: {sim:.3f})"

### **Method: load_capability(cap_id: str) -> GeneratedCapability**

**Purpose:** Load capability code and metadata from storage

**Steps:**
1. Construct file paths
2. Check files exist, raise ValueError if not
3. Read Python code from `capabilities/{cap_id}.py`
4. Read metadata from `capabilities/{cap_id}.json`
5. Parse JSON
6. Construct and return GeneratedCapability object

**Error Handling:**
- FileNotFoundError → raise ValueError with clear message
- JSON parse error → log and raise
- Corrupted files → log error and attempt recovery from git history

**Logging:**
- INFO: "Loading capability {cap_id}"
- WARNING: "Capability {cap_id} not found"

### **Method: increment_reuse(cap_id: str)**

**Purpose:** Track capability reuse statistics

**Steps:**
1. Load metadata JSON
2. Increment `reuse_count` by 1
3. Update `last_used` to current timestamp
4. Write updated JSON back to file
5. Git commit: `"Reuse capability {cap_id} (count: {new_count})"`

**Logging:**
- INFO: "Capability {cap_id} reused (total: {count})"

### **Method: get_capability_stats(cap_id: str) -> Dict**

**Purpose:** Get usage statistics for a capability

**Returns:**
```python
{
  'cap_id': str,
  'request': str,
  'created_at': str,
  'reuse_count': int,
  'last_used': str or None,
  'success_rate': float,  # If tracking multiple executions
  'avg_execution_time': float
}
```

### **Method: list_all_capabilities(sort_by: str = 'created_at') -> List[Dict]**

**Purpose:** List all stored capabilities with metadata

**Parameters:**
- `sort_by`: 'created_at', 'reuse_count', or 'last_used'

**Use Cases:**
- Admin dashboard
- Capability browsing
- Identifying unused capabilities

### **Future Enhancement Methods (Optional):**

**Method: delete_capability(cap_id: str)**
- Remove from git (with commit)
- Remove from vector DB
- Archive instead of delete (safer)

**Method: update_capability(cap_id: str, new_code: str, reason: str)**
- Save as new version
- Update git with clear diff
- Maintain version history

**Method: get_capability_history(cap_id: str) -> List[Dict]**
- Use git log to show all versions
- Return list of commits affecting this capability

---

## **9.6. CAPABILITY COMPOSER** (`layers/capability_composer.py`)

### **Purpose**
Chain atomic capabilities into a complete analysis pipeline by managing variable flow and namespace composition.

### **Class: CapabilityComposer**

### **Method: compose(subtasks: List[Subtask]) -> str**

**Purpose:** Generate executable Python code that chains all subtask codes together

**Composition Steps:**
1. Validate subtask dependency graph (topological sort)
2. Extract code from each subtask (generated or reused)
3. Build namespace initialization
4. Chain code blocks in dependency order
5. Add variable passing logic between subtasks
6. Add final results aggregation
7. Validate composed code syntax
8. Return complete executable pipeline code

**Composition Template:**
```python
"""
Composed analysis pipeline
Generated from {n} atomic capabilities

Subtask chain:
{subtask_1.description}
→ {subtask_2.description}
→ {subtask_3.description}
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from skimage import measure, filters, segmentation

# ===== Subtask 1: {subtask_1.description} =====
# Capability ID: {subtask_1.capability_id or "newly_generated"}
# Input: {subtask_1.input_variables}
# Output: {subtask_1.output_variables}

{subtask_1.code}

# ===== Subtask 2: {subtask_2.description} =====
# Capability ID: {subtask_2.capability_id or "newly_generated"}
# Input: {subtask_2.input_variables}
# Output: {subtask_2.output_variables}

{subtask_2.code}

# ===== Subtask 3: {subtask_3.description} =====
# ...

{subtask_3.code}

# ===== Aggregate Results =====
results = {
    'subtask_1_output': {subtask_1.output_variables[0]},
    'subtask_2_output': {subtask_2.output_variables[0]},
    'subtask_3_output': {subtask_3.output_variables[0]},
    'pipeline_steps': {n},
    'capabilities_reused': {count_reused},
    'capabilities_generated': {count_generated}
}

figure = None  # Individual subtasks may create figures
```

**Variable Flow Management:**
- Each subtask expects certain input variables
- Previous subtasks must produce those variables
- Validate variable chain before composition
- If mismatch detected, log error and attempt repair

**Namespace Isolation:**
- Each subtask code block gets a comment header
- Variable scope is shared across all subtasks
- No explicit function wrapping (direct sequential execution)
- Variables accumulate in shared namespace

**Error Handling:**
- Circular dependencies → raise ValueError before composition
- Missing input variable → log warning and attempt to infer
- Duplicate output variables → use last defined value
- Syntax errors in individual codes → caught during validation

**Validation:**
- Check composed code is syntactically valid (compile test)
- Verify all required imports are included
- Ensure `results` dict is created at the end
- Confirm no forbidden operations introduced during composition

**Logging:**
- INFO: "Composing {n} subtasks into pipeline"
- INFO: "Capability reuse: {reused}/{total} subtasks"
- DEBUG: "Variable flow: images → {var1} → {var2} → results"
- WARNING: "Variable mismatch detected between subtask_{i} and subtask_{j}"

### **Method: validate_dependencies(subtasks: List[Subtask]) -> bool**

**Purpose:** Ensure subtask dependency graph is valid and executable

**Validation Checks:**
1. No circular dependencies (topological sort succeeds)
2. All dependencies reference valid subtask_ids
3. Input variables are satisfied by previous outputs
4. First subtask has "images" as input
5. Dependency order matches data flow

**Returns:**
- True if valid
- Raises ValueError with explanation if invalid

**Logging:**
- INFO: "Dependency graph validated successfully"
- ERROR: "Circular dependency detected: {cycle}"
- ERROR: "Subtask {id} requires {var} but no subtask produces it"

### **Method: repair_variable_chain(subtasks: List[Subtask]) -> List[Subtask]** (Optional, Future)

**Purpose:** Attempt to fix common variable flow issues

**Repair Strategies:**
- Add variable aliasing code between subtasks
- Insert data transformation subtasks
- Rename variables for consistency

**Note:** This is an advanced feature for post-POC enhancement

---

## **10. CODE EXECUTOR** (`core/executor.py`)

### **Purpose**
Safely execute generated Python code with timeout and import restrictions

### **Function: execute_code(code: str, images: np.ndarray, timeout: int = 30) -> ExecutionResult**

**Execution Strategy:**
1. Create isolated namespace dictionary:
   - Add `images` variable
   - Pre-import allowed modules: numpy, scipy, matplotlib, skimage
   - Initialize empty `results = {}` and `figure = None`
2. Set up timeout mechanism (use signal.alarm on Unix or threading.Timer)
3. Execute code: `exec(code, namespace)`
4. Measure execution time
5. Extract `results` and `figure` from namespace
6. Return ExecutionResult with success=True

**Namespace Setup:**
```python
namespace = {
    '__builtins__': {
        # Whitelist safe builtins
        'print': print,
        'range': range,
        'len': len,
        'enumerate': enumerate,
        'zip': zip,
        'map': map,
        'filter': filter,
        'sum': sum,
        'min': min,
        'max': max,
        'abs': abs,
        'round': round,
        'sorted': sorted,
        'list': list,
        'dict': dict,
        'tuple': tuple,
        'set': set,
        'str': str,
        'int': int,
        'float': float,
        'bool': bool,
    },
    'np': numpy,
    'numpy': numpy,
    'plt': matplotlib.pyplot,
    'matplotlib': matplotlib,
    'scipy': scipy,
    'skimage': skimage,
    'images': images,
    'results': {},
    'figure': None,
}
```

**Timeout Implementation:**
```
Unix/Linux: Use signal.alarm
- Set alarm before exec()
- Reset alarm after exec()
- Catch SIGALRM exception

Windows/Cross-platform: Use threading.Timer
- Create timer thread that raises exception
- Cancel timer after successful execution
```

**Error Handling:**
- Timeout → return ExecutionResult with success=False, error="Execution timeout"
- SyntaxError → return ExecutionResult with success=False, error="Syntax error: {details}"
- RuntimeError → return ExecutionResult with success=False, error="Runtime error: {details}"
- Any exception → catch, log, return ExecutionResult with error message

**Validation After Execution:**
- Check `results` is a dict
- Check `figure` is matplotlib.figure.Figure or None
- If validation fails, return error ExecutionResult

**Logging:**
- INFO: "Executing generated code ({n} lines)"
- INFO: "Execution completed in {time:.2f}s"
- ERROR: "Execution failed: {error_type}: {error_message}"

**Security Notes (POC Level):**
- This is NOT production-grade security
- Suitable for trusted single-user development
- For production, use Docker containers or E2B service
- Document these limitations clearly

---

## **11. LANGGRAPH WORKFLOW** (`graph/workflow.py`)

### **Purpose**
Define state machine that orchestrates all components with task decomposition and atomic capability reuse

### **Workflow Structure**

**Nodes (Functions):**
1. `preprocess_node(state: PipelineState) -> PipelineState`
2. `rag_retrieval_node(state: PipelineState) -> PipelineState`
3. `task_decomposition_node(state: PipelineState) -> PipelineState` **(NEW)**
4. `subtask_resolution_node(state: PipelineState) -> PipelineState` **(NEW)**
5. `composition_node(state: PipelineState) -> PipelineState` **(NEW)**
6. `execution_node(state: PipelineState) -> PipelineState`
7. `format_output_node(state: PipelineState) -> PipelineState`

**Edges (Transitions):**
```
START
  → preprocess_node (Load and normalize images)
  → rag_retrieval_node (Get scientific context)
  → task_decomposition_node (Break into atomic subtasks)
  → subtask_resolution_node (For each subtask: search/generate/save atomic capability)
  → composition_node (Chain atomic capabilities into pipeline)
  → execution_node (Run composed pipeline)
  → format_output_node (Package results)
  → END
```

**Note:** The workflow now decomposes complex requests first, then resolves each subtask independently, enabling maximum code reuse at the atomic level.

### **Node Implementations**

**preprocess_node:**
- Call Preprocessor().process(state["images_path"])
- Store result in state["preprocessed_data"]
- Log success
- Handle errors: append to state["errors"]
- Return updated state

**rag_retrieval_node:**
- Call RAGSystem().retrieve(state["user_request"])
- Store result in state["rag_context"]
- Log retrieved chunks count
- Handle errors: append to state["errors"]
- Return updated state

**task_decomposition_node:**
- Extract data_info from state["preprocessed_data"].metadata
- Call TaskDecomposer().decompose(state["user_request"], state["rag_context"], data_info)
- Store result in state["subtasks"] (list of Subtask objects)
- Log: "Decomposed request into {n} atomic subtasks"
- Log subtask chain for debugging
- Handle errors: append to state["errors"], treat as single task if decomposition fails
- Return updated state

**subtask_resolution_node:**
- Initialize CapabilityStore
- Initialize atomic_capabilities dict in state
- For each subtask in state["subtasks"]:
  1. **Search for existing atomic capability:**
     - Call `store.search_similar(subtask.description, threshold=0.85)`
     - If found (similarity >= threshold):
       - Load capability: `cap = store.load_capability(cap_id)`
       - Set subtask.code = cap.code
       - Set subtask.capability_id = cap_id
       - Set subtask.reused = True
       - Call `store.increment_reuse(cap_id)`
       - Log: "Reusing capability {cap_id} for subtask '{subtask.description}'"
     - Else (no match):
       - Set subtask.reused = False
       - Log: "No match for subtask '{subtask.description}', will generate"

  2. **Generate new atomic capability if not reused:**
     - If subtask.reused is False:
       - Call CapabilityManager().generate(subtask.description, state["rag_context"], data_info)
       - Store generated code in subtask.code
       - Mark for later saving (after successful composition + execution)
       - Log: "Generated new code for subtask '{subtask.description}'"

  3. **Store in atomic_capabilities dict:**
     - state["atomic_capabilities"][subtask.subtask_id] = subtask

- Calculate reuse statistics:
  - state["capabilities_reused_count"] = count where subtask.reused == True
  - state["capabilities_generated_count"] = count where subtask.reused == False
- Log: "Resolved {n} subtasks: {reused} reused, {generated} generated"
- Handle errors: append to state["errors"]
- Return updated state

**composition_node:** **(NEW)**
- Call CapabilityComposer().compose(state["subtasks"])
- Store composed code in state["composed_code"]
- Set state["generated_code"] = state["composed_code"] (for execution)
- Log: "Composed {n} atomic capabilities into pipeline"
- Log reuse statistics
- Handle errors: append to state["errors"]
- Return updated state

**execution_node:**
- Call execute_code(state["generated_code"], state["preprocessed_data"].images)
- Store result in state["execution_results"]
- Log execution time and success
- Handle errors: append to state["errors"]
- Return updated state

**save_atomic_capabilities_node:**
- For each subtask in state["subtasks"]:
  - If subtask.reused is False AND state["execution_results"].success is True:
    - Extract relevant execution info for this subtask
    - Call `store.save_capability(subtask.description, subtask.code, execution_info)`
    - Set subtask.capability_id = returned cap_id
    - Log: "Saved new atomic capability {cap_id}"
- Return updated state

**format_output_node:**
- Package execution results into AnalysisResult
- Save figures to output directory
- Generate natural language summary (optional: GPT-4 call)
- Include capability metadata in output:
  - Number of atomic capabilities used
  - Reuse vs generation count
  - List of capability_ids involved
- Store in state["final_output"]
- Return updated state

### **Graph Construction**

```python
1. Create StateGraph with PipelineState type
2. Add all nodes using .add_node():
   - preprocess_node
   - rag_retrieval_node
   - task_decomposition_node (NEW)
   - subtask_resolution_node (NEW)
   - composition_node (NEW)
   - execution_node
   - format_output_node
   - save_atomic_capabilities_node (OPTIONAL - can integrate into subtask_resolution)
3. Set entry point: .set_entry_point("preprocess_node")
4. Add edges:
   - .add_edge("preprocess_node", "rag_retrieval_node")
   - .add_edge("rag_retrieval_node", "task_decomposition_node")
   - .add_edge("task_decomposition_node", "subtask_resolution_node")
   - .add_edge("subtask_resolution_node", "composition_node")
   - .add_edge("composition_node", "execution_node")
   - .add_edge("execution_node", "save_atomic_capabilities_node")
   - .add_edge("save_atomic_capabilities_node", "format_output_node")
   - .add_edge("format_output_node", END)
5. Compile graph: app = workflow.compile()

Note: subtask_resolution_node contains internal loop logic to process
each subtask independently (search → generate if needed → store).
```

**Key Architectural Change:**
- **Old workflow:** Monolithic capability (one request = one big code blob)
- **New workflow:** Atomic capabilities (one request = N small reusable code units)
- **Benefit:** High reuse rate as different complex requests share common subtasks

**Example:**
```
Request 1: "Segment cells and count them"
→ Subtasks: ["Segment cells", "Count segmented regions"]
→ Both generated (0% reuse)

Request 2: "Segment cells and measure intensity"
→ Subtasks: ["Segment cells", "Measure mean intensity per region"]
→ 50% reuse (segment cells reused, measure intensity generated)

Request 3: "Count bright spots and measure intensity"
→ Subtasks: ["Detect bright spots", "Measure mean intensity per region"]
→ 50% reuse (measure intensity reused, detect spots may reuse segment cells)

After 10 requests:
→ 70-80% of subtasks are reused
→ System is 5-10x faster on average
→ Cost reduced by 60-80%
```

### **Execution**

```python
Function: run_workflow(user_request: str, images_path: str) -> AnalysisResult

Steps:
1. Initialize PipelineState with user_request and images_path
2. Invoke compiled graph: app.invoke(initial_state)
3. Extract final_output from returned state
4. Return AnalysisResult
```

### **Error Handling Strategy**
- Each node catches its own errors
- Errors appended to state["errors"] list
- Workflow continues to END even with errors
- Final node checks errors list and generates appropriate output

### **Logging**
- Each node logs entry and exit
- Log state transitions
- Log execution time per node
- Aggregate logs show full workflow trace

---

## **12. MAIN ENTRY POINT** (`main.py`)

### **Purpose**
CLI interface to run full pipeline

### **Command-Line Arguments**

**Required:**
- `--request`: User's analysis request (string)
- `--images`: Path to directory containing PNG frames

**Optional:**
- `--output`: Output directory (default: ./outputs)
- `--verbose`: Enable debug logging
- `--rebuild-rag`: Force rebuild vector database

**Example Usage:**
```bash
python main.py --request "Segment cells and count them" --images ./data/images

python main.py --request "Calculate mean intensity over time" --images ./custom/path --verbose

python main.py --request "Detect calcium transients" --images ./data/images --rebuild-rag
```

### **Main Function Logic**

```
1. Parse command-line arguments
2. Load configuration from config.py
3. Set up logging based on --verbose flag
4. Initialize RAG system (rebuild if --rebuild-rag flag set)
5. Create initial state from arguments
6. Run workflow via run_workflow()
7. Save results to output directory
8. Print summary to console
9. Handle exceptions and print user-friendly errors
```

### **Output Format**

**Console:**
```
=== CALCIUM IMAGING ANALYSIS ===
Request: Segment cells and count them
Images: 5 frames loaded (128x128 pixels)

[Step 1/7] Preprocessing... ✓
[Step 2/7] Retrieving methods... ✓ (3 chunks)
[Step 3/7] Searching capabilities... ✓ (reusing cap_20251007_143045, similarity: 0.91)
[Step 4/7] Code generation... ⊘ (skipped - reused existing)
[Step 5/7] Save capability... ⊘ (skipped - already exists)
[Step 6/7] Executing analysis... ✓ (0.8s)
[Step 7/7] Formatting output... ✓

RESULTS:
- n_cells: 42
- mean_cell_area: 156.3 pixels
- capability_reused: true
- capability_id: cap_20251007_143045

Figures saved to: ./outputs/2025-10-07_14-30-45/
Full report: ./outputs/2025-10-07_14-30-45/report.json

💡 Tip: This request reused existing code. 10x faster than generating new!
```

**Alternative (new capability):**
```
[Step 3/7] Searching capabilities... ✓ (no match found)
[Step 4/7] Code generation... ✓ (45 lines)
[Step 5/7] Save capability... ✓ (saved as cap_20251007_150000)
[Step 6/7] Executing analysis... ✓ (1.2s)

💡 Tip: New capability created! Future similar requests will be faster.
```

**File Outputs:**
```
./outputs/{timestamp}/
  ├── report.json          # Full AnalysisResult as JSON
  ├── generated_code.py    # Code that was executed
  ├── figure_001.png       # Generated visualizations
  └── logs.txt             # Detailed execution log
```

---

## **13. MOCK DATA CREATION**

### **Mock Papers** (`data/papers/*.txt`)

**File 1: segmentation_methods.txt**
```
=== Cell Segmentation in Calcium Imaging ===

Watershed Segmentation:
The watershed algorithm is commonly used for segmenting cells in calcium imaging data.
Process: Apply distance transform to binary image, find local maxima as seeds, then use watershed.
Library: skimage.segmentation.watershed
Parameters: minimum_distance between seeds typically 5-10 pixels depending on cell size.

Blob Detection:
Alternative approach using Laplacian of Gaussian (LoG) blob detection.
Library: skimage.feature.blob_log
Parameters: min_sigma=1, max_sigma=10, threshold=0.1

Region Properties:
After segmentation, extract cell properties using regionprops.
Library: skimage.measure.regionprops
Metrics: area, mean_intensity, centroid, eccentricity
```

**File 2: transient_detection.txt**
```
=== Calcium Transient Detection Methods ===

Bandpass Filtering:
Apply temporal filtering to remove slow drift and high-frequency noise.
Recommended: Butterworth bandpass filter 0.1-5 Hz for typical calcium indicators.
Library: scipy.signal.butter

Peak Detection:
Use scipy.signal.find_peaks to identify transient events.
Parameters:
- prominence: Set to 2× median absolute deviation of baseline
- distance: Minimum 5 frames between peaks (prevents double-counting)
- height: Above baseline + 3× standard deviation

Event Characterization:
For each detected transient, measure:
- Amplitude: Peak height above baseline
- Rise time: Time from 10% to 90% of peak
- Decay time: Time from peak to 50% recovery
```

**File 3: baseline_calculation.txt**
```
=== Baseline and ΔF/F Calculation ===

Baseline Estimation:
Method 1: Percentile-based
baseline = np.percentile(trace, 10)
Rationale: 10th percentile captures low-activity periods

Method 2: Rolling minimum
baseline = scipy.ndimage.minimum_filter1d(trace, size=window)
Typical window: 100-200 frames

ΔF/F Calculation:
Formula: dff = (trace - baseline) / baseline
Where F = fluorescence trace, baseline = F0

Interpretation:
ΔF/F represents fractional change in fluorescence
Values typically range 0.1-3.0 for calcium transients
Larger values indicate stronger calcium influx
```

### **Mock Images** (`data/images/*.png`)

**Generation Strategy:**
Create 5-10 synthetic grayscale PNGs (128×128 pixels) simulating calcium imaging:

**Method: Using numpy + PIL**
```python
For each frame:
1. Create blank 128×128 array
2. Add 10-20 Gaussian blobs (simulated cells):
   - Random positions
   - Radius 3-5 pixels
   - Intensity varies over time (simulate calcium transients)
3. Add Gaussian noise (sigma=0.05)
4. Normalize to 0-255 range
5. Save as PNG using PIL
```

**Temporal Variation:**
- Frame 1-5: Some cells increase in intensity (simulated transients)
- Frame 6-10: Intensity returns to baseline
- This creates realistic data for transient detection

**Implementation Note:**
Can be created with a simple script or manually using image editing software.
Must be grayscale, same dimensions, sequential naming (frame_001.png, frame_002.png, etc.).

---

## **14. TESTING STRATEGY**

### **Unit Tests** (`tests/`)

**test_rag.py:**
- Test vector database creation from papers
- Test retrieval with known queries
- Verify top_k returns correct number of chunks
- Test handling of empty/invalid queries

**test_executor.py:**
- Test execution of valid simple code
- Test timeout functionality (code with infinite loop)
- Test forbidden import detection
- Test namespace isolation

**test_preprocessor.py:**
- Test loading valid PNG sequence
- Test error handling for mismatched dimensions
- Test normalization correctness
- Test metadata extraction

**test_capability_store.py:** **(NEW)**
- Test git repo initialization
- Test saving capability (verify git commit created)
- Test searching for similar capabilities
- Test loading capability by ID
- Test reuse count increment
- Test handling of non-existent capability ID
- Test similarity threshold filtering

### **Integration Test** (`tests/test_workflow.py`)

**Test Case: Full Pipeline with Simple Request**
```python
Request: "Count the number of bright spots in the images"

Expected Behavior:
1. Workflow starts successfully
2. Preprocess node loads images
3. RAG node retrieves segmentation methods
4. Code generation node creates valid code using blob detection
5. Execution node runs code successfully
6. Results contain: {'n_spots': <integer>}
7. No errors in state["errors"]

Assertions:
- final_output is not None
- final_output.data contains 'n_spots'
- execution_results.success == True
- All nodes logged their execution
```

**Test Case: Error Handling**
```python
Request: Valid request, but invalid images path

Expected Behavior:
- Preprocess node fails
- Error appended to state["errors"]
- Workflow completes (doesn't crash)
- Final output indicates error occurred

Assertions:
- state["errors"] is not empty
- Error message is descriptive
```

**Test Case: Capability Reuse** **(NEW)**
```python
# First run
Request 1: "Count the number of cells in the images"

Expected Behavior:
1. Workflow starts successfully
2. Preprocess and RAG nodes execute
3. Capability search finds no match
4. Code generation creates new code
5. Save capability node stores code
6. Execution succeeds
7. state["capability_reused"] == False
8. state["capability_id"] is set

# Second run
Request 2: "Count cells in the images"  # Similar query

Expected Behavior:
1. Workflow starts successfully
2. Preprocess and RAG nodes execute
3. Capability search finds match (similarity > 0.85)
4. Code generation is skipped
5. Save capability node is skipped
6. Execution uses reused code
7. state["capability_reused"] == True
8. state["capability_similarity"] >= 0.85
9. Execution time is faster (no GPT-4 call)

Assertions:
- Both runs succeed
- Second run has capability_reused=True
- Capability git repo has 1 commit (not 2)
- Reuse count in metadata is incremented
```

---

## **15. IMPLEMENTATION ORDER**

### **Phase 1: Foundation** (Day 1)
**Deliverable**: Data models and configuration working

1. Set up project structure (create all directories)
2. Create requirements.txt and install dependencies
3. Implement config.py with environment variable loading
4. Implement data_models.py with all dataclasses
5. Create .env.example file
6. **Test**: Import all modules successfully

### **Phase 2: Preprocessor** (Day 1)
**Deliverable**: Can load images successfully

1. Implement Preprocessor class
2. Create mock images (5 synthetic PNGs)
3. **Test**: Load images and verify output shape/type
4. Add error handling and logging
5. **Test**: Handles missing files gracefully

### **Phase 3: Mock Papers** (Day 1-2)
**Deliverable**: RAG knowledge base ready

1. Write 3 mock paper files (segmentation, transients, baseline)
2. Ensure each file has 3-4 distinct chunks of information
3. **Test**: Files are readable and well-formatted

### **Phase 4: RAG System** (Day 2)
**Deliverable**: Can retrieve relevant paper chunks

1. Implement RAGSystem class using LangChain
2. Initialize with mock papers, create vector database
3. Test retrieval with sample queries
4. **Test**: Retrieve "segmentation" → returns segmentation_methods.txt chunks
5. **Test**: Retrieve "transient" → returns transient_detection.txt chunks
6. Add logging for all operations

### **Phase 5: Code Generator** (Day 2-3)
**Deliverable**: Can generate valid Python code

1. Implement CapabilityManager class
2. Create system and user prompt templates
3. **Test**: Generate code for "count cells" request
4. **Test**: Generated code is syntactically valid
5. Add import validation
6. **Test**: Detects forbidden imports/operations
7. Refine prompts based on output quality

### **Phase 5.5: Capability Store** (Day 3)
**Deliverable**: Can save and retrieve atomic capabilities

1. Implement CapabilityStore class (modified for atomic capabilities)
2. Set up git repository initialization
3. Set up ChromaDB for semantic search
4. Implement save_capability() method (for atomic subtasks)
5. Implement search_similar() method
6. Implement load_capability() method
7. Implement increment_reuse() method
8. **Test**: Save an atomic capability and verify git commit
9. **Test**: Search for similar subtask and verify results
10. **Test**: Load capability and verify code matches

### **Phase 5.6: Task Decomposer** (Day 3-4)
**Deliverable**: Can decompose complex requests into atomic subtasks

1. Implement TaskDecomposer class
2. Create decomposition prompt template
3. Implement decompose() method with JSON parsing
4. Add Subtask dataclass to data_models.py
5. **Test**: Decompose "Segment cells and count them" → verify 2 subtasks
6. **Test**: Decompose "Detect transients and measure amplitude" → verify subtask chain
7. **Test**: Validate dependency graph (no circular dependencies)
8. Add retry logic for invalid JSON responses
9. Add validation for input/output variable chains

### **Phase 5.7: Capability Composer** (Day 4) **(NEW)**
**Deliverable**: Can chain atomic capabilities into executable pipeline

1. Implement CapabilityComposer class
2. Implement compose() method with code chaining
3. Implement validate_dependencies() method
4. Add topological sort for subtask ordering
5. **Test**: Compose 2 subtasks → verify variable flow
6. **Test**: Compose 4 subtasks with dependencies → verify execution order
7. **Test**: Validate composed code syntax
8. Add error handling for missing variables
9. Add aggregation of results from multiple subtasks

### **Phase 6: Code Executor** (Day 3-4)
**Deliverable**: Can safely run generated code

1. Implement execute_code function
2. Set up namespace with restricted builtins
3. Implement timeout mechanism
4. **Test**: Execute simple valid code successfully
5. **Test**: Timeout triggers on infinite loop
6. **Test**: Namespace isolation (can't access filesystem)
7. Add result validation

### **Phase 7: LangGraph Workflow** (Day 4-5)
**Deliverable**: Full pipeline executes end-to-end with task decomposition and atomic capability reuse

1. Update PipelineState TypedDict in graph/state.py:
   - Add subtasks: List[Subtask]
   - Add atomic_capabilities: Dict[str, Subtask]
   - Add composed_code: str
   - Add capabilities_reused_count and capabilities_generated_count
2. Implement all node functions in graph/nodes.py:
   - preprocess_node
   - rag_retrieval_node
   - task_decomposition_node **(NEW)**
   - subtask_resolution_node **(NEW - contains loop)**
   - composition_node **(NEW)**
   - execution_node
   - save_atomic_capabilities_node **(NEW)**
   - format_output_node (updated to show reuse stats)
3. Build workflow graph in graph/workflow.py
4. Add edge definitions (7-8 nodes)
5. Compile and test graph structure
6. **Test**: Run empty state through graph (should not crash)
7. **Test**: Workflow with simple request (single subtask, generates and saves)
8. **Test**: Workflow with complex request (multiple subtasks, all generated)
9. **Test**: Workflow with similar request (high atomic reuse rate)
10. Add error handling to each node
11. Add logging for decomposition and composition steps

### **Phase 8: Integration** (Day 4)
**Deliverable**: Complete system working

1. Connect all components in workflow nodes
2. Implement run_workflow function
3. **Test**: Full pipeline with request "count bright spots"
4. Debug any issues in component integration
5. Verify state flows correctly through all nodes
6. Add comprehensive logging

### **Phase 9: Main Entry Point** (Day 4-5)
**Deliverable**: User can run from command line

1. Implement main.py with argparse
2. Add console output formatting
3. Implement output directory creation and saving
4. **Test**: Run from command line with various requests
5. Add user-friendly error messages
6. Create usage examples in README

### **Phase 10: Testing & Documentation** (Day 5)
**Deliverable**: Polished POC ready for demonstration

1. Write integration test in tests/test_workflow.py
2. Write unit tests for critical components
3. Run all tests and fix failures
4. Document any limitations or known issues
5. Create README with setup instructions
6. Test on fresh Python environment
7. Document next steps and production considerations

---

## **16. LOGGING ARCHITECTURE**

### **Log Format**
```
[TIMESTAMP] [COMPONENT] [LEVEL] Message
[2025-10-07 14:30:45] [PREPROCESS] [INFO] Loading 5 frames from ./data/images
[2025-10-07 14:30:46] [RAG] [INFO] Retrieved 3 chunks for query: 'segment cells'
[2025-10-07 14:30:47] [CAP_SEARCH] [INFO] Searching for similar capabilities
[2025-10-07 14:30:47] [CAP_SEARCH] [INFO] Found capability cap_xxx (similarity: 0.92)
[2025-10-07 14:30:47] [CAP_SEARCH] [INFO] Reusing existing capability, skipping generation
[2025-10-07 14:30:48] [EXECUTOR] [INFO] Execution completed in 0.8s
[2025-10-07 14:30:49] [WORKFLOW] [INFO] Pipeline completed (capability reused)
```

**Alternative flow (new capability):**
```
[2025-10-07 14:30:47] [CAP_SEARCH] [INFO] No similar capability found
[2025-10-07 14:30:48] [CODEGEN] [INFO] Generated 45 lines of code
[2025-10-07 14:30:49] [CAP_SAVE] [INFO] Saved capability cap_yyy (commit: abc123)
[2025-10-07 14:30:50] [EXECUTOR] [INFO] Execution completed in 1.2s
[2025-10-07 14:30:51] [WORKFLOW] [INFO] Pipeline completed (new capability created)
```

### **Log Levels by Component**

**INFO:**
- Component initialization
- Major state transitions
- Successful completions
- Input/output summaries

**WARNING:**
- Skipped corrupted files
- Degraded functionality
- Automatic fallbacks

**ERROR:**
- Component failures
- API errors
- Execution errors
- Validation failures

### **Implementation**
- Use Python's built-in logging module
- Configure in utils/logging_config.py
- Log to both console and file
- In verbose mode, log DEBUG level
- Each component gets named logger (e.g., `logger = logging.getLogger(__name__)`)

---

## **17. EXAMPLE WORKFLOWS**

### **Test Case 1: Cell Counting**
```
Input:
  Request: "Count the number of cells in the images"
  Images: 5 frames with ~15 cells each

Expected Flow:
  L3 → Load 5 frames
  L2 → Retrieve segmentation methods (watershed, blob detection)
  L4 → Generate code using skimage.feature.blob_log
  EXEC → Run code, detect ~15 blobs per frame
  OUT → {"n_cells_per_frame": [15, 14, 16, 15, 15], "mean_n_cells": 15.0}

Success Criteria:
  - Code runs without errors
  - Results contain cell counts
  - Figure shows detected cells overlaid on image
```

### **Test Case 2: Intensity Tracking**
```
Input:
  Request: "Calculate mean intensity over time for each cell"
  Images: 5 frames with intensity variations

Expected Flow:
  L3 → Load 5 frames
  L2 → Retrieve segmentation + intensity measurement methods
  L4 → Generate code: segment → regionprops → extract mean_intensity
  EXEC → Run code, generate time series for each ROI
  OUT → {"roi_traces": array(n_cells × n_frames), "mean_trace": array(n_frames)}

Success Criteria:
  - Time series data in results
  - Figure shows intensity plots
  - No execution errors
```

### **Test Case 3: Transient Detection**
```
Input:
  Request: "Detect calcium transients and measure their amplitude"
  Images: 10 frames with simulated transients

Expected Flow:
  L3 → Load 10 frames
  L2 → Retrieve transient detection + baseline methods
  L4 → Generate code: calculate ΔF/F → find_peaks → measure amplitude
  EXEC → Run code, detect peaks
  OUT → {"n_transients": 8, "mean_amplitude": 0.35, "peak_frames": [2,3,5,7,...]}

Success Criteria:
  - Transient detection works
  - Amplitudes are reasonable
  - Figure shows traces with detected peaks marked
```

### **Test Case 4: Capability Reuse** **(NEW)**
```
# Run 1
Input:
  Request: "Segment cells and count them"
  Images: 5 frames with ~15 cells each

Expected Flow:
  L3 → Load 5 frames
  L2 → Retrieve segmentation methods
  SEARCH → No similar capability found
  L4 → Generate new code using blob detection
  SAVE → Store capability (cap_20251007_143045)
  EXEC → Run code, detect ~15 cells
  OUT → {"n_cells": 15, "capability_id": "cap_20251007_143045", "reused": false}

Time: 3.5 seconds (includes GPT-4 generation)

# Run 2 (similar request)
Input:
  Request: "Count the cells in these images"
  Images: 5 frames with ~15 cells each

Expected Flow:
  L3 → Load 5 frames
  L2 → Retrieve segmentation methods
  SEARCH → Found similar capability (similarity: 0.91)
  LOAD → Load cap_20251007_143045
  SKIP → Skip L4 generation
  SKIP → Skip save (already exists)
  EXEC → Run reused code, detect ~15 cells
  OUT → {"n_cells": 15, "capability_id": "cap_20251007_143045", "reused": true}

Time: 0.5 seconds (10x faster - no GPT-4 call)

Success Criteria:
  - Second run correctly identifies similar request
  - Reused code produces same quality results
  - Significant time savings on second run
  - Git repo shows only 1 capability commit (not 2)
  - Metadata shows reuse_count = 1
```

---

## **18. KNOWN LIMITATIONS (POC SCOPE)**

**L2 RAG:**
- Uses simple text chunking (no semantic section detection)
- No metadata filtering (can't filter by paper date, author)
- No citation tracking (doesn't preserve reference information)
- No hybrid search (semantic only, no keyword boost)

**L4 Code Generation:**
- May generate incorrect code for complex requests
- No iterative refinement (can't retry failed code)
- No code optimization
- Limited domain knowledge (depends on GPT-4 training)

**Capability Store:** **(NEW)**
- Basic similarity threshold (no ML-based matching)
- No automatic capability versioning/updates
- No conflict resolution if similar capabilities diverge
- Limited metadata tracking (no performance trends over time)
- Git-based versioning may not scale to 1000s of capabilities

**Execution:**
- POC-level security only (not production-safe)
- No resource monitoring (CPU/memory limits not enforced)
- No parallel execution (runs sequentially)
- Capabilities are not validated before reuse (assumes past success = future success)

**Workflow:**
- Linear with simple conditional branching
- No human-in-the-loop validation
- No multi-turn conversation
- No ability to ask clarifying questions
- No automatic capability improvement based on user feedback

**These are intentional trade-offs for POC speed.**

---

## **19. SUCCESS METRICS**

After implementation, validate:

✅ **Technical Functionality**
- [ ] All 7-8 workflow nodes execute successfully
- [ ] State passes correctly between nodes
- [ ] RAG retrieves relevant chunks (manual verification)
- [ ] Task decomposition produces 2-5 atomic subtasks for complex requests
- [ ] Generated atomic code is syntactically valid >80% of time
- [ ] Composed pipeline code executes without errors >70% of time
- [ ] **Capability store saves and versions atomic capabilities**
- [ ] **Similar subtasks reuse existing atomic capabilities**
- [ ] **Git repository tracks all atomic capabilities**

✅ **Integration**
- [ ] Can run from command line
- [ ] Outputs are saved correctly
- [ ] Logs capture full execution trace including decomposition
- [ ] Error handling prevents crashes
- [ ] **Atomic capability reuse is logged and tracked per subtask**

✅ **Scientific Validity**
- [ ] Generated code uses appropriate methods from papers
- [ ] Results are biologically interpretable
- [ ] Figures are meaningful visualizations
- [ ] Composed pipelines maintain scientific correctness

✅ **Task Decomposition Performance** **(NEW)**
- [ ] Complex requests are decomposed into 2-5 atomic subtasks
- [ ] Subtask descriptions are clear and actionable
- [ ] Dependency graph is valid (no circular dependencies)
- [ ] Variable flow between subtasks is correct
- [ ] Composition produces executable pipeline code

✅ **Atomic Capability Store Performance** **(NEW)**
- [ ] Search finds similar atomic subtasks (similarity > 0.85)
- [ ] Reused atomic code executes successfully when composed
- [ ] Git commits are created for each new atomic capability
- [ ] Metadata tracks reuse statistics per atomic capability
- [ ] After 5-10 complex requests, reuse rate reaches 50-80%
- [ ] Composed pipelines are 5-10x faster than full generation
- [ ] Cost reduced by 60-80% due to atomic reuse

---

## **20. PRODUCTION MIGRATION PATH**

**After POC validation, prioritize:**

1. **Enhanced Security**: Replace exec() with Docker/E2B sandboxing
2. **Real Papers**: Add PDF loading and embed actual literature
3. **Code Refinement**: Add retry logic and iterative improvement
4. **Capability Intelligence**: **(Enhanced from basic POC)**
   - ML-based similarity matching (beyond cosine distance)
   - Automatic capability updates based on usage patterns
   - A/B testing different versions of same capability
   - Performance trend analysis
5. **Conditional Logic**: Add decision nodes in graph (if X fails, try Y)
6. **Multi-turn**: Allow conversation-style interaction
7. **Validation**: Add human review checkpoints
8. **Performance**: Parallel execution of independent nodes
9. **Advanced Storage**: Migrate to PostgreSQL for better query capabilities
10. **UI**: Build web interface for non-technical users

---

## **21. CRITICAL SETUP INSTRUCTIONS FOR CLAUDE CODE**

**Environment Setup:**
1. Create virtual environment: `python -m venv venv`
2. Activate: `source venv/bin/activate` (Unix) or `venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`
4. Create `.env` file with `OPENAI_API_KEY=your_key_here`

**First Run:**
1. Generate mock images (or create them manually)
2. RAG system will auto-build vector database on first run
3. Start with simplest test case ("count bright spots")
4. Check logs for debugging if issues occur

**Common Pitfalls:**
- Missing API key → Clear error message should appear
- ChromaDB issues → Delete `./data/vector_db` and rebuild
- Timeout on slow machines → Increase CODE_TIMEOUT_SECONDS
- Import errors → Verify all requirements installed

**Debug Mode:**
- Run with `--verbose` flag to see all DEBUG logs
- Check generated code in logs before execution
- Verify RAG chunks are relevant to request

---

## **22. CAPABILITY STORE: KEY FEATURES SUMMARY**

### **What It Does**
The Capability Store provides version control and intelligent reuse of AI-generated analysis code, making the system learn and improve over time.

### **Core Benefits**

**1. Cost Savings**
- First request: Full GPT-4 generation (~$0.01-0.03, 3 seconds)
- Similar requests: Instant reuse (~$0, 0.1 seconds)
- Estimated savings: 70-90% on repeated analysis types

**2. Speed Improvements**
- Eliminates GPT-4 latency for known tasks
- 10-30x faster for reused capabilities
- Scales better with increased usage

**3. Quality Assurance**
- Only saves capabilities that execute successfully
- Tracks reuse count (popular = reliable)
- Git history enables code review and rollback

**4. Continuous Learning**
- System builds library of proven analysis methods
- Semantic search finds relevant past solutions
- Accumulates domain expertise over time

### **Implementation Components**

**Storage Layer:**
- Git repository for version control
- ChromaDB for semantic search
- JSON metadata for statistics

**Integration Points:**
- `capability_search_node`: Before code generation
- `save_capability_node`: After successful execution
- Both integrate seamlessly into LangGraph workflow

### **Usage Pattern**

```
Day 1: "Segment cells"
→ Generate new code (3s)
→ Save to capability store

Day 2: "Count cells in image"  
→ Find similar (similarity: 0.91)
→ Reuse existing code (0.1s)

Day 7: "Segment the neurons"
→ Find similar (similarity: 0.88)
→ Reuse existing code (0.1s)

Week 2: Library has 15 capabilities
→ 80% of requests reuse existing code
→ System is 5x faster on average
```

### **Future Enhancements** (Post-POC)

**Smart Versioning:**
- A/B test multiple versions of same capability
- Automatically update based on user feedback
- Performance-based ranking

**Advanced Search:**
- ML-based similarity (beyond cosine)
- Multi-modal search (code + description + results)
- User-specific capability recommendations

**Analytics:**
- Dashboard showing capability usage
- Identify gaps in capability library
- Suggest new capabilities to build

---

## **READY FOR IMPLEMENTATION**

This plan provides complete specifications for Claude Code to build a working POC system. The implementation should take 5-6 days following the phased approach. All components are defined with clear inputs, outputs, and error handling requirements.

**Key Innovations**:
1. LangGraph state machine orchestrates RAG-enhanced code generation with task decomposition
2. **Task Decomposition (L1.5) breaks complex requests into atomic, reusable subtasks**
3. **Atomic Capability Store with git-based version control enables maximum code reuse**
4. **Capability Composition chains atomic capabilities into complete analysis pipelines**
5. Semantic search finds similar atomic subtasks across different complex requests
6. System learns and improves over time by building a library of atomic, composable analysis methods
7. High reuse rate (50-80% after initial usage) reduces cost and latency dramatically

The result is a self-extending, compositional analysis system that handles arbitrary calcium imaging requests by:
- Decomposing complex tasks into atomic operations
- Reusing proven atomic capabilities across different requests
- Composing atomic capabilities into novel analysis pipelines
- Continuously learning new atomic capabilities while maximizing reuse
- Achieving 5-10x speed improvement and 60-80% cost reduction through intelligent reuse
