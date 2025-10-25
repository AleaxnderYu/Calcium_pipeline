# Calcium Imaging Agentic System

An AI-powered system for calcium imaging analysis with tool-based orchestration, user approval, and error feedback via Open WebUI.

## Features

- **Tool-Based Workflow**: Explicit tool calls (RAG, code generation, execution, verification)
- **User Approval**: Review execution plans before running
- **Error Feedback**: Interactive error handling with retry/skip/abort options
- **RAG-Enhanced**: Retrieves methods from 91 scientific papers
- **Docker Execution**: Secure, isolated code execution
- **Capability Reuse**: Saves successful patterns for future use
- **Open WebUI Integration**: Chat-based interface

## Architecture

```
User Query (Open WebUI)
    ↓
Planner (Create tool-based plan)
    ↓
User Approval (Review & approve)
    ↓
Orchestrator (Execute tools)
    ├─ RAG Tool
    ├─ Code Generation Tool
    ├─ Execute Tool
    ├─ Verify Tool
    └─ Capability Tools
    ↓
Error? → Present to User → Retry/Skip/Abort
    ↓
Results (Return to Open WebUI)
```

**Core Components:**
- **Tool Planner**: Decomposes requests into executable tool calls
- **Orchestrator**: Executes tools with dependency management
- **RAG System**: Retrieves from 91 calcium imaging papers
- **Docker Executor**: Runs generated code in isolated containers
- **Verifier**: Validates execution results for correctness
- **Capability Store**: Git + ChromaDB for code reuse

## Quick Start

### 1. Install Dependencies

```bash
# Using conda (recommended)
conda create -n calcium_pipeline python=3.11
conda activate calcium_pipeline
pip install -r requirements.txt
```

### 2. Install Docker

Docker is required for code execution.

**Linux (WSL):**
```bash
sudo apt-get update
sudo apt-get install docker.io
sudo service docker start
sudo usermod -aG docker $USER
```

**macOS/Windows**: Download Docker Desktop from https://www.docker.com/products/docker-desktop

Verify Docker is running:
```bash
docker --version
docker ps
```

### 3. Build Docker Image

```bash
cd docker
./build_image.sh

# Or manually:
docker build -f Dockerfile.calcium_imaging -t calcium_imaging:latest .
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
nano .env
```

Required variables:
```bash
OPENAI_API_KEY=sk-your-key-here
IMAGES_DIR=data/images
PAPERS_DIR=data/papers
VECTOR_DB_PATH=data/vector_db
```

### 5. Add Scientific Papers

```bash
# Add PDF papers to data/papers/
cp /path/to/papers/*.pdf data/papers/

# Vector database will auto-build on first query
```

See [docs/guides/ADDING_PAPERS_GUIDE.md](docs/guides/ADDING_PAPERS_GUIDE.md) for details.

### 6. Install Docling (Optional, for better PDF parsing)

```bash
# Basic installation
pip install docling

# With OCR support for scanned PDFs
pip install "docling[easyocr]"
```

See [DOCLING_INTEGRATION.md](DOCLING_INTEGRATION.md) for details.

### 7. Start the API Backend

```bash
# Start FastAPI server for Open WebUI
uvicorn api_backend:app --host 0.0.0.0 --port 8000
```

### 8. Configure Open WebUI

In Open WebUI settings:
1. Go to Admin Panel → Connections
2. Add OpenAI API:
   - Base URL: `http://localhost:8000`
   - API Key: (any value, not used)
3. Select the model in chat

See [docs/guides/OPEN_WEBUI_SETUP.md](docs/guides/OPEN_WEBUI_SETUP.md) for detailed setup.

## Usage

### Via Open WebUI

1. Start Open WebUI and select the calcium imaging model
2. Ask questions:
   ```
   "Count cells in the calcium imaging data"
   "What are calcium transients?"
   "Detect calcium events in the time series"
   ```

### Direct Python API

```python
from graph.workflow import run_workflow

result = run_workflow(
    user_request="Count cells in the image",
    images_path="data/images"
)

print(result.data)
# {'cell_count': 42}

print(result.summary)
# "Executed 5 tools for: Count cells..."
```

### Example Workflow

**User Query**: "Count cells in the image"

**Generated Plan**:
```
1. [rag] Retrieve cell segmentation methods from papers
2. [code_generation] Generate watershed segmentation code
3. [execute] Run segmentation in Docker
4. [verify] Verify cell count is reasonable
5. [capability_save] Save pattern for reuse
```

**Execution**:
```
✓ RAG retrieved 5 relevant paper chunks
✓ Generated code using watershed algorithm
✓ Executed in Docker (2.3s)
✓ Verified: Found 42 cells (reasonable)
✓ Saved capability: cell_counting_watershed_20250122
```

**Result**:
```json
{
    "cell_count": 42,
    "segmentation_map": "path/to/segmentation.png",
    "execution_time": 2.3
}
```

## Tool Types

### 1. RAG Tool
Retrieves scientific methods from papers.
```python
Input:  {"query": "calcium transient detection"}
Output: {"chunks": [...], "sources": [...]}
```

### 2. Code Generation Tool
Generates Python code using GPT-4.
```python
Input:  {"task_description": "...", "rag_context": {...}}
Output: {"code": "import numpy as np...", "description": "..."}
```

### 3. Execute Tool
Runs code in Docker sandbox.
```python
Input:  {"code": "...", "images_path": "..."}
Output: {"success": True, "results": {...}, "figures": [...]}
```

### 4. Verify Tool
Validates execution results.
```python
Input:  {"execution_result": {...}, "expected_output": "..."}
Output: {"passed": True, "confidence": 0.95, "issues": []}
```

### 5. Capability Search/Save
Finds or saves reusable code.
```python
Search: {"query": "cell counting"}
Save:   {"code": "...", "description": "..."}
```

## Project Structure

```
calcium_pipeline/
├── core/                       # Core logic
│   ├── tool_planner.py        # Tool-based planner
│   ├── orchestrator.py        # Tool executor
│   ├── verifier.py            # Result validator
│   ├── docker_executor.py     # Docker sandbox
│   └── data_models.py         # Data structures
├── graph/                      # Workflow
│   ├── workflow.py            # Main workflow
│   └── state.py               # PipelineState
├── layers/                     # Supporting layers
│   ├── rag_system.py          # RAG (91 papers)
│   ├── capability_manager.py  # Capability storage
│   └── preprocessor.py        # Image preprocessing
├── docker/                     # Docker configuration
│   ├── Dockerfile.calcium_imaging
│   └── build_image.sh
├── data/
│   ├── papers/                # Scientific PDFs
│   ├── images/                # Calcium imaging data
│   ├── vector_db/             # ChromaDB (auto-generated)
│   └── capability_store/      # Git repo (auto-generated)
├── api_backend.py             # FastAPI for Open WebUI
├── main.py                    # Entry point
├── config.py                  # Configuration
└── requirements.txt           # Python dependencies
```

## Configuration

Edit [config.py](config.py) to customize:

```python
# Models
OPENAI_MODEL = "gpt-4"              # For code generation
ROUTER_MODEL = "gpt-3.5-turbo"      # For planning/verification
EMBEDDING_MODEL = "text-embedding-3-small"

# RAG Settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K_CHUNKS = 5

# Execution
CODE_TIMEOUT_SECONDS = 300
MAX_RETRIES = 3
```

## Data

### Images
Place calcium imaging data in `data/images/`:
- Supported: PNG, TIFF, NPY
- Can be single images or time series
- System auto-detects format

### Papers
Add scientific PDFs to `data/papers/`:
- 91 papers currently included
- Vector database auto-builds on first query
- Supports text PDFs and scanned PDFs (with Docling OCR)

To rebuild vector database:
```bash
rm -rf data/vector_db/
# Will rebuild on next query
```

## Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Tool Planner | GPT-3.5-turbo | Fast, cheap planning |
| Code Generation | GPT-4 | High-quality code |
| Verifier | GPT-3.5-turbo | Quick validation |
| RAG Embeddings | text-embedding-3-small | Vector search |

## Documentation

### Guides
- **[docs/guides/OPEN_WEBUI_SETUP.md](docs/guides/OPEN_WEBUI_SETUP.md)** - Open WebUI integration
- **[docs/guides/ADDING_PAPERS_GUIDE.md](docs/guides/ADDING_PAPERS_GUIDE.md)** - How to add papers
- **[docs/guides/GIT_SETUP.md](docs/guides/GIT_SETUP.md)** - Git configuration
- **[docs/guides/MODEL_SETTINGS_GUIDE.md](docs/guides/MODEL_SETTINGS_GUIDE.md)** - Model configuration

### System Documentation
- **[SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md)** - System design and components
- **[WORKFLOW_ARCHITECTURE.md](WORKFLOW_ARCHITECTURE.md)** - Workflow execution details
- **[CITATION_SYSTEM.md](CITATION_SYSTEM.md)** - Citation and RAG system
- **[HTTP_CITATIONS.md](HTTP_CITATIONS.md)** - Clickable PDF citations
- **[DOCLING_INTEGRATION.md](DOCLING_INTEGRATION.md)** - PDF parsing with Docling
- **[ENHANCED_RAG.md](ENHANCED_RAG.md)** - Enhanced RAG with section-based retrieval

### Archive
Historical documentation moved to `docs/archive/`

## Troubleshooting

### Docker Issues

**Permission denied:**
```bash
sudo usermod -aG docker $USER
newgrp docker
```

**Docker not running:**
```bash
sudo service docker start  # Linux
# Or start Docker Desktop  # macOS/Windows
```

### Import Errors

**Missing modules:**
```bash
pip install -r requirements.txt
```

**Docling not available:**
```bash
pip install docling
# Or with OCR: pip install "docling[easyocr]"
```

### RAG Issues

**No papers found:**
```bash
ls data/papers/  # Should see PDFs
```

**Rebuild vector database:**
```bash
rm -rf data/vector_db/
# Rebuilds on next query
```

### Code Execution Errors

**Check Docker is running:**
```bash
docker ps
```

**Check Docker image exists:**
```bash
docker images | grep calcium_imaging
```

**Rebuild Docker image:**
```bash
cd docker && ./build_image.sh
```

## Development

### Run Tests

```bash
python -m pytest tests/
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python main.py
```

### Add New Tools

1. Define tool type in `core/data_models.py`:
   ```python
   class ToolType(Enum):
       MY_TOOL = "my_tool"
   ```

2. Add tool execution in `core/orchestrator.py`:
   ```python
   def _execute_my_tool(self, inputs):
       # Tool implementation
       return {"output": ...}
   ```

3. Update planner prompt in `core/tool_planner.py` to include new tool

## Contributing

This is a research project for calcium imaging analysis. For questions or contributions, please open an issue.

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Uses Docling (IBM Research) for advanced PDF parsing
- Built with LangGraph for workflow orchestration
- Powered by OpenAI GPT-4 for code generation
- Scientific papers from calcium imaging research community
