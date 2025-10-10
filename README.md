# Calcium Imaging Agentic System

An AI-powered system that processes calcium imaging data through RAG-enhanced code generation using LangGraph orchestration.

## Features

- **Natural Language Interface**: Request analyses in plain English
- **RAG-Enhanced Code Generation**: Retrieves relevant methods from scientific literature
- **Automated Workflow**: LangGraph orchestrates preprocessing, retrieval, code generation, and execution
- **Safe Execution**: Sandboxed Python execution with timeout protection

## Architecture

```
User Request → L3 Preprocessor → L2 RAG System → L4 Code Generator → Executor → Results
               (Load Images)     (Find Methods)   (Generate Code)    (Run Code)
```

## Setup

### Option A: Using Conda (Recommended)

#### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate calcium_pipeline
```

#### 2. Verify Installation

```bash
python test_components.py
```

### Option B: Using pip/venv

#### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

Create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=your_key_here
```

### 4. Generate Mock Data (Optional)

Mock calcium imaging data is already generated. To regenerate:

```bash
python utils/generate_mock_images.py
```

## Usage

### Basic Usage

```bash
python main.py --request "Count the number of cells" --images ./data/images
```

### Advanced Options

```bash
# Verbose logging
python main.py --request "Calculate mean intensity over time" --images ./data/images --verbose

# Rebuild RAG database
python main.py --request "Detect calcium transients" --images ./data/images --rebuild-rag

# Custom output directory
python main.py --request "Segment cells" --images ./data/images --output ./my_results
```

## Example Requests

- "Count the number of cells in the images"
- "Calculate mean intensity over time for each cell"
- "Detect calcium transients and measure their amplitude"
- "Segment cells and extract their properties"
- "Find bright spots and count them"

## Output

Results are saved to `./outputs/YYYY-MM-DD_HH-MM-SS/`:

- `report.json`: Complete analysis results
- `generated_code.py`: Code that was executed
- `figure_001.png`: Generated visualizations (if any)
- `logs.txt`: Detailed execution log

## Configuration

Edit `config.py` to customize:

- `OPENAI_MODEL`: Model for code generation (default: "gpt-4")
- `CODE_TIMEOUT_SECONDS`: Execution timeout (default: 30)
- `CHUNK_SIZE`: RAG chunk size (default: 1000)
- `TOP_K_CHUNKS`: Number of chunks to retrieve (default: 3)

## Project Structure

```
calcium_pipeline/
├── main.py                 # Entry point
├── config.py              # Configuration
├── graph/                 # LangGraph workflow
│   ├── workflow.py       # Workflow definition
│   ├── state.py          # State schema
│   └── nodes.py          # Node functions
├── layers/               # Core components
│   ├── l2_rag_system.py  # RAG retrieval
│   ├── l3_preprocessor.py # Image loading
│   └── l4_capability_manager.py # Code generation
├── core/                 # Utilities
│   ├── data_models.py    # Data structures
│   └── executor.py       # Code execution
├── data/                 # Data files
│   ├── papers/          # Scientific papers (txt)
│   ├── images/          # Sample images (png)
│   └── vector_db/       # ChromaDB storage
└── outputs/             # Generated results
```

## Troubleshooting

### Missing API Key Error

```
ValueError: OPENAI_API_KEY environment variable is required
```

**Solution**: Create `.env` file with your OpenAI API key.

### ChromaDB Issues

If RAG retrieval fails, rebuild the database:

```bash
rm -rf ./data/vector_db
python main.py --request "test" --images ./data/images --rebuild-rag
```

### Timeout Errors

If code execution times out, increase the timeout in `config.py`:

```python
CODE_TIMEOUT_SECONDS = 60
```

### Import Errors

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt --upgrade
```

## Limitations (POC Scope)

- **Security**: Uses `exec()` with namespace restrictions (POC-level only)
- **Code Quality**: Generated code may not always be optimal
- **Error Recovery**: No iterative refinement of failed code
- **Workflow**: Linear only (no conditional branching yet)

## Future Enhancements

1. Docker/E2B sandboxing for production-grade security
2. Real scientific papers (PDF loading)
3. Code refinement with retry logic
4. Multi-turn conversation support
5. Human-in-the-loop validation
6. Web interface
7. Capability caching

## License

MIT

## Support

For issues and questions, see the project documentation or contact the development team.
