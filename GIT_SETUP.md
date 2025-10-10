# Git Version Control Setup

## Repository Information

- **Branch**: `main`
- **Initial Commit**: c5295be
- **Status**: ✓ Clean working tree

## What's Tracked

### Source Code
- All Python modules (`core/`, `layers/`, `graph/`, `utils/`)
- Configuration files (`config.py`, `requirements.txt`, `environment.yml`)
- Test files (`test_components.py`, `tests/`)

### Documentation
- `README.md` - Setup instructions
- `calcium_imaging_poc_plan.md` - Complete system specification
- `CAPABILITY_STORE.md` - Capability store documentation
- `REFACTORING_SUMMARY.md` - Layer numbering removal summary

### Data
- `data/papers/` - Mock scientific papers (3 text files)
- `data/images/` - Mock calcium imaging frames (10 PNG files)

## What's Ignored

### Generated/Runtime Files
- `__pycache__/` - Python bytecode
- `*.pyc`, `*.pyo` - Compiled Python files
- `outputs/*/` - Analysis results and figures
- `*.log` - Log files

### Environment
- `venv/`, `env/` - Virtual environments
- `.env` - API keys and secrets (use `.env.example` as template)

### Data/State
- `data/vector_db/` - ChromaDB vector database (auto-built from papers)
- `data/capability_store/` - Has its own git repository for versioning capabilities

### IDE/OS
- `.vscode/`, `.idea/` - IDE settings
- `.DS_Store`, `Thumbs.db` - OS-specific files

## Nested Repository

The `data/capability_store/` directory has its own git repository for tracking generated capabilities:
- Manages version control for AI-generated code
- Tracks capability reuse statistics
- Maintains separate commit history from main project

## Common Git Commands

### Check Status
```bash
git status
git log --oneline
```

### Stage and Commit Changes
```bash
git add <file>
git commit -m "Descriptive message"
```

### View History
```bash
git log
git log --graph --oneline --all
git diff
```

### Create Branch
```bash
git checkout -b feature/new-feature
```

### Undo Changes
```bash
git checkout -- <file>          # Discard changes in file
git reset HEAD <file>            # Unstage file
git reset --soft HEAD~1          # Undo last commit, keep changes
```

## Project Structure (Tracked)

```
calcium_pipeline/
├── .gitignore                    # Git ignore rules
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
├── config.py                     # Configuration
├── main.py                       # Entry point
├── core/                         # Core functionality
│   ├── data_models.py
│   └── executor.py
├── layers/                       # Main components
│   ├── preprocessor.py
│   ├── rag_system.py
│   ├── capability_manager.py
│   └── capability_store.py
├── graph/                        # LangGraph workflow
│   ├── state.py
│   ├── nodes.py
│   └── workflow.py
├── utils/                        # Utilities
├── data/
│   ├── papers/                   # Tracked
│   └── images/                   # Tracked
└── tests/                        # Test files
```

## Notes

- The main repository tracks the **code and static data**
- The capability_store repository tracks the **generated capabilities**
- Both work together but maintain separate version histories
- Use `.env.example` as a template for your `.env` file (never commit `.env`)
