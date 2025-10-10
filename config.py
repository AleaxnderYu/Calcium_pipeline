"""
Configuration module for the Calcium Imaging Agentic System.
Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ===== API Configuration =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY environment variable is required. "
        "Please create a .env file with your OpenAI API key. "
        "See .env.example for reference."
    )

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ===== Paths =====
PROJECT_ROOT = Path(__file__).parent
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
VECTOR_DB_PATH = PROJECT_ROOT / "data" / "vector_db"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CAPABILITY_STORE_PATH = PROJECT_ROOT / "data" / "capability_store"

# Ensure directories exist
PAPERS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CAPABILITY_STORE_PATH.mkdir(parents=True, exist_ok=True)

# ===== Execution Settings =====
CODE_TIMEOUT_SECONDS = 30
MAX_MEMORY_MB = 512  # Future use
ALLOWED_IMPORTS = ["numpy", "scipy", "matplotlib", "skimage"]

# ===== RAG Settings =====
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_CHUNKS = 3

# ===== Capability Store Settings =====
CAPABILITY_SIMILARITY_THRESHOLD = float(os.getenv("CAPABILITY_SIMILARITY_THRESHOLD", "0.85"))
MAX_CAPABILITY_AGE_DAYS = int(os.getenv("MAX_CAPABILITY_AGE_DAYS", "90"))
ENABLE_CAPABILITY_REUSE = os.getenv("ENABLE_CAPABILITY_REUSE", "true").lower() == "true"

# ===== Logging =====
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = PROJECT_ROOT / "pipeline.log"
