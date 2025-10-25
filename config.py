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

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "gpt-5")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# API Server URL for serving PDFs (used for clickable citations)
# Default: http://localhost:8000 (FastAPI default)
# Set this to your public URL if accessing from remote clients
API_SERVER_URL = os.getenv("API_SERVER_URL", "http://localhost:8000")

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
CODE_TIMEOUT_SECONDS = 300  # 5 minutes timeout for code execution
MAX_MEMORY_MB = 512  # Future use
ALLOWED_IMPORTS = ["numpy", "scipy", "matplotlib", "skimage"]

# Docker Execution Settings
DOCKER_BASE_IMAGE = os.getenv("DOCKER_BASE_IMAGE", "calcium_imaging:latest")
DOCKER_MEMORY_LIMIT = os.getenv("DOCKER_MEMORY_LIMIT", "2g")
DOCKER_CPU_QUOTA = int(os.getenv("DOCKER_CPU_QUOTA", "100000"))  # 100000 = 1 CPU
DOCKER_NETWORK_DISABLED = os.getenv("DOCKER_NETWORK_DISABLED", "true").lower() == "true"

# ===== Legacy Sandbox Settings (DEPRECATED - Not used with Docker) =====
# Docker containers are created and destroyed per execution
# No manual lifecycle management needed
CLOSE_SANDBOX_AFTER_EXECUTION = os.getenv("CLOSE_SANDBOX_AFTER_EXECUTION", "true").lower() == "true"
SANDBOX_REUSE_ENABLED = os.getenv("SANDBOX_REUSE_ENABLED", "false").lower() == "true"
SANDBOX_MAX_IDLE_MINUTES = int(os.getenv("SANDBOX_MAX_IDLE_MINUTES", "5"))

# ===== RAG Settings =====
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_CHUNKS = 3

# Docling OCR Settings
# Options: "easyocr" (default), "tesseract", "tesseract_cli"
DOCLING_OCR_ENGINE = os.getenv("DOCLING_OCR_ENGINE", "easyocr")
DOCLING_ENABLE_OCR = os.getenv("DOCLING_ENABLE_OCR", "true").lower() == "true"

# ===== Capability Store Settings =====
CAPABILITY_SIMILARITY_THRESHOLD = float(os.getenv("CAPABILITY_SIMILARITY_THRESHOLD", "0.85"))
MAX_CAPABILITY_AGE_DAYS = int(os.getenv("MAX_CAPABILITY_AGE_DAYS", "90"))
ENABLE_CAPABILITY_REUSE = os.getenv("ENABLE_CAPABILITY_REUSE", "true").lower() == "true"

# ===== Logging =====
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = PROJECT_ROOT / "pipeline.log"
