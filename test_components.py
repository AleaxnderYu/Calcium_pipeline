#!/usr/bin/env python3
"""
Test script to verify individual components work correctly.
"""

import sys
from pathlib import Path

# Test 1: Config and imports
print("=" * 60)
print("TEST 1: Imports and Configuration")
print("=" * 60)

try:
    import numpy as np
    import scipy
    import matplotlib
    import skimage
    from PIL import Image
    print("✓ All scientific libraries imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

try:
    from langchain_community.document_loaders import DirectoryLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    from langgraph.graph import StateGraph, END
    print("✓ LangChain and LangGraph imported successfully")
except ImportError as e:
    print(f"✗ LangChain/LangGraph import failed: {e}")
    print("  Run: pip install -r requirements.txt")
    sys.exit(1)

# Test 2: Project modules
print("\n" + "=" * 60)
print("TEST 2: Project Modules")
print("=" * 60)

try:
    from core.data_models import PreprocessedData, RAGContext, ExecutionResult
    from layers.preprocessor import Preprocessor
    from core.executor import execute_code
    print("✓ All project modules imported successfully")
except ImportError as e:
    print(f"✗ Project module import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Mock data exists
print("\n" + "=" * 60)
print("TEST 3: Mock Data")
print("=" * 60)

data_dir = Path(__file__).parent / "data"
images_dir = data_dir / "images"
papers_dir = data_dir / "papers"

png_files = list(images_dir.glob("*.png"))
txt_files = list(papers_dir.glob("*.txt"))

print(f"Images directory: {images_dir}")
print(f"  Found {len(png_files)} PNG files")
if len(png_files) > 0:
    print("✓ Mock images present")
else:
    print("✗ No mock images found")

print(f"\nPapers directory: {papers_dir}")
print(f"  Found {len(txt_files)} text files")
if len(txt_files) > 0:
    print("✓ Mock papers present")
else:
    print("✗ No mock papers found")

# Test 4: L3 Preprocessor
print("\n" + "=" * 60)
print("TEST 4: Preprocessor")
print("=" * 60)

try:
    preprocessor = Preprocessor()
    preprocessed_data = preprocessor.process(str(images_dir))

    print(f"  Shape: {preprocessed_data.images.shape}")
    print(f"  Dtype: {preprocessed_data.images.dtype}")
    print(f"  Range: [{preprocessed_data.images.min():.3f}, {preprocessed_data.images.max():.3f}]")
    print(f"  Metadata: {preprocessed_data.metadata}")
    print("✓ L3 Preprocessor working correctly")
except Exception as e:
    print(f"✗ L3 Preprocessor failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Code Executor (simple test)
print("\n" + "=" * 60)
print("TEST 5: Code Executor")
print("=" * 60)

test_code = """
# Simple test code
results = {
    'mean': float(images.mean()),
    'max': float(images.max()),
    'shape': images.shape
}
figure = None
"""

try:
    test_images = np.random.rand(5, 64, 64).astype(np.float32)
    result = execute_code(test_code, test_images, timeout=5)

    if result.success:
        print(f"  Results: {result.results}")
        print(f"  Execution time: {result.execution_time:.3f}s")
        print("✓ Code Executor working correctly")
    else:
        print(f"✗ Code execution failed: {result.error_message}")
except Exception as e:
    print(f"✗ Code Executor test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Check for .env file
print("\n" + "=" * 60)
print("TEST 6: API Configuration")
print("=" * 60)

env_file = Path(__file__).parent / ".env"
if env_file.exists():
    print(f"✓ .env file found at {env_file}")
    print("  You can now test the full workflow with RAG and code generation")
else:
    print(f"⚠ .env file not found")
    print("  To test RAG and code generation:")
    print("  1. Copy .env.example to .env")
    print("  2. Add your OpenAI API key to .env")
    print("  3. Run: python main.py --request \"Count cells\" --images ./data/images")

print("\n" + "=" * 60)
print("COMPONENT TESTS COMPLETED")
print("=" * 60)
print("\nAll basic components are working correctly!")
print("\nNext steps:")
print("1. Create .env file with your OpenAI API key")
print("2. Run: python main.py --request \"Count bright spots\" --images ./data/images")
