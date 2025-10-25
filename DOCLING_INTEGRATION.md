# Docling Integration for Advanced PDF Parsing

## What is Docling?

**Docling** is IBM Research's advanced document parsing library that:
- ✅ Extracts **tables with structure** (unlike PyPDF)
- ✅ Identifies **images and figures**
- ✅ Understands **page layout and reading order**
- ✅ Recognizes **formulas and code blocks**
- ✅ Provides **OCR for scanned PDFs**
- ✅ Powered by AI models (DocLayNet, TableFormer)

**Perfect for scientific papers** with complex layouts, tables, and figures!

## Why We Switched from PyPDF

| Feature | PyPDF | Docling |
|---------|-------|---------|
| Basic text | ✅ | ✅ |
| Tables (structured) | ❌ | ✅ |
| Images/Figures | ❌ | ✅ |
| Page layout | ❌ | ✅ |
| Reading order | ❌ | ✅ |
| Formulas | ❌ | ✅ |
| OCR (scanned PDFs) | ❌ | ✅ |
| Scientific papers | ⚠️ Poor | ✅ Excellent |

## Installation

### Basic Installation (No OCR)
```bash
# Install Docling (basic features)
pip install docling

# Or update all requirements
pip install -r requirements.txt
```

### With OCR Support (Recommended for Scanned PDFs)
```bash
# Option 1: Install with EasyOCR (default, easier)
pip install "docling[easyocr]"

# Option 2: Install with Tesseract (more accurate)
# First install Tesseract system package:
# - Ubuntu/WSL: sudo apt-get install tesseract-ocr
# - macOS: brew install tesseract
# - Windows: Download from GitHub
# Then:
pip install "docling[tesseract]"

# Option 3: Install both OCR backends
pip install "docling[easyocr,tesseract]"
```

## How It Works

### Automatic Fallback

The system automatically uses Docling if available, otherwise falls back to PyPDF:

```python
# In layers/rag_system.py
if DOCLING_AVAILABLE:
    logger.info("Using Docling (advanced)")
    loader = DoclingPDFLoader
else:
    logger.info("Using PyPDF (basic)")
    loader = PyPDFLoader
```

### What Gets Extracted

**With PyPDF (old):**
```
Just raw text, often mangled from tables
```

**With Docling (new):**
```markdown
# Paper Title
## Abstract
...

### Table 1: Calcium Transient Detection Methods
| Method | Sensitivity | Specificity |
|--------|-------------|-------------|
| OASIS  | 95%         | 92%         |
| Suite2p| 93%         | 91%         |

### Figure 1
[Image: ROI segmentation example]
Caption: Watershed segmentation results...

### Formula
ΔF/F₀ = (F - F₀) / F₀
```

## Better RAG Results

### Before (PyPDF)
**Query:** "What methods detect calcium transients?"

**Retrieved chunks:** Garbled text from tables, formulas broken

**Answer:** Limited, missing key details

### After (Docling)
**Query:** "What methods detect calcium transients?"

**Retrieved chunks:** 
- Structured table with method comparisons
- Formulas preserved correctly
- Figure captions with context

**Answer:** Comprehensive with specific metrics!

## Usage

### Rebuild Vector Database

After installing Docling, rebuild to reprocess PDFs:

```bash
# Delete old database
rm -rf data/vector_db/

# Rebuild with Docling
streamlit run app.py
# System will auto-rebuild on first query
```

### Verify Docling is Active

Check logs when RAG loads:

```bash
# Look for this line:
INFO: Using Docling for PDF parsing (advanced tables/images/layout)

# vs old:
INFO: Using PyPDF for PDF parsing (basic text only)
```

## Features You Get

### 1. Table Structure Preservation

**Example from calcium imaging paper:**

Original table in PDF:
```
| ROI | Baseline (AU) | Peak (AU) | ΔF/F₀ |
|-----|--------------|-----------|-------|
| 1   | 100          | 150       | 0.50  |
| 2   | 95           | 180       | 0.89  |
```

**PyPDF result:**
```
ROI Baseline (AU) Peak (AU) ΔF/F₀
1 100 150 0.50
2 95 180 0.89
```
(Lost structure!)

**Docling result:**
```markdown
| ROI | Baseline (AU) | Peak (AU) | ΔF/F₀ |
|-----|--------------|-----------|-------|
| 1   | 100          | 150       | 0.50  |
| 2   | 95           | 180       | 0.89  |
```
(Perfect structure!)

### 2. Formula Recognition

**Calcium transient formula:**
```
ΔF/F₀ = (F - F₀) / F₀
```

- PyPDF: `F/F = (F - F ) / F` (broken)
- Docling: `ΔF/F₀ = (F - F₀) / F₀` (perfect!)

### 3. Figure Captions

Docling extracts figure captions with context:
```
Figure 3: ROI segmentation using watershed algorithm.
(A) Original calcium imaging frame
(B) Segmented ROIs with boundaries
(C) Temporal traces for each ROI
```

PyPDF often loses this or mangles it.

### 4. Reading Order

Docling understands multi-column layouts:
```
Column 1 → Column 2 → Column 3
(correct reading order)
```

PyPDF reads left-to-right across columns (wrong order).

## Configuration

### Docling Options (Advanced)

You can customize Docling behavior in `rag_system.py`:

```python
# Default (automatic)
loader_cls=DoclingPDFLoader

# With options (if needed):
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

# Enable OCR explicitly
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True  # Enable OCR for scanned PDFs
pipeline_options.ocr_engine = "easyocr"  # or "tesseract"

converter = DocumentConverter(
    format_options={
        PdfFormatOption: pipeline_options
    }
)
```

### OCR Configuration

**When is OCR used?**
- Docling **automatically detects** if a PDF has no text layer (scanned image)
- If detected, OCR is triggered automatically (if OCR backend is installed)
- You can force OCR with `do_ocr=True`

**OCR Backends:**

1. **EasyOCR** (Default)
   - Supports 80+ languages
   - Pure Python, easier to install
   - Good accuracy for most cases
   - Install: `pip install "docling[easyocr]"`

2. **Tesseract**
   - Google's OCR engine
   - Better for complex layouts
   - Requires system package installation
   - Install system package first, then: `pip install "docling[tesseract]"`

**Language Support:**
```python
# For non-English papers (e.g., Chinese calcium imaging papers)
pipeline_options.ocr_lang = ["en", "zh"]  # English + Chinese
```

## Performance

### Processing Speed

- **PyPDF**: ~1-2 seconds per paper
- **Docling**: ~3-5 seconds per paper (slower but worth it!)

### First-time Build

Your 91 papers will take:
- **PyPDF**: ~2-3 minutes
- **Docling**: ~5-8 minutes

*Once built, retrieval speed is the same!*

## Supported Formats

Docling supports more than just PDF:

- ✅ PDF (with tables, images, formulas)
- ✅ DOCX (Word documents)
- ✅ PPTX (PowerPoint)
- ✅ XLSX (Excel spreadsheets)
- ✅ HTML
- ✅ Images (PNG, JPEG with OCR)

You can extend RAG to these formats later!

## Troubleshooting

### "Docling not available" warning

**Solution:**
```bash
pip install docling
```

### Rebuild doesn't use Docling

**Solution:**
```bash
# Force rebuild
rm -rf data/vector_db/
python -c "from layers.rag_system import DOCLING_AVAILABLE; print('Docling:', DOCLING_AVAILABLE)"
# Should print: Docling: True
```

### Slow initial load

**Expected!** Docling is slower but more accurate. First-time vectorization takes ~5-8 minutes for 91 papers.

### Out of memory

If processing very large PDFs:
```bash
# Process in smaller batches
# Or increase system RAM
```

## Migration Checklist

- [x] Install Docling: `pip install docling`
- [x] Update requirements.txt
- [x] Modify RAG system with fallback
- [ ] Delete old vector DB: `rm -rf data/vector_db/`
- [ ] Rebuild database (automatic on next query)
- [ ] Verify logs show "Using Docling"
- [ ] Test queries with table/formula content

## Implementing OCR in Your System

### Quick Setup (Recommended)

For **automatic OCR on scanned PDFs**, just install Docling with OCR support:

```bash
# Install with EasyOCR (easiest, works immediately)
pip install "docling[easyocr]"

# Then rebuild vector database
rm -rf data/vector_db/
streamlit run app.py
```

**That's it!** Docling will automatically:
- Detect if a PDF is scanned (no text layer)
- Run OCR to extract text from images
- Process tables, formulas, and figures

### Advanced: Force OCR for All PDFs

If you want to **always use OCR** (even for PDFs with text), modify `layers/rag_system.py`:

```python
# Around line 10, add:
from langchain_community.document_loaders import DoclingPDFLoader

# Replace the DirectoryLoader section (lines 99-114) with:
if DOCLING_AVAILABLE:
    logger.info("Using Docling with OCR enabled")

    # Custom loader with OCR configuration
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    # Configure OCR
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.ocr_engine = "easyocr"  # or "tesseract"

    # Use custom converter in loader
    pdf_loader = DirectoryLoader(
        str(self.papers_dir),
        glob="*.pdf",
        loader_cls=DoclingPDFLoader,
        loader_kwargs={
            "pipeline_options": pipeline_options
        }
    )
```

### Testing OCR

To test if OCR is working:

1. **Add a scanned PDF** to `data/papers/`
2. **Rebuild database**: `rm -rf data/vector_db/`
3. **Check logs** for OCR activity:
   ```
   INFO: Processing scanned PDF with OCR...
   INFO: OCR detected text: [preview]
   ```

### OCR Use Cases for Your Calcium Papers

**When you need OCR:**
- ✅ Old papers scanned from print journals
- ✅ Papers with handwritten annotations
- ✅ Screenshots of figures/tables
- ✅ Low-quality PDF scans

**When you DON'T need OCR:**
- ✅ Modern PDFs from publishers (already have text layer)
- ✅ ArXiv papers (born digital)
- ✅ Most recent papers from PubMed

**Most scientific papers from PubMed/ArXiv don't need OCR**, but having it enabled ensures older scanned papers are handled correctly.

## Next Steps

1. **Install Docling with OCR:**
   ```bash
   pip install "docling[easyocr]"
   ```

2. **Rebuild vector database:**
   ```bash
   rm -rf data/vector_db/
   streamlit run app.py
   ```

3. **Test with queries that need tables:**
   ```
   "What are the performance metrics of different calcium transient detection methods?"
   ```

4. **Check logs** to confirm Docling is active and OCR is available

## Benefits Summary

✅ **Better table extraction** - Structured data preserved  
✅ **Formula recognition** - Math equations readable  
✅ **Figure awareness** - Captions and context  
✅ **Reading order** - Multi-column layouts correct  
✅ **OCR support** - Scanned papers readable  
✅ **AI-powered** - DocLayNet & TableFormer models  

Your RAG just got a major upgrade! 🚀
