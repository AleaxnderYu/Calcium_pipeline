# 📚 Guide: Adding Real Papers to the RAG System

## Quick Start

### 1️⃣ Add Your PDFs

```bash
cd /home/xinra/NotreDame/Calcium_pipeline/data/papers

# Copy your PDF papers here
cp ~/path/to/your/papers/*.pdf ./

# Or use wget/curl if papers are online:
# wget https://example.com/paper.pdf
```

### 2️⃣ Rebuild Vector Database

```bash
cd /home/xinra/NotreDame/Calcium_pipeline

# Delete old database
rm -rf ./data/vector_db

# System will rebuild automatically on next use
```

### 3️⃣ Test It

```bash
# Run test script
python test_rag_papers.py

# Or use Streamlit
streamlit run app.py
# Then ask: "What methods exist for detecting calcium transients?"
```

---

## 📖 Detailed Instructions

### Understanding the Current Setup

**Current papers** (mock text files):
```
data/papers/
├── baseline_calculation.txt (1.8 KB)
├── segmentation_methods.txt (1.6 KB)
└── transient_detection.txt (1.6 KB)
```

**The RAG system supports**:
- ✅ `.txt` files (plain text)
- ✅ `.pdf` files (automatically parsed)
- ✅ Mixed formats (can have both!)

---

## 🎯 Recommended Approach

### Option A: Add PDFs Alongside Mock Files

**Best for**: Testing without removing mock data

```bash
cd /home/xinra/NotreDame/Calcium_pipeline/data/papers

# Add your PDFs
cp ~/Downloads/calcium_papers/*.pdf ./

# Result:
# data/papers/
# ├── baseline_calculation.txt       (mock)
# ├── segmentation_methods.txt       (mock)
# ├── transient_detection.txt        (mock)
# ├── Smith2023_calcium_imaging.pdf  (real)
# └── Jones2022_roi_detection.pdf    (real)
```

### Option B: Replace Mock Files with Real PDFs

**Best for**: Production use with real papers only

```bash
cd /home/xinra/NotreDame/Calcium_pipeline/data/papers

# Backup mock files (optional)
mkdir ../papers_backup
mv *.txt ../papers_backup/

# Add your PDFs
cp ~/Downloads/calcium_papers/*.pdf ./

# Result:
# data/papers/
# ├── Smith2023_calcium_imaging.pdf
# ├── Jones2022_roi_detection.pdf
# └── Lee2021_fluorescence_analysis.pdf
```

---

## 📥 Where to Find Papers

### ArXiv (Free, Pre-prints)
```bash
# Example: Download from arXiv
wget https://arxiv.org/pdf/2301.12345.pdf -O data/papers/ArxivPaper_CalciumImaging.pdf
```

### PubMed Central (Free, Published)
1. Go to https://www.ncbi.nlm.nih.gov/pmc/
2. Search: "calcium imaging two-photon"
3. Filter: "Free full text"
4. Download PDFs

### Your Institution's Library
- Most universities provide access to papers
- Download as PDF and place in `data/papers/`

### Recommended Topics to Search
```
"calcium imaging analysis"
"two-photon calcium imaging"
"calcium transient detection"
"ROI detection calcium imaging"
"fluorescence imaging segmentation"
"ΔF/F0 normalization"
"Suite2p calcium imaging"
"CaImAn calcium imaging"
```

---

## 🔧 Rebuilding the RAG Database

### Why Rebuild?

The vector database (ChromaDB) stores:
- Text chunks from papers
- Embeddings (vector representations)
- Metadata (sources, scores)

When you add new papers, you need to rebuild to include them.

### Method 1: Delete and Auto-Rebuild (Recommended)

```bash
# Delete old database
rm -rf /home/xinra/NotreDame/Calcium_pipeline/data/vector_db

# Next time you use the system, it auto-rebuilds
streamlit run app.py
```

### Method 2: Manual Rebuild with Test

```bash
# Delete old database
rm -rf ./data/vector_db

# Run test to trigger rebuild
python test_rag_papers.py
```

### Method 3: Use Main Script

```bash
python main.py --request "test" --images ./data/images --rebuild-rag
```

---

## ✅ Verification Steps

### 1. Check Files Are Present

```bash
ls -lh /home/xinra/NotreDame/Calcium_pipeline/data/papers/

# Should show your PDFs with file sizes
# Example output:
# -rw-r--r-- 1 user user 2.3M Oct 13 10:00 Smith2023.pdf
# -rw-r--r-- 1 user user 1.8M Oct 13 10:00 Jones2022.pdf
```

### 2. Run Test Script

```bash
python test_rag_papers.py
```

**Expected output**:
```
================================================================================
RAG SYSTEM TEST - Paper Loading
================================================================================

📁 Papers directory: data/papers
📄 Found 5 files:

  • baseline_calculation.txt (1.8 KB)
  • segmentation_methods.txt (1.6 KB)
  • transient_detection.txt (1.6 KB)
  • Smith2023_calcium_imaging.pdf (2300.5 KB)
  • Jones2022_roi_detection.pdf (1850.3 KB)

--------------------------------------------------------------------------------

🔧 Initializing RAG system...

Loading existing vector database from data/vector_db
[OR]
Building new vector database from papers
Loaded 3 text files and 2 PDF files (5 total documents)
Split into 327 chunks
Vector database created at data/vector_db

🔍 Testing retrieval with sample queries:

Query: 'calcium transient detection'
  Retrieved 3 chunks from:
    • transient_detection.txt
    • Smith2023_calcium_imaging.pdf
  Sample: Detecting calcium transients requires identifying periods of rapid fluorescence increase...

✅ RAG system test complete!
```

### 3. Test in Streamlit UI

```bash
streamlit run app.py
```

Ask an informational query:
```
What methods can I use for calcium transient detection?
```

**Check the "Execution Details"** - you should see:
```
📚 Retrieved knowledge from: transient_detection.txt, Smith2023_calcium_imaging.pdf
```

---

## 📊 What Happens During PDF Loading

### Behind the Scenes

1. **Text Extraction** ([layers/rag_system.py:94-99](layers/rag_system.py#L94-L99))
   ```python
   pdf_loader = DirectoryLoader(
       str(self.papers_dir),
       glob="*.pdf",
       loader_cls=PyPDFLoader  # Uses pypdf library
   )
   pdf_documents = pdf_loader.load()
   ```

2. **Chunking** ([layers/rag_system.py:110-112](layers/rag_system.py#L110-L112))
   ```python
   text_splitter = RecursiveCharacterTextSplitter(
       chunk_size=1000,      # 1000 characters per chunk
       chunk_overlap=200     # 200 characters overlap
   )
   ```

3. **Embedding** ([layers/rag_system.py:36-39](layers/rag_system.py#L36-L39))
   ```python
   embeddings = OpenAIEmbeddings(
       model="text-embedding-3-small"  # 1536 dimensions
   )
   ```

4. **Storage** ([layers/rag_system.py:107-111](layers/rag_system.py#L107-L111))
   ```python
   vectorstore = Chroma.from_documents(
       documents=splits,
       embedding=embeddings,
       persist_directory="./data/vector_db"
   )
   ```

### Cost Estimation

**Embedding cost**: ~$0.02 per 1 million tokens

Example:
- 5 papers × 30 pages × 500 words/page = 75,000 words
- ~100,000 tokens
- Cost: **~$0.002** (very cheap!)

---

## 🎨 Best Practices

### File Naming

✅ **Good names** (descriptive):
```
Smith2023_CalciumTransientDetection.pdf
Jones2022_TwoPhotonImagingMethods.pdf
Lee2021_ROI_Segmentation_Review.pdf
```

❌ **Bad names** (not descriptive):
```
paper1.pdf
download.pdf
unnamed.pdf
```

**Why**: Filenames appear as "sources" in RAG results!

### Paper Selection

**Choose papers that cover**:
- Calcium imaging analysis methods
- Image segmentation techniques
- Signal processing for fluorescence
- Specific algorithms (Suite2p, CaImAn, etc.)
- Statistical methods for calcium data

**Avoid papers that are**:
- Too general (broad neuroscience reviews)
- Hardware-focused (microscope engineering)
- Biology-heavy with minimal methods

### Organizing Papers

**Option 1: Single directory** (current setup)
```
data/papers/
├── all_papers_here.pdf
```

**Option 2: Subdirectories** (future enhancement)
```
data/papers/
├── methods/
├── reviews/
└── applications/
```

*Note*: Current system loads from one directory. Subdirectories would require code modification.

---

## 🐛 Troubleshooting

### Issue: "No .txt or .pdf files found"

**Cause**: Papers directory is empty or wrong path

**Fix**:
```bash
ls -la /home/xinra/NotreDame/Calcium_pipeline/data/papers/

# Should show files, not empty
```

### Issue: "Failed to load PDF"

**Cause**: Corrupted or password-protected PDF

**Fix**:
```bash
# Test if PDF is readable
pdfinfo your_paper.pdf

# If password-protected, unlock it first
# (pypdf doesn't handle encrypted PDFs)
```

### Issue: "Embedding failed"

**Cause**: OpenAI API key issue or rate limit

**Fix**:
```bash
# Check API key is set
echo $OPENAI_API_KEY

# Or check .env file
cat .env | grep OPENAI_API_KEY
```

### Issue: RAG retrieves irrelevant chunks

**Cause**: Papers don't contain relevant methods

**Solution**:
- Add more domain-specific papers
- Increase `TOP_K_CHUNKS` in [config.py](config.py#L51):
  ```python
  TOP_K_CHUNKS = 5  # Instead of 3
  ```

---

## 📈 After Adding Papers

### What Changes?

**Before** (mock files):
```
Query: "How to detect calcium transients?"
RAG sources: transient_detection.txt
Response: Basic methods (bandpass filtering, peak detection)
```

**After** (real papers):
```
Query: "How to detect calcium transients?"
RAG sources: Smith2023_calcium_imaging.pdf, transient_detection.txt, Jones2022.pdf
Response: State-of-the-art methods with citations, specific algorithms, parameter recommendations
```

### Quality Improvements

With real papers, the system will:
1. ✅ Cite specific research
2. ✅ Suggest validated parameters
3. ✅ Reference established algorithms
4. ✅ Provide implementation details
5. ✅ Show recent advances

---

## 🚀 Quick Reference

### Complete Workflow

```bash
# 1. Add papers
cd /home/xinra/NotreDame/Calcium_pipeline/data/papers
cp ~/my_papers/*.pdf ./

# 2. Rebuild database
cd /home/xinra/NotreDame/Calcium_pipeline
rm -rf ./data/vector_db

# 3. Test
python test_rag_papers.py

# 4. Use in Streamlit
streamlit run app.py
```

### Files Involved

| File/Directory | Purpose |
|----------------|---------|
| `data/papers/` | Place PDFs here |
| `data/vector_db/` | Auto-generated embeddings (delete to rebuild) |
| `layers/rag_system.py` | Handles PDF loading & retrieval |
| `config.py` | Settings (chunk size, top-k, etc.) |
| `test_rag_papers.py` | Test script to verify loading |

---

## 💡 Pro Tips

1. **Start small**: Add 5-10 papers, test, then add more
2. **Check sources**: In Streamlit, expand "Execution Details" to see which papers were used
3. **Quality over quantity**: 5 high-quality methods papers > 50 general reviews
4. **Update regularly**: Add new papers as field advances
5. **Backup your database**: Once built, `tar -czf vector_db_backup.tar.gz data/vector_db`

---

## 📞 Need Help?

If you encounter issues:

1. Run test script: `python test_rag_papers.py`
2. Check logs: `cat pipeline.log`
3. Verify API key: `cat .env`
4. Check paper count: `ls -lh data/papers/`

**Everything should work automatically once PDFs are in place!** 🎉
