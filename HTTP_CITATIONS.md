# HTTP Citation Links - Implementation Guide

**Date:** October 25, 2025

## Overview

The citation system has been updated to serve PDFs via HTTP instead of `file://` URLs, making citations clickable in web-based interfaces like Open WebUI.

## Problem Solved

**Previous Issue:** Citations used `file://` URLs which are blocked by browsers for security reasons (Same-Origin Policy), making links non-clickable in Open WebUI.

**New Solution:** PDFs are served via HTTP through the FastAPI backend, generating clickable `http://` URLs that work in all browsers.

## How It Works

### 1. PDF Serving Endpoint

A new endpoint has been added to `api_backend.py`:

```python
@app.get("/papers/{filename:path}")
async def serve_paper(filename: str):
    """Serve PDF papers via HTTP for clickable citations."""
    # Security: Ensures path is within PAPERS_DIR (prevents path traversal)
    # Returns: FileResponse with PDF content
```

**URL Format:** `http://localhost:8000/papers/paper.pdf`

**Features:**
- ✅ Serves PDFs from `data/papers/` directory
- ✅ URL-encodes filenames (handles spaces and special characters)
- ✅ Security checks prevent path traversal attacks
- ✅ Returns proper PDF MIME type (`application/pdf`)
- ✅ Works with subdirectories (e.g., `/papers/subdir/paper.pdf`)

### 2. Citation Formatter Updates

The `CitationFormatter` class now generates HTTP URLs:

```python
# Before (file:// URL - not clickable in browser)
file_url = f"file:///home/user/papers/paper.pdf"

# After (HTTP URL - clickable in browser)
paper_url = f"http://localhost:8000/papers/paper.pdf"
```

**Smart URL Generation:**
- Checks for `API_SERVER_URL` configuration
- Falls back to `http://localhost:8000` by default
- Handles special characters via URL encoding

### 3. Configuration

New configuration option in `config.py`:

```python
API_SERVER_URL = os.getenv("API_SERVER_URL", "http://localhost:8000")
```

**Environment Variable:** Add to `.env` file:

```bash
# Local development
API_SERVER_URL=http://localhost:8000

# Remote access (e.g., from another machine)
API_SERVER_URL=http://192.168.1.100:8000

# Docker container
API_SERVER_URL=http://host.docker.internal:8000
```

## Output Format

### Previous Format (file:// - NOT clickable)

```markdown
## References

1. [Calcium signaling dynamics.pdf](file:///home/xinra/NotreDame/Calcium_pipeline/data/papers/Calcium%20signaling%20dynamics.pdf)
2. [Spatial aspects of calcium.pdf](file:///home/xinra/NotreDame/Calcium_pipeline/data/papers/Spatial%20aspects%20of%20calcium.pdf)
```

### New Format (http:// - CLICKABLE)

```markdown
## References

1. [Calcium signaling dynamics.pdf](http://localhost:8000/papers/Calcium%20signaling%20dynamics.pdf)
2. [Spatial aspects of calcium.pdf](http://localhost:8000/papers/Spatial%20aspects%20of%20calcium.pdf)
```

## Usage

### Starting the API Server

The PDF serving endpoint is automatically available when you run the FastAPI backend:

```bash
# Start the API server
python api_backend.py

# Server runs on http://0.0.0.0:8000
# PDF endpoint: http://localhost:8000/papers/
```

### Testing the Endpoint

**Method 1: Browser**
```
http://localhost:8000/papers/your-paper.pdf
```

**Method 2: curl**
```bash
curl -I http://localhost:8000/papers/your-paper.pdf
# Should return: Content-Type: application/pdf
```

**Method 3: Python**
```python
import requests

response = requests.get("http://localhost:8000/papers/your-paper.pdf")
print(f"Status: {response.status_code}")
print(f"Content-Type: {response.headers['Content-Type']}")
# Expected: Status: 200, Content-Type: application/pdf
```

### Example Query Flow

1. **User asks question:**
   ```
   "Explain calcium oscillations"
   ```

2. **RAG retrieves papers and shows citations immediately:**
   ```markdown
   📚 **Retrieved from 5 papers**

   ## References

   1. [Generation, control, and processing of cellular calcium signals.pdf](http://localhost:8000/papers/Generation%2C%20control%2C%20and%20processing%20of%20cellular%20calcium%20signals.pdf)
   2. [Calcium signalling dynamics, homeostasis and remodelling.pdf](http://localhost:8000/papers/Calcium%20signalling%20dynamics%2C%20homeostasis%20and%20remodelling.pdf)
   3. [Spatial and temporal aspects of cellular calcium signaling.pdf](http://localhost:8000/papers/Spatial%20and%20temporal%20aspects%20of%20cellular%20calcium%20signaling.pdf)
   4. [Fundamentals of Cellular Calcium Signaling A Primer.pdf](http://localhost:8000/papers/Fundamentals%20of%20Cellular%20Calcium%20Signaling%20A%20Primer.pdf)
   5. [Function- and agonist-specific Ca2+ signalling.pdf](http://localhost:8000/papers/Function-%20and%20agonist-specific%20Ca2%2B%20signalling.pdf)
   ```

3. **LLM synthesizes answer with short citations:**
   ```markdown
   ## Answer

   Calcium oscillations are rhythmic changes in intracellular Ca²⁺ concentration.
   According to [1], these oscillations encode information in their frequency and amplitude...

   As described in [2] and [3], the oscillation frequency typically ranges from...
   ```

4. **User clicks on `[1]` in the References section** → PDF opens in browser

## Security

### Path Traversal Protection

The endpoint includes security checks to prevent directory traversal attacks:

```python
# Attacker tries: http://localhost:8000/papers/../../etc/passwd
# Result: 403 Forbidden (path must be within PAPERS_DIR)

# Allowed: http://localhost:8000/papers/paper.pdf
# Allowed: http://localhost:8000/papers/subdir/paper.pdf
# Blocked: http://localhost:8000/papers/../config.py
```

### CORS Configuration

CORS is enabled for all origins in development:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Change for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Production Recommendation:** Restrict origins to your Open WebUI domain:

```python
allow_origins=["http://your-open-webui-domain.com"]
```

## Troubleshooting

### Issue 1: Links Still Show file:// URLs

**Cause:** Configuration not loaded

**Fix:**
1. Check `.env` file has `API_SERVER_URL=http://localhost:8000`
2. Restart the API server
3. Run a new query (old cached results may still have file:// URLs)

### Issue 2: 404 Not Found When Clicking Link

**Cause:** Paper file doesn't exist or server not running

**Fix:**
1. Verify API server is running: `curl http://localhost:8000/health`
2. Check paper exists: `ls data/papers/your-paper.pdf`
3. Check server logs for error messages

### Issue 3: PDF Opens as Download Instead of Viewer

**Cause:** Browser setting or MIME type issue

**Note:** This is browser-dependent behavior. Most browsers will:
- Chrome/Edge: Open PDF in built-in viewer
- Firefox: Open PDF in built-in viewer
- Safari: Download or open in Preview

**This is expected behavior and not an error.**

### Issue 4: Links Work Locally But Not from Remote Client

**Cause:** Using `localhost` URL from different machine

**Fix:** Set `API_SERVER_URL` to machine's IP address:

```bash
# In .env on server machine
API_SERVER_URL=http://192.168.1.100:8000
```

Then restart the API server.

### Issue 5: Special Characters in Filename Break Link

**Cause:** Filename not properly URL-encoded

**Status:** ✅ Already handled by `quote()` function in citation formatter

**Example:**
- Filename: `Calcium & Signaling (2020).pdf`
- URL: `http://localhost:8000/papers/Calcium%20%26%20Signaling%20%282020%29.pdf`

## Files Modified

| File | Changes | Purpose |
|------|---------|---------|
| [api_backend.py](api_backend.py) | Added `/papers/{filename}` endpoint | Serve PDFs via HTTP |
| [core/citation_formatter.py](core/citation_formatter.py) | Added `_get_paper_url()` method | Generate HTTP URLs instead of file:// |
| [config.py](config.py) | Added `API_SERVER_URL` setting | Configure server URL |
| [.env.example](.env.example) | Documented `API_SERVER_URL` | User guidance |

## Testing Checklist

- [x] PDF endpoint returns correct MIME type
- [x] Security checks prevent path traversal
- [x] URL encoding handles special characters
- [x] Citations generate HTTP URLs
- [ ] Links clickable in Open WebUI (requires user testing)
- [ ] PDF opens in browser when clicked (requires user testing)

## Next Steps

1. **Test in Open WebUI:**
   - Start API server: `python api_backend.py`
   - Run a query in Open WebUI
   - Click a citation link
   - Verify PDF opens in browser

2. **Configure for Remote Access (if needed):**
   - Find server IP: `ip addr show` or `ifconfig`
   - Set `API_SERVER_URL=http://<your-ip>:8000` in `.env`
   - Restart API server
   - Test from remote client

3. **Production Deployment:**
   - Update CORS settings to restrict origins
   - Use reverse proxy (nginx/apache) for SSL
   - Set production URL in environment

## Benefits

### For Users
✅ **Clickable citations** - One click to view source paper
✅ **Works in browser** - No need to copy/paste file paths
✅ **Cross-platform** - Works on any device with browser
✅ **Transparent sources** - Easy verification of information

### For Developers
✅ **Standards-compliant** - Uses HTTP instead of non-standard file://
✅ **Secure** - Built-in path traversal protection
✅ **Flexible** - Configurable server URL for different environments
✅ **Compatible** - Works with all web-based UIs

## Related Documentation

- [CITATION_FINAL_FIXES.md](CITATION_FINAL_FIXES.md) - Citation system improvements
- [CITATION_FIX_SUMMARY.md](CITATION_FIX_SUMMARY.md) - Implementation summary
- [CITATION_SYSTEM.md](CITATION_SYSTEM.md) - Complete citation system docs
- [api_backend.py](api_backend.py) - FastAPI backend implementation

---

**Status:** ✅ Implemented and ready for testing
**Last Updated:** October 25, 2025
**Compatibility:** Open WebUI, Streamlit, CLI, any HTTP client
