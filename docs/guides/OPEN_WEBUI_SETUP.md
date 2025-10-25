# Open WebUI + FastAPI Backend Setup Guide

This guide explains how to set up Open WebUI to work with the Calcium Imaging Analysis system.

## Architecture Overview

```
User <-> Open WebUI <-> FastAPI Backend (api_backend.py) <-> LangGraph Workflow <-> Codegen Sandbox

Admin <-> Streamlit (app.py) - System monitoring & management
```

## Prerequisites

1. Python 3.9+ with FastAPI dependencies installed
2. Docker (for running Open WebUI)
3. Codegen API key (sign up at https://codegen.com/)
4. OpenAI API key

## Step 1: Configure Environment Variables

Update your `.env` file with the following:

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Codegen API Configuration (replaces E2B)
CODEGEN_API_KEY=your_codegen_api_key_here

# Sandbox Configuration
CLOSE_SANDBOX_AFTER_EXECUTION=true  # Recommended for cost savings
SANDBOX_REUSE_ENABLED=false
SANDBOX_MAX_IDLE_MINUTES=5
```

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- FastAPI for the backend server
- Uvicorn for ASGI server
- All existing dependencies

## Step 3: Start the FastAPI Backend

```bash
python api_backend.py
```

This starts the server on `http://0.0.0.0:8000`

**Verify it's running:**
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-10-14T...",
  "workflow": "initialized"
}
```

## Step 4: Install and Run Open WebUI

### Option A: Docker (Recommended)

```bash
docker run -d \
  --name open-webui \
  -p 3000:8080 \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEY=not-needed \
  -v open-webui:/app/backend/data \
  ghcr.io/open-webui/open-webui:main
```

**Note for Linux users**: Replace `host.docker.internal` with your machine's IP address.

### Option B: Local Installation

```bash
pip install open-webui
open-webui serve --port 3000
```

Then configure the OpenAI API settings in the web UI:
- API Base URL: `http://localhost:8000/v1`
- API Key: `not-needed` (any value works)

## Step 5: Access Open WebUI

1. Open browser to `http://localhost:3000`
2. Create an account (stored locally)
3. Go to Settings → Connections
4. Verify the connection shows "calcium-imaging-v1" model

## Step 6: Test the Integration

Send a test message:
```
Analyze calcium imaging data from ./data/images
```

You should see:
- Real-time streaming updates showing progress
- Clarification, planning, and execution steps
- Final results with visualizations

## Using Streaming vs Non-Streaming

### Streaming Mode (Default in Open WebUI)
- Real-time progress updates
- See each step as it happens
- Better user experience

### Non-Streaming Mode
- Wait for complete result
- Faster for programmatic access
- Use for API integrations

## API Endpoints

### Chat Completion (OpenAI-compatible)
```bash
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "calcium-imaging-v1",
  "messages": [
    {"role": "user", "content": "Detect calcium transients"}
  ],
  "stream": true,
  "images_path": "./data/images",
  "session_id": "optional-session-id"
}
```

### List Available Models
```bash
GET /v1/models
```

### Health Check
```bash
GET /health
```

## Custom Parameters

The API supports custom parameters in addition to OpenAI's standard ones:

- `images_path`: Path to calcium imaging frames (default: `./data/images`)
- `session_id`: Optional session ID for sandbox reuse

Example:
```json
{
  "model": "calcium-imaging-v1",
  "messages": [...],
  "images_path": "/path/to/your/images",
  "session_id": "user123_session1"
}
```

## Streamlit Admin Panel

While Open WebUI is for end users, Streamlit provides admin functionality:

```bash
streamlit run app.py
```

Access at `http://localhost:8501`

**Admin features:**
- View system logs and metrics
- Manage RAG knowledge base (add/remove papers)
- Monitor sandbox usage and costs
- View execution history
- Manage sessions

## Migration from E2B to Codegen

The system is designed to support both E2B and Codegen executors.

**Current status:**
- E2B: Fully implemented but deprecated due to reliability issues
- Codegen: Placeholder ready, needs API key to activate

**To switch to Codegen:**
1. Get API key from https://codegen.com/
2. Add `CODEGEN_API_KEY` to `.env`
3. Update `core/codegen_executor.py` with actual SDK calls
4. System will automatically prefer Codegen over E2B

**Advantages of Codegen:**
- Persistent sandboxes (no timeouts)
- File system persistence between executions
- No connection/health check issues
- Production-grade reliability
- Better for multi-step workflows

## Troubleshooting

### Open WebUI can't connect to backend
- Check FastAPI is running: `curl http://localhost:8000/health`
- Check firewall isn't blocking port 8000
- For Docker on Linux, use machine IP instead of `host.docker.internal`

### "Model not found" error
- Verify `/v1/models` endpoint returns data
- Check API Base URL in Open WebUI settings ends with `/v1`

### Streaming not working
- Ensure Open WebUI streaming is enabled in chat settings
- Check browser console for SSE connection errors

### High costs from sandboxes
- Ensure `CLOSE_SANDBOX_AFTER_EXECUTION=true` in `.env`
- Monitor running sandboxes in Streamlit admin panel
- Use session-based reuse only for active conversations

## Performance Tips

1. **Sandbox Management**: Keep `CLOSE_SANDBOX_AFTER_EXECUTION=true` unless you need immediate follow-up queries
2. **RAG Optimization**: Limit papers to most relevant (~100 max) for faster retrieval
3. **Workflow Caching**: The FastAPI backend caches the workflow to avoid recreation overhead
4. **Session Reuse**: Use consistent `session_id` for conversations to reuse context

## Security Considerations

1. **API Access**: The backend has no authentication. Use reverse proxy with auth for production.
2. **Sandbox Isolation**: Code executes in isolated Codegen sandboxes (secure by design)
3. **File Access**: Backend can only access files it has permissions for
4. **CORS**: Currently allows all origins (`*`). Configure `allow_origins` in production.

## Next Steps

1. Add more scientific papers to RAG: See [ADDING_PAPERS_GUIDE.md](ADDING_PAPERS_GUIDE.md)
2. Customize execution plans: Modify planner prompts in `core/planner.py`
3. Add new analysis capabilities: Extend `graph/nodes.py`
4. Deploy to production: Add authentication, monitoring, and rate limiting

## Support

- System logs: Check `api_backend.log` and Streamlit console
- Workflow visualization: Use LangGraph Studio (optional)
- Paper management: Use Streamlit admin panel
