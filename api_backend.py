#!/usr/bin/env python3
"""
FastAPI Backend for Calcium Imaging Analysis
OpenAI-compatible API for integration with Open WebUI
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, AsyncIterator
import asyncio
import json
import time
import logging
from datetime import datetime
import uuid
from pathlib import Path

# Import workflow components
import config
from core.logging_config import setup_logging
from graph.workflow import create_workflow
from graph.state import PipelineState
from core.session_manager import get_session_manager
from core.streaming_progress import StreamingProgressReporter, ProgressFormatter

# Setup logging - verbose mode to capture everything
logger = setup_logging(
    log_level="DEBUG",  # Changed to DEBUG to capture everything
    log_file="api_backend.log",
    verbose=True
)

# Create FastAPI app
app = FastAPI(
    title="Calcium Imaging Analysis API",
    description="AI-powered calcium imaging analysis with agentic workflow",
    version="2.0.0"
)

# Add CORS middleware for Open WebUI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache workflow
workflow_instance = None

def get_workflow():
    """Get or create cached workflow instance."""
    global workflow_instance
    if workflow_instance is None:
        logger.info("Creating workflow instance...")
        workflow_instance = create_workflow()
        logger.info("Workflow instance created")
    return workflow_instance


# ===== Request/Response Models (OpenAI-compatible) =====

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "calcium-imaging-v1"
    messages: List[Message]
    stream: bool = False
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    # Custom parameters
    images_path: str = Field(default="./data/images", description="Path to image frames")
    session_id: Optional[str] = None

class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[Dict[str, int]] = None

class ChatCompletionChunkDelta(BaseModel):
    content: Optional[str] = None
    role: Optional[str] = None

class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None

class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]


# ===== Helper Functions =====

def extract_user_message(messages: List[Message]) -> str:
    """
    Extract the last user message from conversation.
    Returns None if only system tasks are present (to be handled gracefully).
    """
    # Log all messages for debugging
    logger.info(f"Received {len(messages)} messages:")
    for i, msg in enumerate(messages):
        content_preview = msg.content[:100] if len(msg.content) > 100 else msg.content
        logger.info(f"  Message {i}: role={msg.role}, content={content_preview}...")

    for msg in reversed(messages):
        if msg.role == "user":
            # Filter out Open WebUI system messages (title generation, follow-up suggestions, tagging, etc.)
            content = msg.content.strip()

            # Detect any Open WebUI system task (starts with "### Task:")
            if content.startswith("### Task:"):
                # Check for specific system tasks
                if any(keyword in content for keyword in [
                    "Generate a concise",  # Title generation
                    "title with an emoji",  # Title generation
                    "broad tags categorizing",  # Tagging
                    "follow-up question",  # Follow-ups
                    "suggest 3-5 relevant",  # Suggestions
                    "Suggest 3-5 relevant",  # Suggestions (capital S)
                    "Chat History:",  # Most system tasks include this
                ]):
                    logger.warning(f"[FILTER] Ignoring Open WebUI system task: {content[:80]}...")
                    continue

            return msg.content

    # Return None if only system tasks were found (will be handled gracefully)
    return None

def format_progress_event(event_type: str, data: Dict[str, Any]) -> str:
    """Format progress event as readable text for streaming."""
    if event_type == "clarifier":
        return f"💭 **Clarifying request...**\n_{data.get('clarified_request', '')}_\n\n"

    elif event_type == "planner":
        steps = data.get('num_steps', 0)
        mode = data.get('execution_mode', 'unknown')
        return f"📋 **Planning execution** ({steps} steps, {mode} mode)\n\n"

    elif event_type == "rag_retrieval":
        sources = data.get('sources', [])
        unique_sources = list(set(sources))[:3]
        return f"📚 **Retrieved knowledge** from {len(unique_sources)} papers\n\n"

    elif event_type == "step_start":
        step_id = data.get('step_id', '')
        description = data.get('description', '')
        return f"▶️ **{step_id}**: {description}\n"

    elif event_type == "step_complete":
        step_id = data.get('step_id', '')
        verified = data.get('verified', False)
        status = "✓ Verified" if verified else "✓ Complete"
        return f"{status} **{step_id}**\n\n"

    elif event_type == "capability_reused":
        step_id = data.get('step_id', '')
        cap_id = data.get('cap_id', 'unknown')
        similarity = data.get('similarity', 0)
        return f"♻️ **Reusing capability** for {step_id} (similarity: {similarity:.0%})\n  ↪ Skipping code generation (10x faster!)\n\n"

    elif event_type == "retry_with_error_feedback":
        step_id = data.get('step_id', '')
        attempt = data.get('attempt', 1)
        error = data.get('error', 'Unknown error')[:100]
        return f"🔄 **Retrying {step_id}** (attempt {attempt})\n  ↪ Previous error: {error}...\n\n"

    elif event_type == "step_failed":
        step_id = data.get('step_id', '')
        error = data.get('error', 'Unknown error')
        return f"✗ **{step_id}** failed: {error}\n\n"

    elif event_type == "verification":
        if data.get('passed'):
            confidence = data.get('confidence', 0)
            return f"🔍 Verification passed (confidence: {confidence:.0%})\n\n"
        else:
            issues = data.get('issues', [])
            return f"⚠️ Verification issues:\n" + "\n".join(f"  • {issue}" for issue in issues) + "\n\n"

    return ""

def format_final_result(result: Any) -> str:
    """Format final analysis result as markdown."""
    if not result:
        return "❌ Analysis failed: No result produced"

    status = result.metadata.get("status", "unknown")

    if status == "failed":
        error = result.data.get("error", "Unknown error")
        return f"❌ **Analysis failed**\n\n{error}"

    # Format success response
    # Note: The answer is already streamed during synthesis, so we don't add it here again
    # to avoid duplication. Only add execution results if present.
    response = ""

    # Don't add summary here - it's already been streamed during synthesis
    # if hasattr(result, 'summary') and result.summary:
    #     response += f"{result.summary}\n"

    # Only show "Results" section if there are actual execution results (not for RAG-only queries)
    if result.data and any(key not in ['summary', 'answer'] for key in result.data.keys()):
        response += "\n## Execution Results\n\n"
        for key, value in result.data.items():
            if key not in ['summary', 'answer']:  # Skip summary/answer metadata
                if isinstance(value, float):
                    response += f"- {key}: {value:.4f}\n"
                elif isinstance(value, (int, str)):
                    response += f"- {key}: {value}\n"

    # Add capability reuse info
    if result.metadata.get("capability_reused"):
        similarity = result.metadata.get("capability_similarity", 0)
        response += f"\n♻️ Reused existing capability (similarity: {similarity:.2f})\n"

    return response


# ===== API Endpoints =====

@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "name": "Calcium Imaging Analysis API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "workflow": "initialized" if workflow_instance else "not_initialized"
    }

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "calcium-imaging-v1",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "calcium-imaging-system",
                "permission": [],
                "root": "calcium-imaging-v1",
                "parent": None,
            }
        ]
    }

@app.get("/papers/{filename:path}")
async def serve_paper(filename: str):
    """
    Serve PDF papers via HTTP for clickable citations.

    Args:
        filename: Paper filename or relative path (e.g., "paper.pdf")

    Returns:
        FileResponse with PDF content
    """
    try:
        # Construct full path to paper
        paper_path = Path(config.PAPERS_DIR) / filename

        # Security check: Ensure path is within PAPERS_DIR
        resolved_path = paper_path.resolve()
        papers_dir_resolved = Path(config.PAPERS_DIR).resolve()

        if not str(resolved_path).startswith(str(papers_dir_resolved)):
            logger.warning(f"Attempted path traversal attack: {filename}")
            raise HTTPException(status_code=403, detail="Access denied")

        # Check if file exists
        if not resolved_path.exists():
            logger.warning(f"Paper not found: {filename} (path: {resolved_path})")
            raise HTTPException(status_code=404, detail=f"Paper not found: {filename}")

        # Check if it's a file (not directory)
        if not resolved_path.is_file():
            logger.warning(f"Not a file: {filename}")
            raise HTTPException(status_code=400, detail="Not a valid file")

        logger.info(f"Serving paper: {filename}")

        # Return PDF file with inline display (not download)
        # Content-Disposition: inline tells browser to display in-page
        return FileResponse(
            path=str(resolved_path),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'inline; filename="{Path(filename).name}"'
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error serving paper {filename}")
        raise HTTPException(status_code=500, detail=f"Error serving paper: {str(e)}")

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Chat completions endpoint (OpenAI-compatible).

    Supports both streaming and non-streaming responses.
    """
    logger.info(f"Received chat request: model={request.model}, stream={request.stream}")

    try:
        # Log the raw incoming request for debugging
        logger.info("=" * 80)
        logger.info("INCOMING REQUEST FROM OPEN WEBUI")
        logger.info("=" * 80)
        logger.info(f"Total messages received: {len(request.messages)}")
        for i, msg in enumerate(request.messages):
            logger.info(f"\nMessage {i}:")
            logger.info(f"  Role: {msg.role}")
            logger.info(f"  Content length: {len(msg.content)} chars")
            logger.info(f"  Content preview (first 200 chars):")
            logger.info(f"    {msg.content[:200]}...")
            if len(msg.content) > 200:
                logger.info(f"  Content preview (last 200 chars):")
                logger.info(f"    ...{msg.content[-200:]}")
        logger.info("=" * 80)

        # Extract user message
        user_message = extract_user_message(request.messages)

        # If only system tasks were present, return an empty response
        if user_message is None:
            logger.warning("Only Open WebUI system tasks detected - returning empty response")
            # Return a minimal valid response
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": ""
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }

        logger.info(f"Extracted user message: {user_message[:100]}...")

        # Generate session ID if not provided
        session_id = request.session_id or f"session_{uuid.uuid4().hex[:8]}"

        if request.stream:
            # Streaming response
            return StreamingResponse(
                stream_workflow_response(user_message, request, session_id),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming response
            return await run_workflow_sync(user_message, request, session_id)

    except Exception as e:
        logger.exception("Error processing chat completion")
        raise HTTPException(status_code=500, detail=str(e))


async def stream_workflow_response(
    user_message: str,
    request: ChatCompletionRequest,
    session_id: str
) -> AsyncIterator[str]:
    """
    Stream workflow execution progress as Server-Sent Events.

    Compatible with OpenAI's streaming format with real-time character streaming.
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    # Create streaming progress reporter
    streaming_reporter = StreamingProgressReporter()
    formatter = ProgressFormatter()

    # Create output directory for this session
    import tempfile
    output_path = tempfile.mkdtemp(prefix=f"calcium_output_{session_id}_")

    # Create initial state with streaming reporter
    initial_state: PipelineState = {
        "user_request": user_message,
        "images_path": request.images_path,
        "tool_plan": None,
        "user_approval": None,
        "waiting_for_approval": False,
        "tool_outputs": {},
        "current_error": None,
        "waiting_for_error_response": False,
        "errors": [],
        "final_output": None,
        "progress_callback": None,  # Using streaming_reporter instead
        "interrupted": False,
        "session_id": session_id,
        "output_path": output_path,
        "streaming_reporter": streaming_reporter  # Pass streaming reporter
    }

    # Run workflow in executor (LangGraph is synchronous)
    loop = asyncio.get_event_loop()
    workflow = get_workflow()

    try:
        # Start workflow execution
        future = loop.run_in_executor(None, workflow.invoke, initial_state)

        # Stream progress events as they arrive from the queue
        while not future.done():
            await asyncio.sleep(0.05)  # Check every 50ms for more responsive streaming

            # Get new events from streaming reporter queue
            events = streaming_reporter.get_events(timeout=0.01)

            for event in events:
                content = event.get("content", "")
                if content:
                    chunk = ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=request.model,
                        choices=[ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(content=content),
                            finish_reason=None
                        )]
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

        # Get final result
        final_state = await future
        result = final_state.get("final_output")

        # Stream any remaining events
        remaining_events = streaming_reporter.get_events(timeout=0.01)
        for event in remaining_events:
            content = event.get("content", "")
            if content:
                chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(content=content),
                        finish_reason=None
                    )]
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

        # Stream final result
        final_content = format_final_result(result)
        final_chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(content=final_content),
                finish_reason="stop"
            )]
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.exception("Error during workflow execution")
        error_chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(content=f"\n\n❌ **Error:** {str(e)}"),
                finish_reason="error"
            )]
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"


async def run_workflow_sync(
    user_message: str,
    request: ChatCompletionRequest,
    session_id: str
) -> ChatCompletionResponse:
    """
    Run workflow synchronously and return complete response.

    For non-streaming requests.
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    # Create output directory for this session
    import tempfile
    output_path = tempfile.mkdtemp(prefix=f"calcium_output_{session_id}_")

    # Create initial state (no progress callback for sync)
    initial_state: PipelineState = {
        "user_request": user_message,
        "images_path": request.images_path,
        "tool_plan": None,
        "user_approval": None,
        "waiting_for_approval": False,
        "tool_outputs": {},
        "current_error": None,
        "waiting_for_error_response": False,
        "errors": [],
        "final_output": None,
        "progress_callback": None,
        "interrupted": False,
        "session_id": session_id,
        "output_path": output_path,
        "streaming_reporter": None
    }

    # Run workflow
    loop = asyncio.get_event_loop()
    workflow = get_workflow()

    try:
        final_state = await loop.run_in_executor(None, workflow.invoke, initial_state)
        result = final_state.get("final_output")

        # Format response
        content = format_final_result(result)

        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[ChatCompletionChoice(
                index=0,
                message=Message(role="assistant", content=content),
                finish_reason="stop"
            )],
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        )

    except Exception as e:
        logger.exception("Error during workflow execution")
        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[ChatCompletionChoice(
                index=0,
                message=Message(role="assistant", content=f"❌ **Error:** {str(e)}"),
                finish_reason="error"
            )]
        )


if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 80)
    logger.info("STARTUP: Calcium Imaging Analysis API")
    logger.info("=" * 80)
    logger.info(f"Model: {config.OPENAI_MODEL}")
    logger.info(f"Router Model: {config.ROUTER_MODEL}")
    logger.info(f"Log Level: DEBUG")
    logger.info(f"Log File: api_backend.log (overwrites on each start)")
    logger.info("Initializing workflow...")
    get_workflow()  # Pre-initialize
    logger.info("Workflow initialized successfully")
    logger.info("=" * 80)
    logger.info("API READY - Listening on http://0.0.0.0:8000")
    logger.info("=" * 80)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
