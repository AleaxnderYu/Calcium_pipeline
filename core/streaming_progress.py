"""
Real-time streaming progress system with character-by-character updates.

Similar to ChatGPT's streaming interface - provides live updates as LLM generates responses.
"""

import logging
from typing import Dict, Any, Optional, Callable, Iterator
from datetime import datetime
import asyncio
from queue import Queue, Empty
import threading

logger = logging.getLogger(__name__)


class StreamingProgressReporter:
    """
    Progress reporter with real-time character-by-character streaming.

    Features:
    - Character-by-character streaming for LLM responses
    - Structured progress events (planning, RAG, execution)
    - Thread-safe event queue
    - Compatible with SSE (Server-Sent Events)
    """

    def __init__(self, callback: Optional[Callable[[str, str], None]] = None):
        """
        Initialize streaming progress reporter.

        Args:
            callback: Optional callback(event_type, content) for real-time updates
        """
        self.callback = callback
        self.event_queue = Queue()
        self._lock = threading.Lock()

    def stream_text(self, text: str, event_type: str = "text", chunk_size: int = 1):
        """
        Stream text character-by-character or in small chunks.

        Args:
            text: Text to stream
            event_type: Type of event (planning, code_generation, verification, etc.)
            chunk_size: Number of characters per chunk (1 for char-by-char)
        """
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            self._emit(event_type, chunk)

    def emit_event(self, event_type: str, content: str):
        """
        Emit a complete event (not character-streamed).

        Args:
            event_type: Type of event
            content: Complete content to emit
        """
        self._emit(event_type, content)

    def stream_llm_response(self, event_type: str, stream_iterator: Iterator[str]):
        """
        Stream LLM response chunks in real-time.

        Args:
            event_type: Event type (code_generation, planning, etc.)
            stream_iterator: Iterator yielding LLM response chunks
        """
        for chunk in stream_iterator:
            if chunk:
                self._emit(event_type, chunk)

    def _emit(self, event_type: str, content: str):
        """
        Emit a progress event to queue and callback.

        Args:
            event_type: Type of event
            content: Content chunk
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "content": content
        }

        # Add to queue for async consumers
        self.event_queue.put(event)

        # Call synchronous callback if provided
        if self.callback:
            try:
                self.callback(event_type, content)
            except Exception as e:
                logger.error(f"Streaming callback error: {e}")

    def get_events(self, timeout: float = 0.1) -> list:
        """
        Get all queued events (non-blocking).

        Args:
            timeout: Max time to wait for first event

        Returns:
            List of events
        """
        events = []
        try:
            # Get first event with timeout
            first_event = self.event_queue.get(timeout=timeout)
            events.append(first_event)

            # Get remaining events without blocking
            while True:
                try:
                    event = self.event_queue.get_nowait()
                    events.append(event)
                except Empty:
                    break

        except Empty:
            pass

        return events

    def has_events(self) -> bool:
        """Check if there are queued events."""
        return not self.event_queue.empty()


class ProgressFormatter:
    """Formats progress events for display."""

    @staticmethod
    def format_tool_start(tool_id: str, tool_type: str, description: str) -> str:
        """Format tool execution start."""
        icons = {
            "rag": "📚",
            "code_generation": "💻",
            "execute": "⚙️",
            "verify": "🔍",
            "capability_search": "🔎",
            "capability_save": "💾"
        }
        icon = icons.get(tool_type, "▶️")
        return f"\n{icon} **{tool_id}**: {description}\n"

    @staticmethod
    def format_tool_complete(tool_id: str, success: bool = True) -> str:
        """Format tool execution completion."""
        if success:
            return f"✓ **{tool_id}** complete\n\n"
        else:
            return f"✗ **{tool_id}** failed\n\n"

    @staticmethod
    def format_plan_header(num_tools: int, mode: str) -> str:
        """Format plan creation header."""
        return f"\n📋 **Plan Created** ({num_tools} tools, {mode} mode)\n\n"

    @staticmethod
    def format_rag_retrieval(num_papers: int, sources: list) -> str:
        """Format RAG retrieval summary with full list."""
        # Show ALL papers, not just first 3
        source_names = ", ".join(sources)
        return f"📚 Retrieved from **{num_papers} papers**: {source_names}\n\n"

    @staticmethod
    def format_code_generation_start() -> str:
        """Format code generation start."""
        return "\n💻 **Generating code...**\n```python\n"

    @staticmethod
    def format_code_generation_end() -> str:
        """Format code generation end."""
        return "\n```\n\n"

    @staticmethod
    def format_execution_start() -> str:
        """Format execution start."""
        return "\n⚙️ **Executing code in Docker sandbox...**\n\n"

    @staticmethod
    def format_execution_output(output: str) -> str:
        """Format execution output."""
        return f"```\n{output}\n```\n\n"

    @staticmethod
    def format_verification_start() -> str:
        """Format verification start."""
        return "\n🔍 **Verifying results...**\n"

    @staticmethod
    def format_verification_result(passed: bool, confidence: float, issues: list = None) -> str:
        """Format verification result."""
        if passed:
            return f"✓ Verification passed (confidence: {confidence:.0%})\n\n"
        else:
            result = f"⚠️ Verification issues (confidence: {confidence:.0%}):\n"
            if issues:
                for issue in issues:
                    result += f"  • {issue}\n"
            return result + "\n"

    @staticmethod
    def format_error(error_msg: str) -> str:
        """Format error message."""
        return f"\n❌ **Error**: {error_msg}\n\n"

    @staticmethod
    def format_retry(attempt: int, max_attempts: int) -> str:
        """Format retry attempt."""
        return f"\n🔄 Retrying... (attempt {attempt}/{max_attempts})\n\n"


# Singleton for global access
_streaming_reporter = None


def get_streaming_reporter(callback: Optional[Callable[[str, str], None]] = None) -> StreamingProgressReporter:
    """
    Get or create streaming progress reporter singleton.

    Args:
        callback: Optional callback for real-time updates

    Returns:
        StreamingProgressReporter instance
    """
    global _streaming_reporter
    if _streaming_reporter is None or callback is not None:
        _streaming_reporter = StreamingProgressReporter(callback=callback)
    return _streaming_reporter


def reset_streaming_reporter():
    """Reset the streaming reporter (useful for testing)."""
    global _streaming_reporter
    _streaming_reporter = None
