"""
Session Manager: Lightweight session tracking for Docker-based execution.

Note: Docker containers are created and destroyed per execution, so no sandbox
persistence is needed. This module just tracks dialogue sessions for logging purposes.
"""

import logging
import time
from typing import Optional, Dict
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class Session:
    """Represents a dialogue session (for tracking purposes only)."""

    def __init__(self, session_id: str):
        """
        Initialize a session.

        Args:
            session_id: Unique session identifier
        """
        self.session_id = session_id
        self.created_at = time.time()
        self.query_count = 0

    def increment_query_count(self):
        """Increment the query counter."""
        self.query_count += 1


class SessionManager:
    """Manages dialogue sessions (lightweight tracking only)."""

    def __init__(self):
        self.sessions: Dict[str, Session] = {}
        logger.info("SessionManager initialized (Docker mode - no sandbox persistence)")

    def create_session(self, session_id: Optional[str] = None, timeout: int = 3600) -> str:
        """
        Create a new session.

        Args:
            session_id: Optional session ID (auto-generated if not provided)
            timeout: Ignored in Docker mode (kept for compatibility)

        Returns:
            Session ID
        """
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        if session_id in self.sessions:
            logger.debug(f"Session {session_id} already exists, reusing")
        else:
            self.sessions[session_id] = Session(session_id)
            logger.info(f"Created session: {session_id}")

        return session_id

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get an existing session."""
        return self.sessions.get(session_id)

    def increment_query_count(self, session_id: str):
        """Increment query count for a session."""
        session = self.sessions.get(session_id)
        if session:
            session.increment_query_count()

    def close_session(self, session_id: str):
        """Close and cleanup a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Session {session_id} closed")

    def close_all_sessions(self):
        """Close all active sessions."""
        logger.info(f"Closing all sessions ({len(self.sessions)} active)")
        self.sessions.clear()

    def list_sessions(self) -> list:
        """List all active sessions."""
        return [
            {
                "session_id": session.session_id,
                "created_at": datetime.fromtimestamp(session.created_at).isoformat(),
                "query_count": session.query_count,
                "age_seconds": time.time() - session.created_at
            }
            for session in self.sessions.values()
        ]

    def __del__(self):
        """Cleanup all sessions on deletion."""
        self.close_all_sessions()


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get or create the global session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
