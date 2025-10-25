"""
LangGraph state definition for the calcium imaging pipeline.
"""

from typing import TypedDict, Optional, List, Dict, Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from core.streaming_progress import StreamingProgressReporter

from core.data_models import (
    AnalysisResult,
    ToolBasedPlan,
    UserApprovalResponse,
    ErrorFeedback
)


class PipelineState(TypedDict):
    """State object that flows through the LangGraph workflow (Tool-based)."""
    # User inputs
    user_request: str
    images_path: str

    # Tool-based planning
    tool_plan: Optional[ToolBasedPlan]

    # User approval
    user_approval: Optional[UserApprovalResponse]
    waiting_for_approval: bool

    # Tool execution outputs (keyed by tool_id)
    tool_outputs: Dict[str, Dict[str, Any]]

    # Report synthesis
    report_result: Optional[Dict[str, Any]]

    # Error handling
    current_error: Optional[ErrorFeedback]
    waiting_for_error_response: bool
    errors: List[str]

    # Final output
    final_output: Optional[AnalysisResult]

    # Progress tracking
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]]
    streaming_reporter: Optional[Any]  # StreamingProgressReporter (avoid circular import)

    # Interruption flag
    interrupted: bool

    # Session management
    session_id: Optional[str]
    output_path: Optional[str]
