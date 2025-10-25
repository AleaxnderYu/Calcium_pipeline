"""
Data models for inter-component communication in the calcium imaging pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Literal
import numpy as np
from datetime import datetime
from enum import Enum


@dataclass
class PreprocessedData:
    """Data structure for preprocessed calcium imaging data."""
    images: np.ndarray  # Shape: (T, H, W), dtype: float32
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate preprocessed data."""
        if self.images.ndim != 3:
            raise ValueError(f"Images must be 3D (T×H×W), got shape {self.images.shape}")
        if self.images.dtype != np.float32:
            raise ValueError(f"Images must be float32, got {self.images.dtype}")

        # Populate basic metadata if not provided
        if not self.metadata:
            self.metadata = {
                'n_frames': self.images.shape[0],
                'height': self.images.shape[1],
                'width': self.images.shape[2],
                'normalized': True,
                'pixel_range': (float(self.images.min()), float(self.images.max()))
            }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding numpy array)."""
        return {
            'shape': self.images.shape,
            'dtype': str(self.images.dtype),
            'metadata': self.metadata
        }


@dataclass
class RAGContext:
    """Context retrieved from RAG system."""
    chunks: List[str]
    sources: List[str]  # Filenames only (e.g., "paper.pdf")
    scores: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    pages: List[int] = field(default_factory=list)  # Page numbers for each chunk
    full_paths: List[str] = field(default_factory=list)  # Full file paths for hyperlinks

    def __post_init__(self):
        """Validate RAG context."""
        if len(self.chunks) != len(self.sources):
            raise ValueError("Number of chunks must match number of sources")
        if self.scores and len(self.scores) != len(self.chunks):
            raise ValueError("Number of scores must match number of chunks")
        if self.pages and len(self.pages) != len(self.chunks):
            raise ValueError("Number of pages must match number of chunks")
        if self.full_paths and len(self.full_paths) != len(self.chunks):
            raise ValueError("Number of full_paths must match number of chunks")

    def get_unique_sources(self) -> List[Dict[str, Any]]:
        """
        Get list of unique source papers with their details.

        Strips section names from filenames (e.g., "paper.pdf - Section" → "paper.pdf")
        and deduplicates based on filename only.

        Returns:
            List of dicts with 'filename', 'full_path', 'pages' for each unique source
        """
        unique_sources = {}
        for i, source in enumerate(self.sources):
            # Strip section name if present (format: "filename.pdf - Section Name")
            base_filename = source.split(" - ")[0] if " - " in source else source

            if base_filename not in unique_sources:
                unique_sources[base_filename] = {
                    'filename': base_filename,
                    'full_path': self.full_paths[i] if i < len(self.full_paths) else None,
                    'pages': []
                }
            if i < len(self.pages):
                page = self.pages[i]
                if page not in unique_sources[base_filename]['pages']:
                    unique_sources[base_filename]['pages'].append(page)

        # Sort pages for each source
        for source_info in unique_sources.values():
            source_info['pages'].sort()

        return list(unique_sources.values())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'chunks': self.chunks,
            'sources': self.sources,
            'scores': self.scores,
            'metadata': self.metadata,
            'pages': self.pages,
            'full_paths': self.full_paths,
            'unique_sources': self.get_unique_sources()
        }


@dataclass
class GeneratedCapability:
    """Generated Python code capability."""
    code: str
    description: str
    imports_used: List[str] = field(default_factory=list)
    estimated_runtime: str = "unknown"

    def __post_init__(self):
        """Validate generated capability."""
        if not self.code.strip():
            raise ValueError("Generated code cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'code': self.code,
            'description': self.description,
            'imports_used': self.imports_used,
            'estimated_runtime': self.estimated_runtime
        }


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    results: Dict[str, Any] = field(default_factory=dict)
    figure: Any = None
    execution_time: float = 0.0
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding figure object)."""
        return {
            'success': self.success,
            'results': self.results,
            'has_figure': self.figure is not None,
            'execution_time': self.execution_time,
            'error_message': self.error_message
        }


@dataclass
class AnalysisResult:
    """Final analysis result package."""
    data: Dict[str, Any]
    figures: List[str] = field(default_factory=list)
    summary: str = ""
    code_used: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Add metadata if not provided."""
        if not self.metadata.get('timestamp'):
            self.metadata['timestamp'] = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'data': self.data,
            'figures': self.figures,
            'summary': self.summary,
            'code_used': self.code_used,
            'metadata': self.metadata
        }


@dataclass
class VerificationResult:
    """Result from verification process."""
    passed: bool
    confidence: float  # 0.0 to 1.0
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    should_retry: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'passed': self.passed,
            'confidence': self.confidence,
            'issues': self.issues,
            'suggestions': self.suggestions,
            'should_retry': self.should_retry
        }


# Tool-based orchestration data models

class StepStatus(Enum):
    """Status of a tool call or task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    VERIFIED = "verified"
    RETRY = "retry"

class ToolType(Enum):
    """Available tool types for orchestrator."""
    RAG = "rag"  # Retrieve scientific methods
    CODE_GEN = "code_generation"  # Generate Python code
    EXECUTE = "execute"  # Execute code in sandbox
    VERIFY = "verify"  # Verify execution results
    CAPABILITY_SEARCH = "capability_search"  # Search for reusable capabilities
    CAPABILITY_SAVE = "capability_save"  # Save new capability
    REPORT = "report"  # Synthesize answer from tool outputs


@dataclass
class ToolCall:
    """A tool call in the execution plan."""
    tool_id: str  # Unique ID for this tool call
    tool_type: ToolType
    description: str  # What this tool call does
    inputs: Dict[str, Any] = field(default_factory=dict)  # Tool inputs
    depends_on: List[str] = field(default_factory=list)  # Tool call IDs this depends on1
    status: StepStatus = StepStatus.PENDING
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'tool_id': self.tool_id,
            'tool_type': self.tool_type.value,
            'description': self.description,
            'inputs': self.inputs,
            'depends_on': self.depends_on,
            'status': self.status.value,
            'has_output': self.output is not None,
            'error': self.error,
            'retry_count': self.retry_count
        }


@dataclass
class ToolBasedPlan:
    """Execution plan using tool calls instead of action types."""
    plan_id: str
    original_request: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    execution_mode: Literal["sequential", "parallel", "dag"] = "sequential"
    assumptions: List[str] = field(default_factory=list)
    user_approved: bool = False
    is_complete: bool = False

    def get_ready_tools(self) -> List[ToolCall]:
        """Get tool calls that are ready to execute (dependencies met)."""
        ready = []
        for tool_call in self.tool_calls:
            if tool_call.status == StepStatus.PENDING:
                # Check if all dependencies are completed
                deps_met = all(
                    any(tc.tool_id == dep_id and tc.status in [StepStatus.COMPLETED, StepStatus.VERIFIED]
                        for tc in self.tool_calls)
                    for dep_id in tool_call.depends_on
                ) if tool_call.depends_on else True

                if deps_met:
                    ready.append(tool_call)
        return ready

    def mark_tool_complete(self, tool_id: str, output: Dict[str, Any]):
        """Mark a tool call as complete."""
        for tool_call in self.tool_calls:
            if tool_call.tool_id == tool_id:
                tool_call.status = StepStatus.COMPLETED
                tool_call.output = output
                break

        # Check if all tools are complete
        if all(tc.status in [StepStatus.COMPLETED, StepStatus.VERIFIED, StepStatus.SKIPPED]
               for tc in self.tool_calls):
            self.is_complete = True

    def mark_tool_failed(self, tool_id: str, error: str):
        """Mark a tool call as failed."""
        for tool_call in self.tool_calls:
            if tool_call.tool_id == tool_id:
                tool_call.status = StepStatus.FAILED
                tool_call.error = error
                break

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'plan_id': self.plan_id,
            'original_request': self.original_request,
            'tool_calls': [tc.to_dict() for tc in self.tool_calls],
            'execution_mode': self.execution_mode,
            'assumptions': self.assumptions,
            'user_approved': self.user_approved,
            'is_complete': self.is_complete,
            'total_tools': len(self.tool_calls)
        }


@dataclass
class UserApprovalRequest:
    """Request for user approval of execution plan."""
    plan: ToolBasedPlan
    estimated_time: str = "unknown"
    estimated_cost: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            'plan': self.plan.to_dict(),
            'estimated_time': self.estimated_time,
            'estimated_cost': self.estimated_cost
        }


@dataclass
class UserApprovalResponse:
    """User's response to approval request."""
    approved: bool
    modified_plan: Optional[ToolBasedPlan] = None
    user_feedback: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'approved': self.approved,
            'has_modifications': self.modified_plan is not None,
            'user_feedback': self.user_feedback
        }


@dataclass
class ErrorFeedback:
    """Error feedback to present to user."""
    error_message: str
    failed_tool: ToolCall
    context: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    retry_available: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'error_message': self.error_message,
            'failed_tool': self.failed_tool.to_dict(),
            'context': self.context,
            'suggestions': self.suggestions,
            'retry_available': self.retry_available
        }


@dataclass
class ReportResult:
    """Result from report synthesis."""
    answer: str
    is_complete: bool
    needs_more_tools: bool
    suggested_tools: Optional[List[str]] = None
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'answer': self.answer,
            'is_complete': self.is_complete,
            'needs_more_tools': self.needs_more_tools,
            'suggested_tools': self.suggested_tools,
            'reasoning': self.reasoning
        }
