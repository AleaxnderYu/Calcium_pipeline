"""
LangGraph state definition for the calcium imaging pipeline.
"""

from typing import TypedDict, Optional, List, Dict, Any
from core.data_models import (
    PreprocessedData,
    RAGContext,
    ExecutionResult,
    AnalysisResult
)


class PipelineState(TypedDict):
    """State object that flows through the LangGraph workflow."""
    user_request: str
    images_path: str
    preprocessed_data: Optional[PreprocessedData]
    rag_context: Optional[RAGContext]
    generated_code: Optional[str]
    execution_results: Optional[ExecutionResult]
    final_output: Optional[AnalysisResult]
    errors: List[str]

    # Capability store fields
    capability_reused: bool
    capability_id: Optional[str]
    capability_similarity: Optional[float]
    generated_capability: Optional[Any]  # GeneratedCapability object
