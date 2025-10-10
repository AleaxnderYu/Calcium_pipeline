"""
Data models for inter-component communication in the calcium imaging pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime


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
    sources: List[str]
    scores: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate RAG context."""
        if len(self.chunks) != len(self.sources):
            raise ValueError("Number of chunks must match number of sources")
        if self.scores and len(self.scores) != len(self.chunks):
            raise ValueError("Number of scores must match number of chunks")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'chunks': self.chunks,
            'sources': self.sources,
            'scores': self.scores,
            'metadata': self.metadata
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
