"""
LangGraph workflow definition for the calcium imaging pipeline.
"""

import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END

from graph.state import PipelineState
from graph.nodes import (
    preprocess_node,
    rag_retrieval_node,
    capability_search_node,
    code_generation_node,
    save_capability_node,
    execution_node,
    format_output_node
)
from core.data_models import AnalysisResult

logger = logging.getLogger(__name__)


def create_workflow():
    """
    Create and compile the LangGraph workflow.

    Returns:
        Compiled workflow app
    """
    # Create state graph
    workflow = StateGraph(PipelineState)

    # Add nodes
    workflow.add_node("preprocess", preprocess_node)
    workflow.add_node("rag_retrieval", rag_retrieval_node)
    workflow.add_node("capability_search", capability_search_node)
    workflow.add_node("code_generation", code_generation_node)
    workflow.add_node("execution", execution_node)
    workflow.add_node("save_capability", save_capability_node)
    workflow.add_node("format_output", format_output_node)

    # Set entry point
    workflow.set_entry_point("preprocess")

    # Add edges (linear flow with capability store integration)
    workflow.add_edge("preprocess", "rag_retrieval")
    workflow.add_edge("rag_retrieval", "capability_search")
    workflow.add_edge("capability_search", "code_generation")
    workflow.add_edge("code_generation", "execution")
    workflow.add_edge("execution", "save_capability")
    workflow.add_edge("save_capability", "format_output")
    workflow.add_edge("format_output", END)

    # Compile workflow
    app = workflow.compile()

    logger.info("Workflow compiled successfully")
    return app


def run_workflow(user_request: str, images_path: str) -> AnalysisResult:
    """
    Run the complete calcium imaging analysis workflow.

    Args:
        user_request: User's natural language analysis request
        images_path: Path to directory containing image frames

    Returns:
        AnalysisResult with complete analysis output

    Raises:
        ValueError: If workflow fails to produce valid output
    """
    logger.info(f"Starting workflow for request: '{user_request}'")
    logger.info(f"Images path: {images_path}")

    # Create initial state
    initial_state: PipelineState = {
        "user_request": user_request,
        "images_path": images_path,
        "preprocessed_data": None,
        "rag_context": None,
        "generated_code": None,
        "execution_results": None,
        "final_output": None,
        "errors": [],
        # Capability store fields
        "capability_reused": False,
        "capability_id": None,
        "capability_similarity": None,
        "generated_capability": None
    }

    # Create and run workflow
    app = create_workflow()
    final_state = app.invoke(initial_state)

    # Extract final output
    final_output = final_state.get("final_output")

    if final_output is None:
        error_msg = "Workflow failed to produce output"
        if final_state["errors"]:
            error_msg += f": {'; '.join(final_state['errors'])}"
        raise ValueError(error_msg)

    logger.info("Workflow completed successfully")
    return final_output
