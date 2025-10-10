"""
LangGraph node functions for the calcium imaging pipeline workflow.
"""

import logging
from pathlib import Path
from datetime import datetime

from graph.state import PipelineState
from layers.preprocessor import Preprocessor
from layers.rag_system import RAGSystem
from layers.capability_manager import CapabilityManager
from layers.capability_store import CapabilityStore
from core.executor import execute_code
from core.data_models import AnalysisResult
import config

logger = logging.getLogger(__name__)

# Initialize components (singleton pattern)
_preprocessor = None
_rag_system = None
_capability_manager = None
_capability_store = None


def get_preprocessor() -> Preprocessor:
    """Get or create Preprocessor instance."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = Preprocessor()
    return _preprocessor


def get_rag_system() -> RAGSystem:
    """Get or create RAGSystem instance."""
    global _rag_system
    if _rag_system is None:
        _rag_system = RAGSystem()
    return _rag_system


def get_capability_manager() -> CapabilityManager:
    """Get or create CapabilityManager instance."""
    global _capability_manager
    if _capability_manager is None:
        _capability_manager = CapabilityManager()
    return _capability_manager


def get_capability_store() -> CapabilityStore:
    """Get or create CapabilityStore instance."""
    global _capability_store
    if _capability_store is None:
        _capability_store = CapabilityStore()
    return _capability_store


def preprocess_node(state: PipelineState) -> PipelineState:
    """
    Node 1: Preprocess images (L3).

    Args:
        state: Current pipeline state

    Returns:
        Updated state with preprocessed_data
    """
    logger.info("[NODE: preprocess] Starting image preprocessing")

    try:
        preprocessor = get_preprocessor()
        preprocessed_data = preprocessor.process(state["images_path"])
        state["preprocessed_data"] = preprocessed_data
        logger.info("[NODE: preprocess] ✓ Preprocessing completed")

    except Exception as e:
        error_msg = f"Preprocessing failed: {str(e)}"
        logger.error(f"[NODE: preprocess] {error_msg}")
        state["errors"].append(error_msg)

    return state


def rag_retrieval_node(state: PipelineState) -> PipelineState:
    """
    Node 2: Retrieve relevant methods from papers (L2).

    Args:
        state: Current pipeline state

    Returns:
        Updated state with rag_context
    """
    logger.info("[NODE: rag_retrieval] Starting RAG retrieval")

    try:
        rag_system = get_rag_system()
        rag_context = rag_system.retrieve(state["user_request"])
        state["rag_context"] = rag_context
        logger.info(f"[NODE: rag_retrieval] ✓ Retrieved {len(rag_context.chunks)} chunks")

    except Exception as e:
        error_msg = f"RAG retrieval failed: {str(e)}"
        logger.error(f"[NODE: rag_retrieval] {error_msg}")
        state["errors"].append(error_msg)

    return state


def capability_search_node(state: PipelineState) -> PipelineState:
    """
    Node 3: Search for existing similar capability (NEW).

    Args:
        state: Current pipeline state

    Returns:
        Updated state with capability info if found
    """
    logger.info("[NODE: capability_search] Searching for similar capabilities")

    # Skip if capability reuse is disabled
    if not config.ENABLE_CAPABILITY_REUSE:
        logger.info("[NODE: capability_search] ⊘ Capability reuse disabled")
        state["capability_reused"] = False
        return state

    try:
        store = get_capability_store()

        # Search for similar capabilities
        similar = store.search_similar(
            state["user_request"],
            threshold=config.CAPABILITY_SIMILARITY_THRESHOLD,
            top_k=3
        )

        if similar:
            # Use the best match
            best_match = similar[0]
            cap_id = best_match["cap_id"]
            similarity = best_match["similarity"]

            logger.info(f"[NODE: capability_search] ✓ Found capability {cap_id} (similarity: {similarity:.2f})")

            # Load the capability
            capability = store.load_capability(cap_id)

            # Update state
            state["generated_code"] = capability.code
            state["capability_reused"] = True
            state["capability_id"] = cap_id
            state["capability_similarity"] = similarity

            # Increment reuse count
            store.increment_reuse(cap_id)

            logger.info(f"[NODE: capability_search] ✓ Reusing capability {cap_id}")
        else:
            logger.info("[NODE: capability_search] No similar capability found")
            state["capability_reused"] = False

    except Exception as e:
        error_msg = f"Capability search failed: {str(e)}"
        logger.error(f"[NODE: capability_search] {error_msg}")
        state["errors"].append(error_msg)
        state["capability_reused"] = False

    return state


def code_generation_node(state: PipelineState) -> PipelineState:
    """
    Node 4: Generate Python code (L4).

    Args:
        state: Current pipeline state

    Returns:
        Updated state with generated_code
    """
    logger.info("[NODE: code_generation] Starting code generation")

    # Skip if capability was reused
    if state.get("capability_reused", False):
        logger.info("[NODE: code_generation] ⊘ Skipped - using reused capability")
        return state

    try:
        # Extract data info from preprocessed data
        if state["preprocessed_data"] is None:
            raise ValueError("Preprocessed data is required for code generation")

        if state["rag_context"] is None:
            raise ValueError("RAG context is required for code generation")

        data_info = state["preprocessed_data"].metadata

        capability_manager = get_capability_manager()
        capability = capability_manager.generate(
            user_request=state["user_request"],
            rag_context=state["rag_context"],
            data_info=data_info
        )

        state["generated_code"] = capability.code
        state["generated_capability"] = capability
        logger.info(f"[NODE: code_generation] ✓ Generated {len(capability.code.split())} lines of code")

    except Exception as e:
        error_msg = f"Code generation failed: {str(e)}"
        logger.error(f"[NODE: code_generation] {error_msg}")
        state["errors"].append(error_msg)

    return state


def execution_node(state: PipelineState) -> PipelineState:
    """
    Node 5: Execute generated code.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with execution_results
    """
    logger.info("[NODE: execution] Starting code execution")

    try:
        if state["generated_code"] is None:
            raise ValueError("Generated code is required for execution")

        if state["preprocessed_data"] is None:
            raise ValueError("Preprocessed data is required for execution")

        execution_result = execute_code(
            code=state["generated_code"],
            images=state["preprocessed_data"].images
        )

        state["execution_results"] = execution_result

        if execution_result.success:
            logger.info(f"[NODE: execution] ✓ Execution succeeded in {execution_result.execution_time:.2f}s")
        else:
            logger.error(f"[NODE: execution] Execution failed: {execution_result.error_message}")
            state["errors"].append(f"Execution error: {execution_result.error_message}")

    except Exception as e:
        error_msg = f"Execution node failed: {str(e)}"
        logger.error(f"[NODE: execution] {error_msg}")
        state["errors"].append(error_msg)

    return state


def save_capability_node(state: PipelineState) -> PipelineState:
    """
    Node 6: Save capability if newly generated and successful (NEW).

    Args:
        state: Current pipeline state

    Returns:
        Updated state with capability_id
    """
    logger.info("[NODE: save_capability] Checking if should save capability")

    # Skip if capability was reused
    if state.get("capability_reused", False):
        logger.info("[NODE: save_capability] ⊘ Skipped - capability was reused")
        return state

    # Skip if execution failed
    if not state.get("execution_results") or not state["execution_results"].success:
        logger.info("[NODE: save_capability] ⊘ Skipped - execution failed")
        return state

    # Skip if capability reuse is disabled
    if not config.ENABLE_CAPABILITY_REUSE:
        logger.info("[NODE: save_capability] ⊘ Capability reuse disabled")
        return state

    try:
        store = get_capability_store()

        cap_id = store.save_capability(
            request=state["user_request"],
            capability=state["generated_capability"],
            execution_result=state["execution_results"]
        )

        state["capability_id"] = cap_id
        logger.info(f"[NODE: save_capability] ✓ Saved new capability {cap_id}")

    except Exception as e:
        error_msg = f"Save capability failed: {str(e)}"
        logger.error(f"[NODE: save_capability] {error_msg}")
        state["errors"].append(error_msg)

    return state


def format_output_node(state: PipelineState) -> PipelineState:
    """
    Node 7: Format final output and save results.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with final_output
    """
    logger.info("[NODE: format_output] Formatting final output")

    try:
        # Check if there were errors
        if state["errors"]:
            error_summary = "; ".join(state["errors"])
            final_output = AnalysisResult(
                data={"error": error_summary},
                figures=[],
                summary=f"Analysis failed: {error_summary}",
                code_used=state.get("generated_code", ""),
                metadata={
                    "request": state["user_request"],
                    "timestamp": datetime.now().isoformat(),
                    "status": "failed",
                    "capability_reused": state.get("capability_reused", False),
                    "capability_id": state.get("capability_id")
                }
            )
            state["final_output"] = final_output
            logger.warning("[NODE: format_output] Pipeline completed with errors")
            return state

        # Extract results
        execution_results = state["execution_results"]
        if execution_results is None or not execution_results.success:
            raise ValueError("No successful execution results available")

        # Save figure if present
        figures = []
        if execution_results.figure is not None:
            # Create timestamped output directory
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = config.OUTPUT_DIR / timestamp
            output_dir.mkdir(parents=True, exist_ok=True)

            figure_path = output_dir / "figure_001.png"
            execution_results.figure.savefig(figure_path, dpi=150, bbox_inches='tight')
            figures.append(str(figure_path))
            logger.info(f"Saved figure to {figure_path}")

        # Generate summary
        summary = _generate_summary(state["user_request"], execution_results.results, state)

        # Create final output
        final_output = AnalysisResult(
            data=execution_results.results,
            figures=figures,
            summary=summary,
            code_used=state["generated_code"],
            metadata={
                "request": state["user_request"],
                "timestamp": datetime.now().isoformat(),
                "execution_time": execution_results.execution_time,
                "status": "success",
                "capability_reused": state.get("capability_reused", False),
                "capability_id": state.get("capability_id"),
                "capability_similarity": state.get("capability_similarity")
            }
        )

        state["final_output"] = final_output
        logger.info("[NODE: format_output] ✓ Output formatting completed")

    except Exception as e:
        error_msg = f"Output formatting failed: {str(e)}"
        logger.error(f"[NODE: format_output] {error_msg}")
        state["errors"].append(error_msg)

        # Create error output
        final_output = AnalysisResult(
            data={"error": error_msg},
            figures=[],
            summary=f"Failed to format output: {error_msg}",
            code_used=state.get("generated_code", ""),
            metadata={
                "request": state["user_request"],
                "timestamp": datetime.now().isoformat(),
                "status": "failed"
            }
        )
        state["final_output"] = final_output

    return state


def _generate_summary(request: str, results: dict, state: PipelineState) -> str:
    """Generate natural language summary of results."""

    summary_parts = [f"Analysis request: {request}", ""]

    # Add capability reuse info
    if state.get("capability_reused", False):
        similarity = state.get("capability_similarity", 0)
        cap_id = state.get("capability_id", "unknown")
        summary_parts.append(f"💡 Reused existing capability {cap_id} (similarity: {similarity:.2f})")
        summary_parts.append("")

    summary_parts.append("Results:")
    for key, value in results.items():
        if isinstance(value, (int, float)):
            summary_parts.append(f"  - {key}: {value}")
        elif isinstance(value, list):
            summary_parts.append(f"  - {key}: {len(value)} items")
        else:
            summary_parts.append(f"  - {key}: {type(value).__name__}")

    return "\n".join(summary_parts)
