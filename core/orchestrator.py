"""
Orchestrator: Executes tool calls from the plan.
"""

import logging
from typing import Dict, Any, List
import time
from core.data_models import (
    ToolBasedPlan, ToolCall, ToolType, StepStatus,
    RAGContext, ExecutionResult, VerificationResult,
    ErrorFeedback
)
from tools.rag_system_enhanced import EnhancedRAGSystem
from tools.capability_manager import CapabilityManager
from core.docker_executor import execute_code_docker
from core.verifier import get_verifier
from core.code_generator import get_code_generator, CodeGenerator
from core.streaming_progress import StreamingProgressReporter, ProgressFormatter

logger = logging.getLogger(__name__)


class Orchestrator:
    """Orchestrates execution of tool-based plans."""

    def __init__(self, streaming_reporter: StreamingProgressReporter = None):
        """Initialize orchestrator with tools."""
        self.rag_system = None
        self.capability_manager = None
        self.code_generator = None
        self.verifier = get_verifier()
        self.streaming_reporter = streaming_reporter
        self.formatter = ProgressFormatter()

    def _get_rag_system(self) -> EnhancedRAGSystem:
        """Lazy load enhanced RAG system."""
        if self.rag_system is None:
            self.rag_system = EnhancedRAGSystem()
        return self.rag_system

    def _get_capability_manager(self) -> CapabilityManager:
        """Lazy load capability manager."""
        if self.capability_manager is None:
            self.capability_manager = CapabilityManager()
        return self.capability_manager

    def _get_code_generator(self) -> CodeGenerator:
        """Lazy load code generator with streaming support."""
        if self.code_generator is None:
            self.code_generator = get_code_generator(streaming_reporter=self.streaming_reporter)
        return self.code_generator

    def _resolve_input_references(self, inputs: Dict[str, Any], tool_outputs: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Resolve input references like "$t1.output.chunks" to actual values.

        Args:
            inputs: Tool inputs with possible references
            tool_outputs: Dictionary of completed tool outputs {tool_id: output}

        Returns:
            Resolved inputs with actual values
        """
        resolved = {}
        for key, value in inputs.items():
            if isinstance(value, str) and value.startswith("$"):
                # Parse reference like "$t1.output.chunks"
                parts = value[1:].split(".")
                if len(parts) >= 2:
                    tool_id = parts[0]
                    if tool_id in tool_outputs:
                        ref_value = tool_outputs[tool_id]
                        # Navigate through nested fields
                        for part in parts[1:]:
                            if isinstance(ref_value, dict):
                                ref_value = ref_value.get(part, value)
                            else:
                                break
                        resolved[key] = ref_value
                    else:
                        logger.warning(f"Reference {value} not found in tool outputs")
                        resolved[key] = None
                else:
                    resolved[key] = value
            else:
                resolved[key] = value
        return resolved

    def execute_tool(
        self,
        tool_call: ToolCall,
        tool_outputs: Dict[str, Dict],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a single tool call.

        Args:
            tool_call: The tool to execute
            tool_outputs: Outputs from previously executed tools
            context: Additional context (images_path, etc.)

        Returns:
            Tool output dictionary

        Raises:
            Exception: If tool execution fails
        """
        logger.info(f"[ORCHESTRATOR] Executing {tool_call.tool_id}: [{tool_call.tool_type.value}] {tool_call.description}")

        # Stream tool start event
        if self.streaming_reporter:
            start_msg = self.formatter.format_tool_start(
                tool_call.tool_id,
                tool_call.tool_type.value,
                tool_call.description
            )
            self.streaming_reporter.emit_event("tool_start", start_msg)

        # Resolve input references
        inputs = self._resolve_input_references(tool_call.inputs, tool_outputs)
        # Add context to inputs
        inputs.update({f"$user.{k}": v for k, v in context.items()})

        start_time = time.time()

        try:
            if tool_call.tool_type == ToolType.RAG:
                output = self._execute_rag(inputs)

            elif tool_call.tool_type == ToolType.CODE_GEN:
                output = self._execute_code_generation(inputs)

            elif tool_call.tool_type == ToolType.EXECUTE:
                output = self._execute_code(inputs, context)

            elif tool_call.tool_type == ToolType.VERIFY:
                output = self._execute_verify(inputs)

            elif tool_call.tool_type == ToolType.CAPABILITY_SEARCH:
                output = self._execute_capability_search(inputs)

            elif tool_call.tool_type == ToolType.CAPABILITY_SAVE:
                output = self._execute_capability_save(inputs, context)

            else:
                raise ValueError(f"Unknown tool type: {tool_call.tool_type}")

            execution_time = time.time() - start_time
            logger.info(f"[ORCHESTRATOR] ✓ {tool_call.tool_id} completed in {execution_time:.2f}s")

            # Stream completion event
            if self.streaming_reporter:
                complete_msg = self.formatter.format_tool_complete(tool_call.tool_id, success=True)
                self.streaming_reporter.emit_event("tool_complete", complete_msg)

            return output

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"[ORCHESTRATOR] ✗ {tool_call.tool_id} failed after {execution_time:.2f}s: {e}")

            # Stream failure event
            if self.streaming_reporter:
                fail_msg = self.formatter.format_tool_complete(tool_call.tool_id, success=False)
                error_msg = self.formatter.format_error(str(e))
                self.streaming_reporter.emit_event("tool_failed", fail_msg + error_msg)

            raise

    def _execute_rag(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute RAG tool with enhanced features.

        Supported inputs:
        - query: Search query (required)
        - top_k: Number of sections to retrieve (optional, defaults to config)
        - return_full_papers: Return full paper if multiple sections match (default: True)
        - multi_section_threshold: Min sections to trigger full paper (default: 2)
        """
        query = inputs.get("query", "")
        top_k = inputs.get("top_k", None)  # Let orchestrator/planner decide
        return_full_papers = inputs.get("return_full_papers", True)
        multi_section_threshold = inputs.get("multi_section_threshold", 2)

        rag_system = self._get_rag_system()
        rag_context = rag_system.retrieve(
            query=query,
            top_k=top_k,
            return_full_papers=return_full_papers,
            multi_section_threshold=multi_section_threshold
        )

        # Stream RAG results with full reference list
        if self.streaming_reporter:
            from core.citation_formatter import CitationFormatter

            # Format complete citation list to show immediately
            citation_list = CitationFormatter.format_citation_list(rag_context, format="markdown")

            # Emit both the summary and the full list
            unique_sources = rag_context.get_unique_sources()
            rag_msg = f"📚 **Retrieved from {len(unique_sources)} papers**\n\n{citation_list}\n"

            self.streaming_reporter.emit_event("rag_complete", rag_msg)

        return {
            "chunks": rag_context.chunks,
            "sources": rag_context.sources,
            "scores": rag_context.scores,
            "metadata": rag_context.metadata,
            "rag_context": rag_context  # Store full RAG context for citation generation
        }

    def _execute_code_generation(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute code generation tool by calling CodeGenerator agent.

        The orchestrator doesn't generate code itself - it delegates to the CodeGenerator agent.
        """
        task_description = inputs.get("task_description", "")
        rag_context = inputs.get("rag_context")
        previous_error = inputs.get("previous_error")  # For retry with feedback

        # Get code generator agent (with streaming support)
        code_gen = self._get_code_generator()

        # Call the agent to generate code
        code = code_gen.generate_code(
            task_description=task_description,
            rag_context=rag_context,
            previous_error=previous_error
        )

        return {
            "code": code,
            "description": task_description
        }

    def _execute_code(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code in Docker sandbox."""
        code = inputs.get("code", "")
        images_path = context.get("images_path", "")
        output_path = context.get("output_path", "")
        session_id = context.get("session_id")

        # Stream execution start
        if self.streaming_reporter:
            exec_msg = self.formatter.format_execution_start()
            self.streaming_reporter.emit_event("execution_start", exec_msg)

        exec_result = execute_code_docker(
            code=code,
            images_path=images_path,
            output_path=output_path,
            session_id=session_id
        )

        # Stream execution output/results
        if self.streaming_reporter and exec_result.success:
            if exec_result.results:
                result_str = "\n".join([f"  • {k}: {v}" for k, v in exec_result.results.items()])
                self.streaming_reporter.emit_event("execution_results",
                                                   f"**Results:**\n{result_str}\n\n")

        return {
            "success": exec_result.success,
            "results": exec_result.results,
            "figures": [exec_result.figure] if exec_result.figure else [],
            "execution_time": exec_result.execution_time,
            "error_message": exec_result.error_message
        }

    def _execute_verify(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute verification tool."""
        execution_result = inputs.get("execution_result", {})
        expected_output = inputs.get("expected_output", "")
        task_description = inputs.get("task_description", "Analysis task")

        # Create ExecutionResult from dict
        from core.data_models import ExecutionResult as ExecResult

        exec_result = ExecResult(
            success=execution_result.get("success", False),
            results=execution_result.get("results", {}),
            execution_time=execution_result.get("execution_time", 0.0),
            error_message=execution_result.get("error_message", "")
        )

        # Call verifier with new signature
        verification = self.verifier.verify(
            task_description=task_description,
            result=exec_result,
            expected_output=expected_output
        )

        return {
            "passed": verification.passed,
            "confidence": verification.confidence,
            "issues": verification.issues,
            "suggestions": verification.suggestions,
            "should_retry": verification.should_retry
        }

    def _execute_capability_search(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute capability search tool."""
        query = inputs.get("query", "")
        cap_manager = self._get_capability_manager()

        result = cap_manager.search_capability(query)

        return {
            "found": result is not None,
            "capability_id": result.get("capability_id") if result else None,
            "code": result.get("code") if result else None,
            "similarity": result.get("similarity", 0.0) if result else 0.0
        }

    def _execute_capability_save(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute capability save tool."""
        code = inputs.get("code", "")
        description = inputs.get("description", "")
        cap_manager = self._get_capability_manager()

        capability_id = cap_manager.save_capability(
            code=code,
            description=description,
            request=context.get("original_request", "")
        )

        return {
            "capability_id": capability_id,
            "saved": True
        }


# Don't use singleton - create new instance each time to support streaming reporter
def get_orchestrator(streaming_reporter: StreamingProgressReporter = None) -> Orchestrator:
    """
    Create Orchestrator instance with optional streaming reporter.

    Args:
        streaming_reporter: Optional StreamingProgressReporter for real-time updates

    Returns:
        Orchestrator instance
    """
    return Orchestrator(streaming_reporter=streaming_reporter)
