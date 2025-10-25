"""
LangGraph workflow definition for the calcium imaging pipeline.

Tool-based workflow with user approval and error feedback:
User Query → Planner → User Approval → Orchestrator → (Error? → User) → Results
"""

import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from datetime import datetime
import tempfile

from graph.state import PipelineState
from core.data_models import AnalysisResult, ToolBasedPlan, UserApprovalResponse, ErrorFeedback, StepStatus, ReportResult
from core.tool_planner import get_tool_planner
from core.orchestrator import get_orchestrator
from core.report_generator import get_report_generator
from core.streaming_progress import ProgressFormatter
import config

logger = logging.getLogger(__name__)


def planner_node(state: PipelineState) -> PipelineState:
    """
    Node 1: Create tool-based execution plan.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with tool_plan
    """
    logger.info(f"[NODE: planner] Planning for: '{state['user_request']}'")

    try:
        planner = get_tool_planner()
        tool_plan = planner.create_plan(state["user_request"])
        state["tool_plan"] = tool_plan
        state["waiting_for_approval"] = True

        # Stream plan creation event
        streaming_reporter = state.get("streaming_reporter")
        if streaming_reporter:
            formatter = ProgressFormatter()
            plan_msg = formatter.format_plan_header(len(tool_plan.tool_calls), tool_plan.execution_mode)
            streaming_reporter.emit_event("plan_created", plan_msg)

            # Stream each tool in the plan
            for i, tc in enumerate(tool_plan.tool_calls, 1):
                tool_desc = f"{i}. **[{tc.tool_type.value}]** {tc.description}\n"
                streaming_reporter.emit_event("plan_tool", tool_desc)

            streaming_reporter.emit_event("plan_complete", "\n")

        logger.info(f"[NODE: planner] ✓ Created plan with {len(tool_plan.tool_calls)} tools")

    except Exception as e:
        error_msg = f"Planning failed: {str(e)}"
        logger.error(f"[NODE: planner] {error_msg}")
        state["errors"].append(error_msg)
        state["waiting_for_approval"] = False

    return state


def user_approval_node(state: PipelineState) -> PipelineState:
    """
    Node 2: Wait for user approval of plan.

    This is a HUMAN-IN-THE-LOOP node. In practice, this would:
    1. Present plan to user via UI
    2. Wait for user response (approve/reject/modify)
    3. Continue when approved

    For now, auto-approve for testing.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with user_approval
    """
    logger.info("[NODE: user_approval] Requesting user approval")

    # TODO: Implement actual user approval UI
    # For now, auto-approve
    if state.get("tool_plan"):
        plan = state["tool_plan"]

        # Present plan to user (log for now)
        logger.info(f"[NODE: user_approval] Plan to approve:")
        logger.info(f"  Request: {plan.original_request}")
        logger.info(f"  Assumptions: {plan.assumptions}")
        logger.info(f"  Tools: {len(plan.tool_calls)} tool calls")
        for tc in plan.tool_calls:
            logger.info(f"    - [{tc.tool_type.value}] {tc.description}")

        # Auto-approve for now
        state["user_approval"] = UserApprovalResponse(
            approved=True,
            modified_plan=None,
            user_feedback="Auto-approved for testing"
        )
        state["waiting_for_approval"] = False
        plan.user_approved = True

        logger.info("[NODE: user_approval] ✓ Plan approved")

    return state


def orchestrator_node(state: PipelineState) -> PipelineState:
    """
    Node 3: Execute tool calls from the plan.

    Handles:
    - Sequential execution
    - Parallel execution
    - DAG execution with dependencies
    - Error handling and retry

    Args:
        state: Current pipeline state

    Returns:
        Updated state with tool_outputs or current_error
    """
    logger.info("[NODE: orchestrator] Starting tool execution")

    plan = state.get("tool_plan")
    if not plan or not plan.user_approved:
        logger.error("[NODE: orchestrator] No approved plan to execute")
        return state

    # Get or create orchestrator with streaming reporter
    streaming_reporter = state.get("streaming_reporter")
    orchestrator = get_orchestrator(streaming_reporter=streaming_reporter)
    tool_outputs = state.get("tool_outputs", {})

    # Context for tool execution
    context = {
        "images_path": state["images_path"],
        "output_path": state["output_path"],
        "session_id": state.get("session_id"),
        "original_request": state["user_request"]
    }

    try:
        # Execute tools based on mode
        if plan.execution_mode == "sequential":
            # Sequential execution
            for tool_call in plan.tool_calls:
                if tool_call.status == StepStatus.PENDING:
                    tool_call.status = StepStatus.RUNNING

                    try:
                        output = orchestrator.execute_tool(tool_call, tool_outputs, context)
                        tool_outputs[tool_call.tool_id] = output
                        plan.mark_tool_complete(tool_call.tool_id, output)

                    except Exception as e:
                        logger.error(f"[NODE: orchestrator] Tool {tool_call.tool_id} failed: {e}")
                        plan.mark_tool_failed(tool_call.tool_id, str(e))

                        # Create error feedback for user
                        state["current_error"] = ErrorFeedback(
                            error_message=str(e),
                            failed_tool=tool_call,
                            context=context,
                            suggestions=[
                                "Retry the failed tool",
                                "Modify the plan",
                                "Abort execution"
                            ],
                            retry_available=tool_call.retry_count < 3
                        )
                        state["waiting_for_error_response"] = True
                        state["tool_outputs"] = tool_outputs
                        return state

        elif plan.execution_mode in ["parallel", "dag"]:
            # DAG execution with dependency tracking
            while not plan.is_complete:
                ready_tools = plan.get_ready_tools()

                if not ready_tools:
                    # Check if there are pending tools that can't proceed
                    pending = [tc for tc in plan.tool_calls if tc.status == StepStatus.PENDING]
                    if pending:
                        logger.error(f"[NODE: orchestrator] Deadlock: {len(pending)} tools pending but none ready")
                        break
                    else:
                        break

                # Execute ready tools (can be parallel in future)
                for tool_call in ready_tools:
                    tool_call.status = StepStatus.RUNNING

                    try:
                        output = orchestrator.execute_tool(tool_call, tool_outputs, context)
                        tool_outputs[tool_call.tool_id] = output
                        plan.mark_tool_complete(tool_call.tool_id, output)

                    except Exception as e:
                        logger.error(f"[NODE: orchestrator] Tool {tool_call.tool_id} failed: {e}")
                        plan.mark_tool_failed(tool_call.tool_id, str(e))

                        # Create error feedback
                        state["current_error"] = ErrorFeedback(
                            error_message=str(e),
                            failed_tool=tool_call,
                            context=context,
                            suggestions=["Retry the failed tool", "Skip and continue", "Abort"],
                            retry_available=tool_call.retry_count < 3
                        )
                        state["waiting_for_error_response"] = True
                        state["tool_outputs"] = tool_outputs
                        return state

        state["tool_outputs"] = tool_outputs
        logger.info(f"[NODE: orchestrator] ✓ Completed {len(tool_outputs)} tool executions")

    except Exception as e:
        error_msg = f"Orchestrator failed: {str(e)}"
        logger.error(f"[NODE: orchestrator] {error_msg}")
        state["errors"].append(error_msg)

    return state


def report_synthesis_node(state: PipelineState) -> PipelineState:
    """
    Node: Synthesize answer from tool outputs.

    Takes all tool outputs (RAG chunks, execution results, etc.) and uses
    an LLM to generate a coherent natural language answer. Also evaluates
    if the answer is complete or if more tools need to be called.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with report_result
    """
    logger.info("[NODE: report_synthesis] Synthesizing answer from tool outputs")

    try:
        tool_outputs = state.get("tool_outputs", {})
        user_question = state["user_request"]
        plan = state.get("tool_plan")

        # Get streaming reporter
        streaming_reporter = state.get("streaming_reporter")

        # Get report generator
        report_gen = get_report_generator(streaming_reporter=streaming_reporter)

        # Synthesize answer
        report_result = report_gen.synthesize_answer(
            user_question=user_question,
            tool_outputs=tool_outputs,
            plan=plan
        )

        state["report_result"] = report_result
        logger.info(f"[NODE: report_synthesis] ✓ Synthesis complete (complete: {report_result.get('is_complete', False)})")

        # Citations are already shown when RAG retrieves, so we don't stream them again here

        # Check if more tools are needed
        if report_result.get("needs_more_tools", False):
            logger.info(f"[NODE: report_synthesis] Additional tools needed: {report_result.get('suggested_tools', [])}")
            # TODO: Create new plan with suggested tools and re-execute
            # For now, just continue to output

    except Exception as e:
        error_msg = f"Report synthesis failed: {str(e)}"
        logger.error(f"[NODE: report_synthesis] {error_msg}")
        state["errors"].append(error_msg)

        # Fallback: provide raw output
        state["report_result"] = {
            "answer": f"Error synthesizing answer: {e}",
            "is_complete": False,
            "needs_more_tools": False,
            "reasoning": f"Synthesis error: {e}"
        }

    return state


def error_handler_node(state: PipelineState) -> PipelineState:
    """
    Node: Handle errors and wait for user decision.

    Args:
        state: Current pipeline state

    Returns:
        Updated state after error handling
    """
    logger.info("[NODE: error_handler] Handling error")

    error = state.get("current_error")
    if error:
        logger.error(f"[NODE: error_handler] Error in tool {error.failed_tool.tool_id}: {error.error_message}")
        logger.info(f"[NODE: error_handler] Suggestions: {error.suggestions}")

        # TODO: Implement actual user error response UI
        # For now, auto-retry once
        if error.retry_available and error.failed_tool.retry_count == 0:
            logger.info("[NODE: error_handler] Auto-retrying failed tool")
            error.failed_tool.retry_count += 1
            error.failed_tool.status = StepStatus.RETRY
            state["current_error"] = None
            state["waiting_for_error_response"] = False
        else:
            logger.info("[NODE: error_handler] Max retries reached, aborting")
            state["waiting_for_error_response"] = False

    return state


def format_output_node(state: PipelineState) -> PipelineState:
    """
    Node 4: Format final output from report synthesis.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with final_output
    """
    logger.info("[NODE: format_output] Formatting results")

    try:
        from core.citation_formatter import CitationFormatter

        tool_outputs = state.get("tool_outputs", {})
        report_result = state.get("report_result", {})
        plan = state.get("tool_plan")

        # Collect all results, figures, code, and RAG contexts from tool outputs
        all_results = {}
        all_figures = []
        all_code = []
        rag_contexts = []  # Collect all RAG contexts for citations

        for tool_id, output in tool_outputs.items():
            if isinstance(output, dict):
                # Handle execution results
                if "results" in output:
                    all_results.update(output["results"])
                if "figures" in output:
                    all_figures.extend(output["figures"])
                if "code" in output:
                    all_code.append(output["code"])
                # Collect RAG contexts for citation generation
                if "rag_context" in output and output["rag_context"]:
                    logger.info(f"[NODE: format_output] Found RAG context in {tool_id}")
                    rag_contexts.append(output["rag_context"])

        logger.info(f"[NODE: format_output] Collected {len(rag_contexts)} RAG context(s) from {len(tool_outputs)} tool outputs")

        # Use synthesized answer from report if available
        if report_result and "answer" in report_result:
            # Don't add "# Answer" header - the answer is already streamed during synthesis
            # Just use the answer directly (it will be shown in streaming, this is for non-streaming fallback)
            summary = f"{report_result['answer']}\n\n"

            # Add raw results if available (from code execution, etc.)
            if all_results:
                summary += f"## Analysis Results\n\n{all_results}\n\n"
        else:
            # Fallback to old behavior if no report synthesis
            if plan:
                summary = f"Executed {len(tool_outputs)} tools for: {plan.original_request}\n\n"
                summary += f"Assumptions: {', '.join(plan.assumptions)}\n\n"
                if all_results:
                    summary += f"## Analysis Results:\n{all_results}"
            else:
                summary = f"Results: {all_results}"

        # Add citations from all RAG contexts
        if rag_contexts:
            logger.info(f"[NODE: format_output] Adding citations from {len(rag_contexts)} RAG context(s)")

            # Merge all RAG contexts into one comprehensive list
            all_chunks = []
            all_sources = []
            all_scores = []
            all_pages = []
            all_full_paths = []

            for rag_ctx in rag_contexts:
                all_chunks.extend(rag_ctx.chunks)
                all_sources.extend(rag_ctx.sources)
                all_scores.extend(rag_ctx.scores)

                # Handle pages (default to 0 if not available)
                if hasattr(rag_ctx, 'pages') and rag_ctx.pages:
                    all_pages.extend(rag_ctx.pages)
                else:
                    all_pages.extend([0] * len(rag_ctx.chunks))

                # Handle full_paths (default to empty string if not available)
                if hasattr(rag_ctx, 'full_paths') and rag_ctx.full_paths:
                    all_full_paths.extend(rag_ctx.full_paths)
                else:
                    # Try to construct from sources
                    for source in rag_ctx.sources:
                        full_path = str(config.PAPERS_DIR / source) if source else ""
                        all_full_paths.append(full_path)

            from core.data_models import RAGContext
            merged_rag_context = RAGContext(
                chunks=all_chunks,
                sources=all_sources,
                scores=all_scores,
                pages=all_pages,
                full_paths=all_full_paths
            )

            # Format citations for summary only (already streamed in report_synthesis_node)
            citation_list = CitationFormatter.format_citation_list(merged_rag_context, format="markdown")
            if citation_list:
                summary += f"\n\n{citation_list}\n"

                # Log paper summary (don't stream again - already done in report_synthesis_node)
                rag_summary = CitationFormatter.format_rag_summary(merged_rag_context)
                logger.info(f"[NODE: format_output] {rag_summary}")

        final_output = AnalysisResult(
            data=all_results,
            figures=all_figures,
            summary=summary,
            code_used="\n\n".join(all_code),
            metadata={
                "request": state["user_request"],
                "timestamp": datetime.now().isoformat(),
                "status": "success" if not state["errors"] else "partial",
                "tool_count": len(tool_outputs),
                "errors": state["errors"],
                "report_complete": report_result.get("is_complete", False) if report_result else False,
                "needs_more_tools": report_result.get("needs_more_tools", False) if report_result else False,
                "papers_cited": len(rag_contexts) if rag_contexts else 0
            }
        )

        state["final_output"] = final_output
        logger.info("[NODE: format_output] ✓ Output formatted with citations")

    except Exception as e:
        error_msg = f"Output formatting failed: {str(e)}"
        logger.error(f"[NODE: format_output] {error_msg}")
        state["errors"].append(error_msg)

    return state


def should_handle_error(state: PipelineState) -> str:
    """Conditional edge: check if there's an error to handle."""
    if state.get("waiting_for_error_response"):
        return "error_handler"
    else:
        return "report_synthesis"


def should_continue_after_error(state: PipelineState) -> str:
    """Conditional edge: continue or end after error handling."""
    if state.get("current_error") and not state.get("waiting_for_error_response"):
        # Error resolved, continue orchestration
        return "orchestrator"
    else:
        # No more retries, go to report synthesis
        return "report_synthesis"


def create_workflow():
    """
    Create and compile the LangGraph workflow (V2 - Tool-based with report synthesis).

    New structure:
    planner → user_approval → orchestrator → [error_handler] → report_synthesis → format_output → END

    Returns:
        Compiled workflow app
    """
    workflow = StateGraph(PipelineState)

    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("user_approval", user_approval_node)
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("error_handler", error_handler_node)
    workflow.add_node("report_synthesis", report_synthesis_node)
    workflow.add_node("format_output", format_output_node)

    # Set entry point
    workflow.set_entry_point("planner")

    # Linear flow with conditional error handling
    workflow.add_edge("planner", "user_approval")
    workflow.add_edge("user_approval", "orchestrator")

    # Conditional: handle error or continue to report synthesis
    workflow.add_conditional_edges(
        "orchestrator",
        should_handle_error,
        {
            "error_handler": "error_handler",
            "report_synthesis": "report_synthesis"
        }
    )

    # After error handling: retry or continue to report synthesis
    workflow.add_conditional_edges(
        "error_handler",
        should_continue_after_error,
        {
            "orchestrator": "orchestrator",
            "report_synthesis": "report_synthesis"
        }
    )

    # Report synthesis always goes to format output
    workflow.add_edge("report_synthesis", "format_output")

    # End
    workflow.add_edge("format_output", END)

    # Compile workflow
    app = workflow.compile()

    logger.info("Workflow V2 compiled successfully (tool-based with report synthesis)")
    return app


def run_workflow(user_request: str, images_path: str, session_id: str = None) -> AnalysisResult:
    """
    Run the complete calcium imaging analysis workflow (V2).

    Args:
        user_request: User's natural language request
        images_path: Path to directory containing image frames
        session_id: Optional session ID for sandbox reuse

    Returns:
        AnalysisResult with complete analysis output

    Raises:
        ValueError: If workflow fails to produce valid output
    """
    logger.info(f"Starting workflow V2 for request: '{user_request}'")
    logger.info(f"Images path: {images_path}")

    # Create output directory
    output_path = tempfile.mkdtemp(prefix="calcium_output_")
    logger.info(f"Output directory: {output_path}")

    # Create initial state
    initial_state: PipelineState = {
        "user_request": user_request,
        "images_path": images_path,
        "tool_plan": None,
        "user_approval": None,
        "waiting_for_approval": False,
        "tool_outputs": {},
        "report_result": None,
        "current_error": None,
        "waiting_for_error_response": False,
        "errors": [],
        "final_output": None,
        "progress_callback": None,
        "streaming_reporter": None,
        "interrupted": False,
        "session_id": session_id,
        "output_path": output_path
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

    logger.info("Workflow V2 completed successfully")
    return final_output
