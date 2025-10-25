"""
Report Generator Agent - Synthesizes answers from tool outputs.

Takes RAG chunks, execution results, and other tool outputs to generate
a coherent natural language answer to the user's question.
"""

import logging
import re
from typing import Dict, Any, Optional
from openai import OpenAI
import config
from core.streaming_progress import StreamingProgressReporter, ProgressFormatter
from core.data_models import ToolBasedPlan, RAGContext

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Agent responsible for synthesizing answers from tool outputs.

    Takes all tool outputs (RAG chunks, execution results, etc.) and generates
    a coherent answer to the user's question. Also evaluates if the answer is
    complete or if more tools need to be called.
    """

    def __init__(self, streaming_reporter: Optional[StreamingProgressReporter] = None):
        """
        Initialize report generator.

        Args:
            streaming_reporter: Optional reporter for real-time streaming
        """
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.streaming_reporter = streaming_reporter
        self.formatter = ProgressFormatter()
        self.rag_context = None  # Will be set during synthesis if available

    def synthesize_answer(
        self,
        user_question: str,
        tool_outputs: Dict[str, Dict[str, Any]],
        plan: Optional[ToolBasedPlan] = None
    ) -> Dict[str, Any]:
        """
        Synthesize a natural language answer from tool outputs.

        Args:
            user_question: Original user question
            tool_outputs: Dictionary of tool outputs (keyed by tool_id)
            plan: Optional tool plan to evaluate completeness

        Returns:
            Dictionary with:
                - answer: Synthesized natural language answer
                - is_complete: Whether the answer is complete
                - needs_more_tools: Whether additional tool calls are needed
                - suggested_tools: Optional list of suggested next tools
        """
        logger.info(f"[REPORT_GENERATOR] Synthesizing answer for: {user_question[:100]}...")

        # Extract RAG context if available (for making citations clickable later)
        rag_context = None
        for tool_id, output in tool_outputs.items():
            if isinstance(output, dict) and "rag_context" in output:
                rag_context = output["rag_context"]
                break

        # Build context from tool outputs
        context = self._build_context(tool_outputs)

        # Build system prompt
        system_prompt = """You are an expert scientific report generator for calcium imaging analysis.

Your task is to synthesize information from various tool outputs (RAG retrieval, code execution, etc.) and generate a clear, comprehensive answer to the user's question.

Guidelines:
- Be direct and informative
- When citing information from papers, use SHORT numbered citations like [1], [2], [3], etc.
- DO NOT use full paper filenames in your answer (e.g., avoid "[Spatial and temporal aspects.pdf]")
- The papers are already listed with full names ABOVE your answer, so readers can reference them by number
- Example: "According to [1], calcium oscillations..." or "As described in [2] and [3]..."
- IMPORTANT: Only cite each source ONCE per statement. Do NOT repeat citations like [1, 2][1, 2] - just write [1, 2] once
- DO NOT add a "References" section at the end - the complete numbered reference list is already shown ABOVE
- Your citations [1], [2], etc. should match the numbered list that was already displayed
- Explain technical concepts clearly
- If information is missing, acknowledge the limitations
- Structure your answer logically with appropriate sections or bullet points when helpful
- Focus on accuracy and clarity over formality
"""

        # Build user prompt
        user_prompt = f"""User Question: {user_question}

Available Information:
{context}

Please provide a comprehensive answer based on the available information."""

        # Stream synthesis start with a clear header
        if self.streaming_reporter:
            self.streaming_reporter.emit_event(
                "synthesis_start",
                "\n---\n\n## Answer\n\n"
            )

        # Call LLM using gpt-5 Responses API
        try:
            # Combine system and user prompts for Responses API
            combined_input = f"{system_prompt}\n\n{user_prompt}"

            # Log input size
            logger.info(f"[REPORT_GENERATOR] Input length: {len(combined_input)} chars (~{len(combined_input)//4} tokens)")

            # Use responses.create() for gpt-5
            logger.debug(f"[REPORT_GENERATOR] Calling gpt-5 with model={config.OPENAI_MODEL}")
            response = self.client.responses.create(
                model=config.OPENAI_MODEL,
                input=combined_input
                # No max_output_tokens - let the model finish naturally
            )

            logger.debug(f"[REPORT_GENERATOR] Response object: {response}")
            logger.debug(f"[REPORT_GENERATOR] Response type: {type(response)}")
            logger.debug(f"[REPORT_GENERATOR] Response dict: {response.model_dump() if hasattr(response, 'model_dump') else 'N/A'}")

            # Extract text from the response object
            # The response has an 'output' array with reasoning and message items
            raw_response = None
            if hasattr(response, 'output') and response.output:
                logger.debug(f"[REPORT_GENERATOR] Found {len(response.output)} items in output array")
                for idx, item in enumerate(response.output):
                    logger.debug(f"[REPORT_GENERATOR] Item {idx}: type={item.type if hasattr(item, 'type') else 'N/A'}")
                    if hasattr(item, 'type') and item.type == 'message':
                        if hasattr(item, 'content') and item.content:
                            logger.debug(f"[REPORT_GENERATOR] Message has {len(item.content)} content items")
                            for cidx, content_item in enumerate(item.content):
                                if hasattr(content_item, 'text'):
                                    logger.debug(f"[REPORT_GENERATOR] Content item {cidx} text length: {len(content_item.text)}")
                                    logger.debug(f"[REPORT_GENERATOR] Content item {cidx} text preview: {content_item.text[:200]}")
                                    raw_response = content_item.text
                                    break
                        break

            # Fallback to output_text if it exists
            if not raw_response and hasattr(response, 'output_text'):
                raw_response = response.output_text

            logger.debug(f"[REPORT_GENERATOR] raw_response type: {type(raw_response)}")

            # Log what we received
            logger.info(f"[REPORT_GENERATOR] Raw LLM response length: {len(raw_response) if raw_response else 0}")
            if raw_response:
                logger.debug(f"[REPORT_GENERATOR] First 500 chars: {raw_response[:500]}")

            if not raw_response or not raw_response.strip():
                raise ValueError("LLM returned empty response")

            # Make citations clickable and remove duplicates
            processed_response = self._make_citations_clickable(raw_response, rag_context)

            # Stream the complete response at once if streaming is enabled
            if self.streaming_reporter:
                self.streaming_reporter.emit_event("synthesis", processed_response)

            # Stream end marker (minimal - answer is already displayed)
            if self.streaming_reporter:
                self.streaming_reporter.emit_event(
                    "synthesis_end",
                    "\n\n"
                )

            logger.info(f"[REPORT_GENERATOR] ✓ Generated answer ({len(processed_response)} chars)")

            # Return the processed text response with clickable citations
            return {
                "answer": processed_response,
                "is_complete": True,
                "needs_more_tools": False,
                "suggested_tools": None,
                "reasoning": None
            }

        except Exception as e:
            logger.error(f"[REPORT_GENERATOR] ✗ Synthesis failed: {e}")

            # Fallback: return raw context
            return {
                "answer": f"Error during synthesis: {e}\n\nRetrieved information:\n{context}",
                "is_complete": False,
                "needs_more_tools": False,
                "suggested_tools": None,
                "reasoning": f"Synthesis error: {e}"
            }

    def _make_citations_clickable(self, text: str, rag_context: Optional[RAGContext]) -> str:
        """
        Convert inline citation numbers like [1], [2], [3] to clickable markdown links.
        Also removes duplicate consecutive citations like [1, 2][1, 2].

        Args:
            text: Answer text with citations
            rag_context: RAG context with paper information

        Returns:
            Text with clickable citations and duplicates removed
        """
        if not rag_context:
            return text

        # First, remove duplicate consecutive citations like [1, 2, 3][1, 2, 3][1, 2, 3]
        # Pattern: [digits/spaces/commas] followed by identical pattern
        def remove_duplicate_citations(match):
            citation = match.group(0)
            # Find repeating pattern
            pattern = r'\[([0-9,\s]+)\]'
            matches = re.findall(pattern, citation)
            if matches:
                # Keep only first occurrence
                return f'[{matches[0]}]'
            return citation

        # Remove patterns like [1, 2][1, 2][1, 2] -> [1, 2]
        text = re.sub(r'(\[[0-9,\s]+\])(\1)+', remove_duplicate_citations, text)

        # Get unique sources for URL mapping
        unique_sources = rag_context.get_unique_sources()

        # Import citation formatter to get URLs
        from core.citation_formatter import CitationFormatter

        # Build mapping of citation number -> URL
        citation_urls = {}
        for i, source_info in enumerate(unique_sources, 1):
            if source_info.get('full_path'):
                url = CitationFormatter._get_paper_url(source_info['full_path'])
                citation_urls[i] = url

        # Convert individual citation numbers to clickable links
        # Pattern: [single digit] or [digit, digit, digit]
        def make_citation_link(match):
            citation_text = match.group(1)  # e.g., "1" or "1, 2, 3"

            # Parse citation numbers
            numbers = [int(n.strip()) for n in citation_text.split(',') if n.strip().isdigit()]

            if not numbers:
                return match.group(0)  # Return unchanged if no numbers

            # If single citation, make it a direct link
            if len(numbers) == 1:
                num = numbers[0]
                if num in citation_urls:
                    url = citation_urls[num]
                    return f'[[{num}]]({url})'
                else:
                    return f'[{num}]'

            # If multiple citations, make each clickable
            links = []
            for num in numbers:
                if num in citation_urls:
                    url = citation_urls[num]
                    links.append(f'[[{num}]]({url})')
                else:
                    links.append(f'[{num}]')

            return '[' + ', '.join(links) + ']'

        # Replace [1], [2, 3], etc. with clickable versions
        text = re.sub(r'\[([0-9,\s]+)\]', make_citation_link, text)

        return text

    def _build_context(self, tool_outputs: Dict[str, Dict[str, Any]]) -> str:
        """
        Build context string from tool outputs.

        Args:
            tool_outputs: Dictionary of tool outputs

        Returns:
            Formatted context string
        """
        context_parts = []

        for tool_id, output in tool_outputs.items():
            if not isinstance(output, dict):
                continue

            # Handle RAG output
            if "chunks" in output and "sources" in output:
                chunks = output["chunks"]
                sources = output["sources"]

                context_parts.append(f"\n=== Retrieved Information from Papers ===\n")
                for chunk, source in zip(chunks, sources):
                    context_parts.append(f"[Source: {source}]\n{chunk}\n")

            # Handle execution results
            elif "results" in output:
                results = output["results"]
                context_parts.append(f"\n=== Analysis Results ({tool_id}) ===\n")
                context_parts.append(f"{results}\n")

            # Handle code output
            elif "code" in output:
                code = output["code"]
                context_parts.append(f"\n=== Generated Code ({tool_id}) ===\n")
                context_parts.append(f"```python\n{code}\n```\n")

            # Handle verification results
            elif "verification" in output:
                verification = output["verification"]
                context_parts.append(f"\n=== Verification ({tool_id}) ===\n")
                context_parts.append(f"{verification}\n")

            # Handle other outputs
            else:
                context_parts.append(f"\n=== Tool Output ({tool_id}) ===\n")
                context_parts.append(f"{output}\n")

        return "\n".join(context_parts) if context_parts else "No information available."


# Singleton instance
_report_generator = None


def get_report_generator(streaming_reporter: Optional[StreamingProgressReporter] = None) -> ReportGenerator:
    """
    Get or create ReportGenerator instance.

    Args:
        streaming_reporter: Optional streaming reporter for real-time updates

    Returns:
        ReportGenerator instance
    """
    # Don't use singleton if streaming_reporter is provided
    if streaming_reporter is not None:
        return ReportGenerator(streaming_reporter=streaming_reporter)

    global _report_generator
    if _report_generator is None:
        _report_generator = ReportGenerator()
    return _report_generator
