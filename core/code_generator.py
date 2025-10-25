"""
Code Generator Agent - Generates Python code for calcium imaging analysis.

Separate from orchestrator - the orchestrator calls this agent as a tool.
"""

import logging
from typing import Dict, Any, Optional
from openai import OpenAI
import config
from core.streaming_progress import StreamingProgressReporter, ProgressFormatter

logger = logging.getLogger(__name__)


class CodeGenerator:
    """
    Agent responsible for generating Python code for analysis tasks.

    This is a separate agent that the orchestrator calls, not part of orchestrator logic.
    """

    def __init__(self, streaming_reporter: Optional[StreamingProgressReporter] = None):
        """
        Initialize code generator.

        Args:
            streaming_reporter: Optional reporter for real-time streaming
        """
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.streaming_reporter = streaming_reporter
        self.formatter = ProgressFormatter()

    def generate_code(
        self,
        task_description: str,
        rag_context: Optional[Dict[str, Any]] = None,
        previous_error: Optional[str] = None
    ) -> str:
        """
        Generate Python code for the given task.

        Args:
            task_description: Description of what the code should do
            rag_context: Optional RAG context with scientific methods
            previous_error: Optional previous error for retry with feedback

        Returns:
            Generated Python code as string
        """
        logger.info(f"[CODE_GENERATOR] Generating code for: {task_description[:100]}...")

        # Build system prompt
        system_prompt = """You are an expert Python code generator for calcium imaging analysis.

Generate production-ready Python code that:
- Uses numpy, scipy, scikit-image, matplotlib
- Has proper error handling
- Returns results as a dictionary
- Saves figures to output_path if visualization is needed
- Is well-commented and clear

The code will run in a sandboxed environment with these functions available:
- load_images(images_path) -> np.ndarray  # Returns (T, H, W) array

Always include this at the start:
```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load images
images = load_images(images_path)
output_path = Path(output_path)

# Your analysis code here
...

# Return results
results = {
    "key": value,
    ...
}
```

IMPORTANT: Only output the Python code, nothing else. The code will be extracted and executed directly.
"""

        # Build user prompt
        user_prompt = f"Task: {task_description}\n\n"

        # Add RAG context if available
        if rag_context:
            chunks = rag_context.get("chunks", [])
            sources = rag_context.get("sources", [])
            if chunks:
                context_text = "\n\n".join([
                    f"[Source: {src}]\n{chunk}"
                    for chunk, src in zip(chunks, sources)
                ])
                user_prompt += f"Scientific context from research papers:\n{context_text}\n\n"

        # Add error feedback if this is a retry
        if previous_error:
            user_prompt += f"\nPREVIOUS ATTEMPT FAILED WITH ERROR:\n{previous_error}\n\n"
            user_prompt += "Please fix the code to avoid this error.\n\n"

        user_prompt += "Generate the complete Python code:"

        # Stream code generation start
        if self.streaming_reporter:
            self.streaming_reporter.emit_event(
                "code_generation_start",
                self.formatter.format_code_generation_start()
            )

        # Call LLM using gpt-5 Responses API
        try:
            # Combine system and user prompts for Responses API
            combined_input = f"{system_prompt}\n\n{user_prompt}"

            # Use responses.create() for gpt-5
            response = self.client.responses.create(
                model=config.OPENAI_MODEL,
                input=combined_input,
                max_output_tokens=2000
            )

            # Extract text from the response object
            # The response has an 'output' array with reasoning and message items
            code = None
            if hasattr(response, 'output') and response.output:
                for item in response.output:
                    if hasattr(item, 'type') and item.type == 'message':
                        if hasattr(item, 'content') and item.content:
                            for content_item in item.content:
                                if hasattr(content_item, 'text'):
                                    code = content_item.text
                                    break
                        break

            # Fallback to output_text if it exists
            if not code and hasattr(response, 'output_text'):
                code = response.output_text

            # Stream complete code if streaming reporter is available
            if self.streaming_reporter:
                self.streaming_reporter.emit_event("code_generation", code)
                self.streaming_reporter.emit_event(
                    "code_generation_end",
                    self.formatter.format_code_generation_end()
                )

            # Extract code from markdown if present
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                # Handle case where it's just ```
                code = code.split("```")[1].split("```")[0].strip()

            logger.info(f"[CODE_GENERATOR] ✓ Generated {len(code)} characters of code")
            return code

        except Exception as e:
            logger.error(f"[CODE_GENERATOR] ✗ Code generation failed: {e}")
            raise


# Singleton instance
_code_generator = None


def get_code_generator(streaming_reporter: Optional[StreamingProgressReporter] = None) -> CodeGenerator:
    """
    Get or create CodeGenerator instance.

    Args:
        streaming_reporter: Optional streaming reporter for real-time updates

    Returns:
        CodeGenerator instance
    """
    # Don't use singleton if streaming_reporter is provided
    if streaming_reporter is not None:
        return CodeGenerator(streaming_reporter=streaming_reporter)

    global _code_generator
    if _code_generator is None:
        _code_generator = CodeGenerator()
    return _code_generator
