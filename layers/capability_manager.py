"""
Capability Manager: Generate executable Python code using GPT-4/Claude with RAG context.
"""

import logging
import re
from typing import Dict, List
from openai import OpenAI

from core.data_models import GeneratedCapability, RAGContext
import config

logger = logging.getLogger(__name__)


class CapabilityManager:
    """Generates Python code for calcium imaging analysis using LLM."""

    # System prompt template
    SYSTEM_PROMPT = """You are a code generation system for calcium imaging analysis.

TASK: Generate complete, executable Python code based on user requests and scientific literature.

INPUT VARIABLES PROVIDED TO YOUR CODE:
- `images`: numpy.ndarray, shape (T, H, W), dtype float32, range [0, 1]
  - T = number of time frames
  - H, W = image height and width in pixels

OUTPUT REQUIREMENTS:
Your code must create two variables:
1. `results`: dict containing numerical outputs
   - Example: {'n_cells': 42, 'mean_intensity': 0.65}
2. `figure`: matplotlib figure object or None
   - Use plt.figure() to create visualizations

ALLOWED IMPORTS:
- numpy (as np)
- scipy (scipy.signal, scipy.ndimage)
- matplotlib.pyplot (as plt)
- skimage (skimage.measure, skimage.segmentation, skimage.filters, skimage.feature)

CODE STYLE:
- Include docstring explaining biological context
- Add comments for non-obvious steps
- Use descriptive variable names
- Keep functions under 50 lines each
- Total code under 150 lines

SAFETY CONSTRAINTS:
- NO file I/O operations (no open, read, write)
- NO network calls (no requests, urllib)
- NO system commands (no os.system, subprocess)
- NO eval or exec
- NO infinite loops

RETURN FORMAT:
Return ONLY Python code inside a ```python code fence."""

    def __init__(self, model: str = None):
        """
        Initialize capability manager.

        Args:
            model: OpenAI model name (defaults to config.OPENAI_MODEL)
        """
        self.model = model or config.OPENAI_MODEL
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.temperature = 0.2  # Deterministic code generation

    def generate(
        self,
        user_request: str,
        rag_context: RAGContext,
        data_info: Dict
    ) -> GeneratedCapability:
        """
        Generate Python code for the user's analysis request.

        Args:
            user_request: User's natural language request
            rag_context: Retrieved context from papers
            data_info: Information about the data (shape, etc.)

        Returns:
            GeneratedCapability with code and metadata
        """
        logger.info(f"Generating code for: '{user_request}'")
        logger.info(f"Using {len(rag_context.chunks)} RAG chunks from sources: {', '.join(set(rag_context.sources))}")

        # Build user prompt
        user_prompt = self._build_user_prompt(user_request, rag_context, data_info)

        # Call OpenAI API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature
            )

            generated_text = response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"Code generation failed: {e}")

        # Extract code from markdown fence
        code = self._extract_code(generated_text)

        # Validate code
        self._validate_code(code)

        # Extract imports
        imports_used = self._extract_imports(code)

        # Count lines for estimation
        n_lines = len(code.split('\n'))

        logger.info(f"Generated {n_lines} lines of code with imports: {imports_used}")

        return GeneratedCapability(
            code=code,
            description=user_request,
            imports_used=imports_used,
            estimated_runtime="fast" if n_lines < 50 else "medium"
        )

    def _build_user_prompt(
        self,
        user_request: str,
        rag_context: RAGContext,
        data_info: Dict
    ) -> str:
        """Build user prompt with request, RAG context, and data info."""

        # Format RAG chunks
        rag_chunks_text = "\n\n".join([
            f"--- Chunk {i+1} (from {rag_context.sources[i]}) ---\n{chunk}"
            for i, chunk in enumerate(rag_context.chunks)
        ])

        prompt = f"""USER REQUEST:
{user_request}

RELEVANT METHODS FROM SCIENTIFIC LITERATURE:
{rag_chunks_text}

DATA SPECIFICATIONS:
- Shape: ({data_info.get('n_frames', '?')}, {data_info.get('height', '?')}, {data_info.get('width', '?')})
- Type: Calcium imaging time-series
- Pixel range: [0.0, 1.0] (normalized)

TASK:
Generate Python code that accomplishes the user's request using the scientific methods described above.
Ensure the code creates both `results` dict and `figure` object as specified in the system prompt.
If no visualization is needed, set `figure = None`."""

        return prompt

    def _extract_code(self, generated_text: str) -> str:
        """Extract Python code from markdown fence."""

        # Try to extract from ```python fence
        pattern = r"```python\s*(.*?)\s*```"
        match = re.search(pattern, generated_text, re.DOTALL)

        if match:
            return match.group(1).strip()

        # Try generic ``` fence
        pattern = r"```\s*(.*?)\s*```"
        match = re.search(pattern, generated_text, re.DOTALL)

        if match:
            return match.group(1).strip()

        # If no fence, assume entire text is code
        logger.warning("No code fence found, using entire response as code")
        return generated_text.strip()

    def _validate_code(self, code: str):
        """Validate code for forbidden operations."""

        forbidden_patterns = [
            (r'\bopen\s*\(', "file I/O (open)"),
            (r'\bos\.system\b', "system command (os.system)"),
            (r'\bsubprocess\b', "subprocess"),
            (r'\bexec\s*\(', "exec()"),
            (r'\beval\s*\(', "eval()"),
            (r'\b__import__\b', "__import__"),
            (r'\brequests\b', "network requests"),
            (r'\burllib\b', "urllib"),
        ]

        for pattern, description in forbidden_patterns:
            if re.search(pattern, code):
                raise ValueError(f"Generated code contains forbidden operation: {description}")

        logger.debug("Code validation passed")

    def _extract_imports(self, code: str) -> List[str]:
        """Extract import statements from code."""

        imports = []
        for line in code.split('\n'):
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                # Extract module name
                if line.startswith('import '):
                    module = line.split()[1].split('.')[0]
                else:  # from X import Y
                    module = line.split()[1].split('.')[0]
                imports.append(module)

        return list(set(imports))
