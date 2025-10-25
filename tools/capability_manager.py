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

EXECUTION ENVIRONMENT:
Your code will run in an isolated Docker container with the following specifications:
- OS: Debian-based Linux
- Python: 3.11
- User: Non-root (calcium user)
- Network: DISABLED (no internet access)
- Working directory: /workspace
- Available paths:
  * /data/images (read-only) - contains image files
  * /workspace/outputs (read-write) - for saving figures
- Resource limits: 2GB RAM, 1 CPU core (configurable)

DATA LOADING:
You have access to a `load_images()` function that loads calcium imaging data:

```python
images = load_images(
    max_frames=10,        # Optional: limit number of frames
    frame_indices=[0,5]   # Optional: specific frame indices
)
# Returns: numpy.ndarray, shape (T, H, W), dtype float32, range [0, 1]
```

You decide what data to load based on the user's request:
- "analyze first image" → load_images(max_frames=1)
- "analyze first 5 frames" → load_images(max_frames=5)
- "analyze frames 10-20" → load_images(frame_indices=list(range(10, 21)))
- "analyze all images" → load_images() (no parameters)

OUTPUT REQUIREMENTS:
Your code must create two variables:
1. `results`: dict containing numerical outputs
   - Example: {'n_cells': 42, 'mean_intensity': 0.65}
   - ALL values must be JSON-serializable (int, float, str, list, dict)
   - Convert numpy types: int(value), float(value), value.tolist()
2. `figure`: matplotlib figure object or None
   - Use plt.figure() to create visualizations
   - The system will automatically save the figure to /workspace/outputs

ALLOWED IMPORTS AND VERSIONS:
- numpy>=1.24.0 (as np)
- scipy>=1.10.0 (scipy.signal, scipy.ndimage)
- matplotlib>=3.7.0 (matplotlib.pyplot as plt)
- scikit-image>=0.25.0 (skimage.measure, skimage.segmentation, skimage.filters, skimage.feature)

IMPORTANT API NOTES FOR SCIKIT-IMAGE 0.25+:
- peak_local_max() no longer has 'indices' parameter - returns boolean mask by default
- Use peak_local_max(image, min_distance=X) instead of peak_local_max(image, indices=False, min_distance=X)
- For coordinates, use: np.column_stack(np.where(peaks)) where peaks = peak_local_max(...)
- watershed() requires integer labels, convert float images: labels.astype(np.int64)
- watershed(image, markers) - markers must be integer type (np.int32 or np.int64)

COMMON PITFALLS TO AVOID:
1. Type errors: Always convert numpy integers to Python int for results dict
   ❌ results = {'count': np.int64(42)}
   ✓ results = {'count': int(42)}
2. Watershed dtype: Ensure markers are integers
   ❌ markers = np.zeros(image.shape)
   ✓ markers = np.zeros(image.shape, dtype=np.int32)
3. Division by zero: Check for zero before dividing
   ✓ if denominator != 0: ratio = numerator / denominator else: ratio = 0.0
4. Empty arrays: Check array size before operations
   ✓ if len(peaks) > 0: mean_peak = peaks.mean() else: mean_peak = 0.0

CODE STRUCTURE:
1. Call load_images() with appropriate parameters
2. Process the data
3. Create results dict and figure

CODE STYLE:
- Include docstring explaining biological context
- Add comments for non-obvious steps
- Use descriptive variable names
- Keep functions under 50 lines each
- Total code under 150 lines

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

    def regenerate_with_error_feedback(
        self,
        user_request: str,
        rag_context,
        data_info: Dict,
        previous_code: str,
        error_message: str,
        stdout: str = "",
        stderr: str = ""
    ) -> GeneratedCapability:
        """
        Regenerate code with error feedback from previous attempt.

        Args:
            user_request: User's natural language request
            rag_context: Retrieved context from papers
            data_info: Information about the data (shape, etc.)
            previous_code: Code that failed
            error_message: Error message from execution
            stdout: Standard output from execution
            stderr: Standard error from execution

        Returns:
            GeneratedCapability with fixed code
        """
        logger.info(f"Regenerating code with error feedback for: '{user_request}'")
        logger.warning(f"Previous error: {error_message[:200]}")

        # Build error feedback prompt
        error_feedback_prompt = self._build_error_feedback_prompt(
            user_request=user_request,
            rag_context=rag_context,
            data_info=data_info,
            previous_code=previous_code,
            error_message=error_message,
            stdout=stdout,
            stderr=stderr
        )

        # Call OpenAI API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": error_feedback_prompt}
                ],
                temperature=self.temperature
            )

            generated_text = response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"Code regeneration failed: {e}")

        # Extract code from markdown fence
        code = self._extract_code(generated_text)

        # Validate code
        self._validate_code(code)

        # Extract imports
        imports_used = self._extract_imports(code)

        # Count lines for estimation
        n_lines = len(code.split('\n'))

        logger.info(f"Regenerated {n_lines} lines of code with imports: {imports_used}")

        return GeneratedCapability(
            code=code,
            description=user_request,
            imports_used=imports_used,
            estimated_runtime="fast" if n_lines < 50 else "medium"
        )

    def _build_error_feedback_prompt(
        self,
        user_request: str,
        rag_context,
        data_info: Dict,
        previous_code: str,
        error_message: str,
        stdout: str,
        stderr: str
    ) -> str:
        """Build prompt with error feedback for code regeneration."""

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

PREVIOUS CODE (FAILED):
```python
{previous_code}
```

ERROR FROM EXECUTION:
{error_message}

STDOUT:
{stdout if stdout else "(empty)"}

STDERR:
{stderr if stderr else "(empty)"}

TASK:
The previous code failed with the error shown above. Please fix the error and generate corrected code.

Common fixes needed:
1. If TypeError about numpy types: Convert numpy integers/floats to Python types using int() or float()
2. If TypeError about watershed markers: Ensure markers are integer dtype (np.int32 or np.int64)
3. If API errors with peak_local_max: Remove 'indices' parameter (not supported in scikit-image 0.25+)
4. If division by zero: Add checks before division operations
5. If IndexError or empty array errors: Check array size before indexing

Generate the CORRECTED Python code that fixes these errors.
Ensure the code creates both `results` dict and `figure` object as specified in the system prompt."""

        return prompt
