"""
Result Verifier: Validates execution results for sanity and correctness.
"""

import logging
from typing import Dict, Any
from openai import OpenAI
import json
import config
from core.data_models import ExecutionResult, VerificationResult

logger = logging.getLogger(__name__)


class ResultVerifier:
    """Verifies execution results make sense."""

    def __init__(self):
        """Initialize the verifier with OpenAI client."""
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.ROUTER_MODEL  # Use lightweight model

    def verify(
        self,
        task_description: str,
        result: ExecutionResult,
        expected_output: str = None,
        data_context: Dict[str, Any] = None
    ) -> VerificationResult:
        """
        Verify that execution result is reasonable.

        Args:
            task_description: Description of what the code was supposed to do
            result: The execution result to verify
            expected_output: Optional description of expected output
            data_context: Context about the data (frame count, dimensions, etc.)

        Returns:
            VerificationResult with pass/fail and suggestions
        """
        logger.info(f"[VERIFIER] Verifying task: {task_description[:100]}...")

        # Quick checks first
        if not result.success:
            logger.warning(f"[VERIFIER] Step failed execution: {result.error_message}")
            return VerificationResult(
                passed=False,
                confidence=0.0,
                issues=[f"Execution failed: {result.error_message}"],
                suggestions=["Check code for syntax errors", "Verify input data format"],
                should_retry=True
            )

        # Use LLM for semantic verification
        system_prompt = """You are a result verification assistant for calcium imaging analysis.

Your job is to verify if execution results make sense given the context.

Respond in JSON format:
{
    "passed": true or false,
    "confidence": 0.0 to 1.0,
    "issues": ["List of problems found"],
    "suggestions": ["How to fix or improve"],
    "should_retry": true or false
}

Check for:
1. Data type correctness (e.g., counts should be integers, not negative)
2. Value reasonableness (e.g., cell count shouldn't be 10000 in small image)
3. Output completeness (e.g., did it produce expected fields?)
4. Physical plausibility (e.g., intensity shouldn't exceed pixel range)

Examples:

Step: "Count cells"
Result: {"cell_count": 42}
Context: Image is 512x512
Verification: {
    "passed": true,
    "confidence": 0.95,
    "issues": [],
    "suggestions": [],
    "should_retry": false
}

Step: "Calculate mean intensity"
Result: {"mean_intensity": -50.5}
Context: Pixel range is [0, 255]
Verification: {
    "passed": false,
    "confidence": 0.9,
    "issues": ["Mean intensity is negative, but pixel range is [0, 255]"],
    "suggestions": ["Check if normalization was applied incorrectly", "Verify input data preprocessing"],
    "should_retry": true
}

Step: "Segment cells"
Result: {"num_cells": 0, "segmentation_map": "array"}
Context: Image shows clear cell structures
Verification: {
    "passed": false,
    "confidence": 0.85,
    "issues": ["Segmentation found 0 cells, but image should contain cells"],
    "suggestions": ["Try adjusting threshold parameters", "Use different segmentation method"],
    "should_retry": true
}"""

        # Prepare verification prompt
        user_prompt = f"""Task: {task_description}

Execution result:
{json.dumps(result.results, indent=2, default=str)}

Execution time: {result.execution_time:.2f}s
Has figure: {result.figure is not None}

"""

        if expected_output:
            user_prompt += f"Expected output: {expected_output}\n\n"

        if data_context:
            user_prompt += f"\nData context:\n{json.dumps(data_context, indent=2, default=str)}\n"

        user_prompt += "\nPlease verify if this result makes sense."

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=400,
                response_format={"type": "json_object"}
            )

            result_json = json.loads(response.choices[0].message.content)

            verification = VerificationResult(
                passed=result_json.get("passed", True),
                confidence=result_json.get("confidence", 0.5),
                issues=result_json.get("issues", []),
                suggestions=result_json.get("suggestions", []),
                should_retry=result_json.get("should_retry", False)
            )

            status = "✓ PASS" if verification.passed else "✗ FAIL"
            logger.info(f"[VERIFIER] {status} (confidence: {verification.confidence:.2f})")

            if verification.issues:
                for issue in verification.issues:
                    logger.warning(f"[VERIFIER] Issue: {issue}")

            return verification

        except Exception as e:
            logger.error(f"[VERIFIER] Verification failed: {e}")
            # On error, assume it passed (conservative approach)
            return VerificationResult(
                passed=True,
                confidence=0.5,
                issues=[f"Verification error: {str(e)}"],
                suggestions=[],
                should_retry=False
            )


# Singleton instance
_verifier = None


def get_verifier() -> ResultVerifier:
    """Get or create ResultVerifier singleton."""
    global _verifier
    if _verifier is None:
        _verifier = ResultVerifier()
    return _verifier
