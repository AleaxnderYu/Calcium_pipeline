"""
Tool-Based Planner: Creates execution plans using tool calls.
"""

import logging
from typing import List
from openai import OpenAI
import json
import uuid
from datetime import datetime
import config
from core.data_models import ToolBasedPlan, ToolCall, ToolType, StepStatus

logger = logging.getLogger(__name__)


class ToolBasedPlanner:
    """Plans execution using explicit tool calls (RAG, code gen, execute, verify)."""

    def __init__(self):
        """Initialize the planner with OpenAI client."""
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.ROUTER_MODEL  # Use GPT-3.5-turbo for planning

    def create_plan(self, user_request: str) -> ToolBasedPlan:
        """
        Create tool-based execution plan from user request.

        Args:
            user_request: Original user request

        Returns:
            ToolBasedPlan with tool calls and dependencies
        """
        logger.info(f"[TOOL PLANNER] Creating execution plan for: '{user_request}'")

        system_prompt = """You are an execution planner for calcium imaging analysis.

Your job is to decompose user requests into a sequence of TOOL CALLS.

Available tools:
1. **rag**: Retrieve scientific methods from papers (ENHANCED with section-based chunking)
   - Use when: Need domain knowledge about calcium imaging methods
   - Inputs:
     * "query": "search query" (required)
     * "top_k": number of sections to retrieve (REQUIRED - you must decide!)
     * "return_full_papers": true/false - return full paper if multiple sections match (default: true)
     * "multi_section_threshold": min sections from same paper to return full (default: 2)
   - Outputs: {"chunks": [...], "sources": [...], "metadata": {...}}
   - **How to choose top_k:**
     * Quick fact lookup (single definition, concept): top_k=3
     * Specific method details (one technique): top_k=5-8
     * Comprehensive research methods (full methodology): top_k=10-15
     * Deep analysis (compare multiple approaches): top_k=15-20
     * Full paper content needed: top_k=20-30
   - **How to use return_full_papers:**
     * When you need complete methodology: "return_full_papers": true (default)
     * When you only need specific sections: "return_full_papers": false
   - **Think about query complexity and adjust top_k accordingly!**

2. **code_generation**: Generate Python code for analysis
   - Use when: Need to write code for data processing/analysis
   - Inputs: {"task_description": "...", "rag_context": {...}}
   - Outputs: {"code": "...", "description": "..."}

3. **execute**: Execute Python code in Docker sandbox
   - Use when: Need to run generated code
   - Inputs: {"code": "...", "images_path": "..."}
   - Outputs: {"results": {...}, "figures": [...]}

4. **verify**: Verify execution results make sense
   - Use when: After code execution (always)
   - Inputs: {"execution_result": {...}, "expected_output": "..."}
   - Outputs: {"passed": true/false, "issues": [...]}

5. **capability_search**: Search for reusable code
   - Use when: Task might have been done before
   - Inputs: {"query": "..."}
   - Outputs: {"found": true/false, "capability_id": "...", "code": "..."}

6. **capability_save**: Save successful code for reuse
   - Use when: After successful execution
   - Inputs: {"code": "...", "description": "..."}
   - Outputs: {"capability_id": "..."}

Respond in JSON format:
{
    "assumptions": ["List of assumptions made"],
    "execution_mode": "sequential" or "parallel" or "dag",
    "tool_calls": [
        {
            "tool_id": "t1",
            "tool_type": "rag",
            "description": "Retrieve methods for cell segmentation",
            "inputs": {"query": "calcium imaging cell segmentation methods"},
            "depends_on": []
        },
        {
            "tool_id": "t2",
            "tool_type": "code_generation",
            "description": "Generate cell segmentation code",
            "inputs": {
                "task_description": "Segment cells using watershed",
                "rag_context": "$t1.output"
            },
            "depends_on": ["t1"]
        },
        {
            "tool_id": "t3",
            "tool_type": "execute",
            "description": "Run segmentation code",
            "inputs": {
                "code": "$t2.output.code",
                "images_path": "$user.images_path"
            },
            "depends_on": ["t2"]
        },
        {
            "tool_id": "t4",
            "tool_type": "verify",
            "description": "Verify segmentation results",
            "inputs": {
                "execution_result": "$t3.output",
                "expected_output": "segmentation map with cell count"
            },
            "depends_on": ["t3"]
        }
    ]
}

Guidelines:
- **Identify query type first:**
  * **Informational queries** (asking about papers, methods, concepts): ONLY use RAG, no code
  * **Analysis tasks** (count cells, measure intensity, etc.): Use RAG → code_gen → execute → verify
- **For informational queries**: Just use RAG tool, return the information
  * Example: "What are the methods in paper X?" → Only RAG tool
  * Example: "Explain calcium imaging" → Only RAG tool
  * Example: "What is OASIS algorithm?" → Only RAG tool
- **For analysis tasks**: Full pipeline
  * Example: "Count cells" → RAG → code_gen → execute → verify
  * Example: "Measure fluorescence" → RAG → code_gen → execute → verify
- Use sequential mode for dependent tasks (most common)
- Use parallel mode when tools can run simultaneously
- Use DAG mode for complex dependencies
- Reference previous tool outputs with "$tool_id.output.field"

Examples:

Request: "Count cells in the image"
{
    "assumptions": [
        "Use watershed segmentation",
        "Process first frame if time series",
        "Count connected components as cells"
    ],
    "execution_mode": "sequential",
    "tool_calls": [
        {
            "tool_id": "t1",
            "tool_type": "rag",
            "description": "Retrieve cell segmentation and counting methods",
            "inputs": {"query": "cell segmentation and counting in calcium imaging", "top_k": 8},
            "depends_on": []
        },
        {
            "tool_id": "t2",
            "tool_type": "code_generation",
            "description": "Generate code to segment and count cells",
            "inputs": {
                "task_description": "Segment cells using watershed and count them",
                "rag_context": "$t1.output"
            },
            "depends_on": ["t1"]
        },
        {
            "tool_id": "t3",
            "tool_type": "execute",
            "description": "Execute segmentation and counting",
            "inputs": {
                "code": "$t2.output.code",
                "images_path": "$user.images_path"
            },
            "depends_on": ["t2"]
        },
        {
            "tool_id": "t4",
            "tool_type": "verify",
            "description": "Verify cell count is reasonable",
            "inputs": {
                "execution_result": "$t3.output",
                "expected_output": "positive integer cell count"
            },
            "depends_on": ["t3"]
        },
        {
            "tool_id": "t5",
            "tool_type": "capability_save",
            "description": "Save cell counting capability",
            "inputs": {
                "code": "$t2.output.code",
                "description": "Cell segmentation and counting using watershed"
            },
            "depends_on": ["t4"]
        }
    ]
}

Request: "Show me the first frame"
{
    "assumptions": ["Display first time point"],
    "execution_mode": "sequential",
    "tool_calls": [
        {
            "tool_id": "t1",
            "tool_type": "code_generation",
            "description": "Generate code to load and display first frame",
            "inputs": {
                "task_description": "Load images and display the first frame",
                "rag_context": null
            },
            "depends_on": []
        },
        {
            "tool_id": "t2",
            "tool_type": "execute",
            "description": "Display first frame",
            "inputs": {
                "code": "$t1.output.code",
                "images_path": "$user.images_path"
            },
            "depends_on": ["t1"]
        }
    ]
}

Request: "What are calcium transients?"
{
    "assumptions": ["User wants basic definition, not in-depth analysis"],
    "execution_mode": "sequential",
    "tool_calls": [
        {
            "tool_id": "t1",
            "tool_type": "rag",
            "description": "Retrieve basic information about calcium transients",
            "inputs": {"query": "what are calcium transients definition mechanism", "top_k": 3},
            "depends_on": []
        }
    ]
}

Request: "What are the research methods used in the paper about RNF13 and lysosomal positioning?"
{
    "assumptions": ["User wants comprehensive methodology from specific paper"],
    "execution_mode": "sequential",
    "tool_calls": [
        {
            "tool_id": "t1",
            "tool_type": "rag",
            "description": "Retrieve comprehensive research methods from RNF13 paper",
            "inputs": {"query": "RNF13 lysosomal positioning research methods experimental design", "top_k": 15, "return_full_papers": true},
            "depends_on": []
        }
    ]
}
"""

        user_prompt = f"Request: {user_request}\n\nCreate an execution plan with tool calls."

        try:
            # Combine system and user prompts for Responses API
            combined_input = f"{system_prompt}\n\n{user_prompt}"

            # Use responses.create() for gpt-5
            logger.debug(f"[TOOL PLANNER] Calling gpt-5 with model={self.model}, input_length={len(combined_input)}")
            response = self.client.responses.create(
                model=self.model,
                input=combined_input,
                max_output_tokens=1500
            )

            logger.debug(f"[TOOL PLANNER] Response object: {response}")
            logger.debug(f"[TOOL PLANNER] Response type: {type(response)}")
            logger.debug(f"[TOOL PLANNER] Response dict: {response.model_dump() if hasattr(response, 'model_dump') else 'N/A'}")

            # Extract text from the response object
            # The response has an 'output' array with reasoning and message items
            raw_content = None
            if hasattr(response, 'output') and response.output:
                for item in response.output:
                    if hasattr(item, 'type') and item.type == 'message':
                        if hasattr(item, 'content') and item.content:
                            for content_item in item.content:
                                if hasattr(content_item, 'text'):
                                    raw_content = content_item.text
                                    break
                        break

            # Fallback to output_text if it exists
            if not raw_content and hasattr(response, 'output_text'):
                raw_content = response.output_text

            logger.debug(f"[TOOL PLANNER] raw_content type: {type(raw_content)}, length: {len(raw_content) if raw_content else 0}")
            logger.debug(f"[TOOL PLANNER] raw_content value: {raw_content}")

            # Extract JSON from markdown code blocks if present
            json_content = raw_content.strip()
            if "```json" in json_content:
                json_content = json_content.split("```json")[1].split("```")[0].strip()
            elif "```" in json_content:
                json_content = json_content.split("```")[1].split("```")[0].strip()

            result_json = json.loads(json_content)

            # Create tool calls
            tool_calls = []
            for tool_data in result_json.get("tool_calls", []):
                tool_call = ToolCall(
                    tool_id=tool_data.get("tool_id", f"t{len(tool_calls)+1}"),
                    tool_type=ToolType(tool_data.get("tool_type")),
                    description=tool_data.get("description", ""),
                    inputs=tool_data.get("inputs", {}),
                    depends_on=tool_data.get("depends_on", [])
                )
                tool_calls.append(tool_call)

            # Create plan
            plan = ToolBasedPlan(
                plan_id=f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
                original_request=user_request,
                tool_calls=tool_calls,
                execution_mode=result_json.get("execution_mode", "sequential"),
                assumptions=result_json.get("assumptions", [])
            )

            logger.info(f"[TOOL PLANNER] ✓ Created plan with {len(tool_calls)} tool calls ({plan.execution_mode} mode)")
            for i, tc in enumerate(tool_calls, 1):
                logger.info(f"  {i}. [{tc.tool_type.value}] {tc.description}")

            return plan

        except Exception as e:
            logger.error(f"[TOOL PLANNER] Planning failed: {e}")
            # Return minimal fallback plan - just use RAG for safety
            # (Better to retrieve info than try to generate code blindly)
            # Determine appropriate top_k based on query length/complexity
            query_length = len(user_request.split())
            if query_length <= 5:
                top_k = 3  # Simple query
            elif query_length <= 15:
                top_k = 8  # Medium query
            else:
                top_k = 15  # Complex/comprehensive query

            return ToolBasedPlan(
                plan_id=f"plan_fallback_{uuid.uuid4().hex[:8]}",
                original_request=user_request,
                tool_calls=[
                    ToolCall(
                        tool_id="t1",
                        tool_type=ToolType.RAG,
                        description="Retrieve relevant information",
                        inputs={"query": user_request, "top_k": top_k},
                        depends_on=[]
                    )
                ],
                execution_mode="sequential",
                assumptions=[f"Fallback plan due to planning error - using RAG with top_k={top_k}"]
            )


# Singleton instance
_tool_planner = None


def get_tool_planner() -> ToolBasedPlanner:
    """Get or create ToolBasedPlanner singleton."""
    global _tool_planner
    if _tool_planner is None:
        _tool_planner = ToolBasedPlanner()
    return _tool_planner
