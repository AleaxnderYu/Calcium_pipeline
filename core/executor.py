"""
Code Executor: Safely execute generated Python code with timeout and restrictions.
"""

import logging
import time
import signal
import threading
from typing import Dict, Any
import numpy as np
import scipy
import scipy.signal
import scipy.ndimage
import matplotlib
import matplotlib.pyplot as plt
import skimage
import skimage.measure
import skimage.segmentation
import skimage.filters
import skimage.feature
import skimage.morphology

from core.data_models import ExecutionResult
import config

logger = logging.getLogger(__name__)

# Use non-interactive backend to prevent GUI windows
matplotlib.use('Agg')


class TimeoutError(Exception):
    """Raised when code execution times out."""
    pass


def _timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Code execution timeout")


def execute_code(
    code: str,
    images: np.ndarray,
    timeout: int = None
) -> ExecutionResult:
    """
    Execute generated Python code in a restricted namespace.

    Args:
        code: Python code to execute
        images: Input image data (T×H×W numpy array)
        timeout: Timeout in seconds (defaults to config.CODE_TIMEOUT_SECONDS)

    Returns:
        ExecutionResult with success status and results
    """
    timeout = timeout or config.CODE_TIMEOUT_SECONDS
    n_lines = len(code.split('\n'))
    logger.info(f"Executing generated code ({n_lines} lines)")

    # Create restricted namespace
    namespace = _create_namespace(images)

    start_time = time.time()

    try:
        # Set up timeout (Unix-like systems)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)

            try:
                exec(code, namespace)
            finally:
                signal.alarm(0)  # Cancel alarm
        else:
            # Fallback for Windows (threading-based, less reliable)
            exec_thread = threading.Thread(target=lambda: exec(code, namespace))
            exec_thread.daemon = True
            exec_thread.start()
            exec_thread.join(timeout)

            if exec_thread.is_alive():
                raise TimeoutError("Code execution timeout")

        execution_time = time.time() - start_time

        # Extract results from namespace
        results = namespace.get('results', {})
        figure = namespace.get('figure', None)

        # Validate outputs
        if not isinstance(results, dict):
            raise ValueError(f"'results' must be a dict, got {type(results)}")

        if figure is not None and not isinstance(figure, matplotlib.figure.Figure):
            raise ValueError(f"'figure' must be matplotlib.figure.Figure or None, got {type(figure)}")

        logger.info(f"Execution completed in {execution_time:.2f}s")

        return ExecutionResult(
            success=True,
            results=results,
            figure=figure,
            execution_time=execution_time,
            error_message=""
        )

    except TimeoutError as e:
        execution_time = time.time() - start_time
        error_msg = f"Execution timeout after {timeout}s"
        logger.error(error_msg)
        return ExecutionResult(
            success=False,
            results={},
            figure=None,
            execution_time=execution_time,
            error_message=error_msg
        )

    except SyntaxError as e:
        execution_time = time.time() - start_time
        error_msg = f"Syntax error: {str(e)}"
        logger.error(error_msg)
        return ExecutionResult(
            success=False,
            results={},
            figure=None,
            execution_time=execution_time,
            error_message=error_msg
        )

    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Execution failed: {error_msg}")
        return ExecutionResult(
            success=False,
            results={},
            figure=None,
            execution_time=execution_time,
            error_message=error_msg
        )


def _create_namespace(images: np.ndarray) -> Dict[str, Any]:
    """
    Create a restricted namespace for code execution.

    Args:
        images: Input image data

    Returns:
        Dictionary namespace with allowed modules and variables
    """
    # Import builtins module to get access to all builtins
    import builtins

    # Create a copy of builtins but block dangerous ones
    safe_builtins = {}
    blocked = {'open', 'compile', 'exec', 'eval', 'execfile', 'input'}

    for name in dir(builtins):
        if not name.startswith('_') or name in ['__name__', '__doc__', '__import__']:
            if name not in blocked:
                safe_builtins[name] = getattr(builtins, name)

    # Need to explicitly allow __import__ for numpy/scipy to work
    safe_builtins['__import__'] = __import__

    # Add safe special names
    safe_builtins['True'] = True
    safe_builtins['False'] = False
    safe_builtins['None'] = None

    # Create namespace with allowed modules
    namespace = {
        '__builtins__': safe_builtins,
        'np': np,
        'numpy': np,
        'plt': plt,
        'matplotlib': matplotlib,
        'scipy': scipy,
        'skimage': skimage,
        'images': images,
        'results': {},
        'figure': None,
    }

    return namespace
