"""
Docker-based code executor for secure, isolated execution.

Executes generated Python code in Docker containers with:
- Full container isolation
- Volume mounts for data access (no upload needed)
- Resource limits (CPU, memory)
- Network isolation
- Git-tracked capabilities with exact environment reproducibility
"""

import logging
import json
import tempfile
from pathlib import Path
from typing import Optional, List
import time

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

from core.data_models import ExecutionResult
import config

logger = logging.getLogger(__name__)


class DockerExecutor:
    """Execute Python code in Docker containers."""

    def __init__(
        self,
        base_image: str = None,
        memory_limit: str = "2g",
        cpu_quota: int = 100000,
        network_disabled: bool = True
    ):
        """
        Initialize Docker executor.

        Args:
            base_image: Base Docker image to use (default from config)
            memory_limit: Container memory limit (e.g., "2g")
            cpu_quota: CPU quota (100000 = 1 CPU)
            network_disabled: Disable network access for security
        """
        if not DOCKER_AVAILABLE:
            raise ImportError(
                "Docker SDK not installed. Install with: pip install docker"
            )

        try:
            self.client = docker.from_env()
            self.client.ping()
        except Exception as e:
            raise RuntimeError(f"Docker daemon not available: {e}")

        self.base_image = base_image or config.DOCKER_BASE_IMAGE
        self.memory_limit = memory_limit
        self.cpu_quota = cpu_quota
        self.network_disabled = network_disabled

        logger.info(f"Docker executor initialized with image: {self.base_image}")

    def execute(
        self,
        code: str,
        images_path: str,
        output_path: str = None,
        timeout: int = 300,
        custom_image: str = None,
        custom_packages: List[str] = None
    ) -> ExecutionResult:
        """
        Execute code in Docker container.

        Args:
            code: Python code to execute
            images_path: Path to image data (mounted read-only)
            output_path: Path for outputs (mounted read-write)
            timeout: Execution timeout in seconds
            custom_image: Use custom image instead of base
            custom_packages: Additional packages to install (pip install)

        Returns:
            ExecutionResult with success status and outputs
        """
        start_time = time.time()

        # Choose image
        image = custom_image or self.base_image

        # Verify image exists
        try:
            self.client.images.get(image)
        except docker.errors.ImageNotFound:
            return ExecutionResult(
                success=False,
                error_message=f"Docker image '{image}' not found. Run: docker build -f docker/Dockerfile.calcium_imaging -t {image} .",
                execution_time=time.time() - start_time
            )

        # Create temporary output directory if not provided
        if output_path is None:
            output_path = tempfile.mkdtemp(prefix="calcium_output_")
        else:
            Path(output_path).mkdir(parents=True, exist_ok=True)

        # Build execution script
        script_content = self._build_script(code, custom_packages)

        # Create temporary script file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            script_path = f.name

        try:
            # Set up volume mounts
            volumes = {
                str(Path(images_path).absolute()): {
                    "bind": "/data/images",
                    "mode": "ro"  # Read-only
                },
                str(Path(output_path).absolute()): {
                    "bind": "/workspace/outputs",
                    "mode": "rw"  # Read-write
                },
                script_path: {
                    "bind": "/workspace/execute.py",
                    "mode": "ro"
                }
            }

            logger.info(f"Starting Docker container with image: {image}")
            logger.debug(f"Volume mounts: {volumes}")

            # Run container (don't remove yet - need to get logs first)
            container = self.client.containers.run(
                image=image,
                command=["python", "/workspace/execute.py"],
                volumes=volumes,
                detach=True,
                remove=False,  # Don't auto-remove, we'll remove manually after getting logs
                mem_limit=self.memory_limit,
                cpu_quota=self.cpu_quota,
                network_disabled=self.network_disabled,
                user="calcium",  # Non-root user
                working_dir="/workspace"
            )

            # Wait for completion with timeout
            result = container.wait(timeout=timeout)
            execution_time = time.time() - start_time

            # Get container logs before removing
            try:
                logs = container.logs(stdout=True, stderr=True).decode('utf-8')
                stdout = logs
                stderr = ""
            except Exception as log_error:
                logger.warning(f"Could not retrieve container logs: {log_error}")
                stdout = ""
                stderr = str(log_error)

            # Now remove the container
            try:
                container.remove()
            except Exception as e:
                logger.warning(f"Could not remove container: {e}")

            # Parse results from output directory
            return self._parse_results(
                output_path=output_path,
                container_result=result,
                stdout=stdout,
                stderr=stderr,
                execution_time=execution_time
            )

        except docker.errors.ContainerError as e:
            return ExecutionResult(
                success=False,
                error_message=f"Container error: {e}",
                execution_time=time.time() - start_time
            )

        except Exception as e:
            logger.error(f"Docker execution failed: {e}")
            return ExecutionResult(
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )

        finally:
            # Cleanup temporary script
            try:
                Path(script_path).unlink()
            except Exception:
                pass

    def _build_script(self, user_code: str, custom_packages: List[str] = None) -> str:
        """
        Build complete execution script with load_images() function.

        Args:
            user_code: User's Python code
            custom_packages: Additional packages to install

        Returns:
            Complete Python script
        """
        # Install custom packages if specified
        install_section = ""
        if custom_packages:
            packages = " ".join(custom_packages)
            install_section = f"""
# Install custom packages
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", {packages!r}])
"""

        script = f"""
import sys
import json
import traceback
from pathlib import Path
import numpy as np
from PIL import Image

{install_section}

def load_images(max_frames=None, frame_indices=None):
    \"\"\"
    Load calcium imaging data from mounted volume.

    Args:
        max_frames: Optional limit on number of frames to load
        frame_indices: Optional list of specific frame indices to load

    Returns:
        numpy.ndarray of shape (T, H, W), dtype float32, values in [0, 1]
    \"\"\"
    # Data is mounted at /data/images
    images_dir = Path("/data/images")

    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {{images_dir}}")

    # List all image files (sorted)
    image_files = sorted(images_dir.glob("*.png"))

    if not image_files:
        raise ValueError(f"No PNG images found in {{images_dir}}")

    print(f"Found {{len(image_files)}} total images")

    # Apply frame selection
    if frame_indices is not None:
        if not isinstance(frame_indices, (list, tuple)):
            frame_indices = [frame_indices]
        image_files = [image_files[i] for i in frame_indices if i < len(image_files)]
        print(f"Loading {{len(image_files)}} specific frames: {{frame_indices}}")
    elif max_frames is not None:
        image_files = image_files[:max_frames]
        print(f"Loading first {{len(image_files)}} frames")
    else:
        print(f"Loading all {{len(image_files)}} frames")

    # Load images
    images = []
    for i, img_path in enumerate(image_files):
        img = np.array(Image.open(img_path))
        images.append(img)

        if (i + 1) % 10 == 0:
            print(f"Loaded {{i + 1}}/{{len(image_files)}} images")

    images_array = np.array(images, dtype=np.float32)

    # Determine bit depth and normalize appropriately
    max_val = images_array.max()

    if max_val > 255:
        # 16-bit or 12-bit image
        if max_val > 4095:
            bit_depth = 16
            max_possible = 65535.0
        else:
            bit_depth = 12
            max_possible = 4095.0
        print(f"Detected {{bit_depth}}-bit image (max value: {{max_val:.0f}})")

        # For low-light microscopy, use percentile-based normalization
        # This enhances contrast for dim calcium imaging data
        p1, p99 = np.percentile(images_array, [1, 99])
        print(f"Intensity range: 1st percentile={{p1:.0f}}, 99th percentile={{p99:.0f}}")

        if p99 > p1:
            # Percentile-based contrast stretching
            images_array = np.clip((images_array - p1) / (p99 - p1), 0, 1)
            print(f"Applied percentile normalization (1-99%) for contrast enhancement")
        else:
            # Fallback: simple normalization
            images_array = images_array / max_possible
            print(f"Applied simple normalization (/ {{max_possible}})")
    elif max_val > 1.0:
        # 8-bit image
        images_array = images_array / 255.0
        print(f"Detected 8-bit image, normalized by 255")
    else:
        print(f"Image already normalized to [0, 1]")

    print(f"Loaded images array: shape={{images_array.shape}}, dtype={{images_array.dtype}}, range=[{{images_array.min():.4f}}, {{images_array.max():.4f}}]")

    return images_array


# Initialize results and figure
results = None
figure = None

try:
    # Execute user code
{chr(10).join('    ' + line for line in user_code.split(chr(10)))}

    # Save results
    output_dir = Path("/workspace/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results as JSON
    if results is not None:
        results_path = output_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {{results_path}}")

    # Save figure if created
    if figure is not None:
        import matplotlib.pyplot as plt
        figure_path = output_dir / "figure.png"
        figure.savefig(figure_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {{figure_path}}")
        plt.close(figure)

    print("\\n=== EXECUTION SUCCESS ===")
    sys.exit(0)

except Exception as e:
    print(f"\\n=== EXECUTION FAILED ===", file=sys.stderr)
    print(f"Error: {{e}}", file=sys.stderr)
    print(f"\\nTraceback:", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
"""
        return script

    def _parse_results(
        self,
        output_path: str,
        container_result: dict,
        stdout: str,
        stderr: str,
        execution_time: float
    ) -> ExecutionResult:
        """
        Parse execution results from container output.

        Args:
            output_path: Path to output directory
            container_result: Container exit result
            stdout: Container stdout
            stderr: Container stderr
            execution_time: Execution time in seconds

        Returns:
            ExecutionResult
        """
        output_dir = Path(output_path)

        # Check exit code
        exit_code = container_result.get("StatusCode", 1)
        success = exit_code == 0

        # Load results JSON if exists
        results = None
        results_path = output_dir / "results.json"
        if results_path.exists():
            try:
                with open(results_path) as f:
                    results = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load results JSON: {e}")

        # Check for figure
        figure_path = output_dir / "figure.png"
        figure = str(figure_path) if figure_path.exists() else None

        # Extract error from stderr if failed
        error_message = ""
        if not success:
            error_message = stderr if stderr else stdout if stdout else "Execution failed (no error message)"

        return ExecutionResult(
            success=success,
            results=results,
            figure=figure,
            error_message=error_message,
            execution_time=execution_time
        )

    def build_base_image(self, dockerfile_path: str = None):
        """
        Build base Docker image for calcium imaging.

        Args:
            dockerfile_path: Path to Dockerfile (default: docker/Dockerfile.calcium_imaging)
        """
        if dockerfile_path is None:
            dockerfile_path = Path(__file__).parent.parent / "docker" / "Dockerfile.calcium_imaging"

        if not Path(dockerfile_path).exists():
            raise FileNotFoundError(f"Dockerfile not found: {dockerfile_path}")

        logger.info(f"Building Docker image: {self.base_image}")

        try:
            image, logs = self.client.images.build(
                path=str(Path(dockerfile_path).parent),
                dockerfile=str(Path(dockerfile_path).name),
                tag=self.base_image,
                rm=True
            )

            for log in logs:
                if 'stream' in log:
                    print(log['stream'].strip())

            logger.info(f"Successfully built image: {self.base_image}")
            return image

        except Exception as e:
            logger.error(f"Failed to build Docker image: {e}")
            raise


def execute_code_docker(
    code: str,
    images_path: str,
    output_path: str = None,
    timeout: int = 300,
    docker_image: str = None,
    custom_packages: List[str] = None
) -> ExecutionResult:
    """
    Convenience function to execute code in Docker container.

    Args:
        code: Python code to execute
        images_path: Path to calcium imaging data
        output_path: Path for outputs (optional)
        timeout: Execution timeout in seconds
        docker_image: Custom Docker image (optional)
        custom_packages: Additional pip packages (optional)

    Returns:
        ExecutionResult
    """
    executor = DockerExecutor()
    return executor.execute(
        code=code,
        images_path=images_path,
        output_path=output_path,
        timeout=timeout,
        custom_image=docker_image,
        custom_packages=custom_packages
    )


# Singleton instance
_executor_instance = None


def get_docker_executor() -> DockerExecutor:
    """Get or create Docker executor singleton."""
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = DockerExecutor()
    return _executor_instance
