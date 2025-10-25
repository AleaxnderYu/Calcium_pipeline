"""
Preprocessor: Hard-coded data loading and normalization for calcium imaging.
"""

import logging
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image

from core.data_models import PreprocessedData

logger = logging.getLogger(__name__)


class Preprocessor:
    """Preprocessor for loading and normalizing calcium imaging PNG sequences."""

    def process(self, images_path: str, max_frames: int = None, frame_indices: List[int] = None) -> PreprocessedData:
        """
        Load and preprocess a sequence of PNG images.

        Args:
            images_path: Path to directory containing PNG frames
            max_frames: Maximum number of frames to load (default: load all)
            frame_indices: Specific frame indices to load (overrides max_frames)

        Returns:
            PreprocessedData object with normalized images and metadata

        Raises:
            ValueError: If directory is empty, doesn't exist, or images have inconsistent dimensions
        """
        images_dir = Path(images_path)

        # Validate directory exists
        if not images_dir.exists():
            raise ValueError(f"Images directory does not exist: {images_path}")
        if not images_dir.is_dir():
            raise ValueError(f"Path is not a directory: {images_path}")

        # Discover PNG files
        all_png_files = sorted(images_dir.glob("*.png"))
        if not all_png_files:
            raise ValueError(f"No PNG files found in directory: {images_path}")

        # Select frames based on parameters
        if frame_indices is not None:
            # Load specific frames
            png_files = [all_png_files[i] for i in frame_indices if i < len(all_png_files)]
            logger.info(f"Loading {len(png_files)} specific frames (indices: {frame_indices[:10]}{'...' if len(frame_indices) > 10 else ''}) from {images_path}")
        elif max_frames is not None:
            # Limit number of frames
            png_files = all_png_files[:max_frames]
            logger.info(f"Loading first {len(png_files)} frames (out of {len(all_png_files)} total) from {images_path}")
        else:
            # Load all frames
            png_files = all_png_files
            logger.info(f"Loading all {len(png_files)} frames from {images_path}")

        # Load images
        loaded_images = []
        dimensions = None

        for png_file in png_files:
            try:
                # Load image
                img = Image.open(png_file)

                # Convert to grayscale if needed
                if img.mode != 'L':
                    img = img.convert('L')

                # Convert to numpy array
                img_array = np.array(img, dtype=np.float32)

                # Check dimensions consistency
                if dimensions is None:
                    dimensions = img_array.shape
                elif img_array.shape != dimensions:
                    raise ValueError(
                        f"Image {png_file.name} has shape {img_array.shape}, "
                        f"expected {dimensions}"
                    )

                loaded_images.append(img_array)

            except Exception as e:
                logger.warning(f"Skipped corrupted image: {png_file.name} - {str(e)}")
                continue

        if not loaded_images:
            raise ValueError("No valid images could be loaded")

        # Stack into 3D array (T × H × W)
        images = np.stack(loaded_images, axis=0)

        # Normalize to [0, 1] range
        images = images / 255.0

        # Extract metadata
        metadata = {
            'n_frames': images.shape[0],
            'height': images.shape[1],
            'width': images.shape[2],
            'normalized': True,
            'pixel_range': (float(images.min()), float(images.max())),
            'source_path': str(images_path)
        }

        logger.info(
            f"Preprocessed data: {metadata['n_frames']} frames, "
            f"{metadata['height']}×{metadata['width']} pixels"
        )

        return PreprocessedData(images=images, metadata=metadata)
