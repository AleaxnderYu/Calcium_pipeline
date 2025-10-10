"""
Script to generate synthetic calcium imaging data for testing.
Creates PNG sequences simulating cells with calcium transients.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import scipy.ndimage as ndi


def generate_mock_calcium_images(
    output_dir: Path,
    n_frames: int = 10,
    image_size: int = 128,
    n_cells: int = 15,
    cell_radius: int = 4,
    noise_level: float = 0.05
):
    """
    Generate synthetic calcium imaging data.

    Args:
        output_dir: Directory to save PNG files
        n_frames: Number of time frames
        image_size: Image dimensions (square)
        n_cells: Number of simulated cells
        cell_radius: Approximate radius of each cell in pixels
        noise_level: Standard deviation of Gaussian noise
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate random cell positions
    np.random.seed(42)  # For reproducibility
    cell_positions = []
    margin = cell_radius + 5

    for _ in range(n_cells):
        x = np.random.randint(margin, image_size - margin)
        y = np.random.randint(margin, image_size - margin)
        cell_positions.append((x, y))

    # Generate temporal profiles for each cell (some have transients, some don't)
    temporal_profiles = []
    for i in range(n_cells):
        # Baseline intensity
        baseline = np.random.uniform(0.3, 0.5)
        profile = np.ones(n_frames) * baseline

        # Add transient to some cells
        if np.random.rand() > 0.3:  # 70% of cells have transients
            # Transient parameters
            onset = np.random.randint(2, n_frames - 4)
            amplitude = np.random.uniform(0.2, 0.5)
            tau_rise = np.random.uniform(1.0, 2.0)
            tau_decay = np.random.uniform(2.0, 4.0)

            # Generate transient waveform
            for t in range(n_frames):
                if t >= onset:
                    time_since_onset = t - onset
                    # Rise phase
                    if time_since_onset < tau_rise:
                        profile[t] = baseline + amplitude * (time_since_onset / tau_rise)
                    # Decay phase
                    else:
                        decay_time = time_since_onset - tau_rise
                        profile[t] = baseline + amplitude * np.exp(-decay_time / tau_decay)

        temporal_profiles.append(profile)

    # Generate frames
    for frame_idx in range(n_frames):
        # Create blank frame
        frame = np.zeros((image_size, image_size), dtype=np.float32)

        # Add each cell
        for cell_idx, (x, y) in enumerate(cell_positions):
            # Create Gaussian blob
            y_coords, x_coords = np.ogrid[:image_size, :image_size]
            distance = np.sqrt((x_coords - x)**2 + (y_coords - y)**2)
            cell_blob = np.exp(-(distance**2) / (2 * cell_radius**2))

            # Scale by temporal profile
            intensity = temporal_profiles[cell_idx][frame_idx]
            frame += cell_blob * intensity

        # Add Gaussian noise
        frame += np.random.normal(0, noise_level, frame.shape)

        # Clip to valid range [0, 1]
        frame = np.clip(frame, 0, 1)

        # Convert to 8-bit and save
        frame_8bit = (frame * 255).astype(np.uint8)
        img = Image.fromarray(frame_8bit, mode='L')

        # Save with zero-padded filename
        filename = f"frame_{frame_idx+1:03d}.png"
        img.save(output_dir / filename)

    print(f"Generated {n_frames} frames with {n_cells} cells in {output_dir}")


if __name__ == "__main__":
    # Generate mock images
    output_path = Path(__file__).parent.parent / "data" / "images"
    generate_mock_calcium_images(output_path)
    print("Mock calcium imaging data generated successfully!")
